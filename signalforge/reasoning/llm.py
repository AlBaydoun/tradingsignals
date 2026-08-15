"""Optional Claude reasoning layer.

The quantitative engine decides *whether* to trade and *how much*. This layer is
a second opinion on top: it reads the same evidence packet a human analyst would
be handed and answers a narrow set of questions — is there a contradiction the
numbers missed, is the narrative consistent, what would invalidate this trade.

Three hard rules keep it useful rather than dangerous:

1. **It cannot create signals.** It only ever comments on a signal the
   quantitative model already produced.
2. **Its confidence adjustment is clamped.** It can nudge confidence by at most
   `max_confidence_override`; it cannot talk a weak signal into a strong one.
3. **It is never required.** With no API key configured, the engine runs exactly
   as before using the deterministic explanations in `rules.py`.

Structured outputs are used so the response is a validated object rather than
prose the engine has to parse hopefully.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from signalforge.config import ReasoningConfig

log = logging.getLogger(__name__)

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_ANTHROPIC = False


SYSTEM_PROMPT = """You are a risk-focused trading analyst reviewing signals produced by a \
quantitative model. You do not generate trade ideas; you review the ones you are given.

Your job is to find reasons a signal might be wrong that the model's features cannot \
see: contradictions between the technical picture and the news flow, event risk the \
model has under-weighted, structural context that makes the stop placement naive, or \
a narrative that does not hang together.

Ground every assessment in the evidence provided. If the evidence is thin, say so and \
lower your confidence — do not invent supporting detail. You are explicitly permitted, \
and expected, to recommend skipping a trade.

Be concise and concrete. Write for a trader who will read this on a phone before \
risking real money."""


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "assessment": {
            "type": "string",
            "enum": ["endorse", "endorse_with_caution", "reject"],
            "description": "Your overall verdict on taking this trade.",
        },
        "confidence_adjustment": {
            "type": "number",
            "description": (
                "How much to shift the model's confidence, from -0.15 to +0.15. "
                "Negative lowers it. Use 0 when you have no strong view."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": (
                "Two to four sentences explaining the setup and your verdict, "
                "written for a trader on a phone."
            ),
        },
        "key_risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific, concrete risks to this trade. Empty if none.",
        },
        "invalidation": {
            "type": "string",
            "description": "What would prove this trade idea wrong, in one sentence.",
        },
        "news_conflict": {
            "type": "boolean",
            "description": "True if the news flow contradicts the technical direction.",
        },
    },
    "required": [
        "assessment",
        "confidence_adjustment",
        "reasoning",
        "key_risks",
        "invalidation",
        "news_conflict",
    ],
    "additionalProperties": False,
}


@dataclass
class LLMAssessment:
    """The reviewed verdict on a signal."""

    assessment: str
    confidence_adjustment: float
    reasoning: str
    key_risks: list[str] = field(default_factory=list)
    invalidation: str = ""
    news_conflict: bool = False
    used_llm: bool = False
    error: str | None = None

    @property
    def rejected(self) -> bool:
        return self.assessment == "reject"

    def to_dict(self) -> dict:
        return {
            "assessment": self.assessment,
            "confidence_adjustment": self.confidence_adjustment,
            "reasoning": self.reasoning,
            "key_risks": self.key_risks,
            "invalidation": self.invalidation,
            "news_conflict": self.news_conflict,
            "used_llm": self.used_llm,
            "error": self.error,
        }


class ReasoningEngine:
    """Wraps the Claude API call, degrading cleanly when unavailable."""

    def __init__(self, config: ReasoningConfig | None = None):
        self.config = config or ReasoningConfig()
        self._client = None
        self._unavailable_reason: str | None = None

        if not self.config.enabled:
            self._unavailable_reason = "disabled in config"
        elif not HAS_ANTHROPIC:
            self._unavailable_reason = "anthropic package not installed"
        elif not (
            os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
        ):
            self._unavailable_reason = "no ANTHROPIC_API_KEY set"

    @property
    def available(self) -> bool:
        return self._unavailable_reason is None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def _get_client(self):
        if self._client is None:
            self._client = anthropic.Anthropic(timeout=self.config.timeout_seconds)
        return self._client

    def review(
        self, evidence: dict, *, fallback_reasoning: str = ""
    ) -> LLMAssessment:
        """Review one signal's evidence packet.

        Any failure — missing key, network error, malformed response — returns
        the deterministic fallback rather than raising. A trading engine must
        not stop producing signals because an API call timed out.
        """
        if not self.available:
            return LLMAssessment(
                assessment="endorse_with_caution",
                confidence_adjustment=0.0,
                reasoning=fallback_reasoning,
                used_llm=False,
                error=self._unavailable_reason,
            )

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.config.effort,
                    "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
                },
                messages=[
                    {
                        "role": "user",
                        "content": self._build_prompt(evidence),
                    }
                ],
            )

            if response.stop_reason == "refusal":
                return LLMAssessment(
                    "endorse_with_caution", 0.0, fallback_reasoning,
                    used_llm=False, error="model declined to respond",
                )

            text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            payload = json.loads(text)

            # Clamp the adjustment regardless of what came back.
            limit = self.config.max_confidence_override
            adjustment = float(payload.get("confidence_adjustment", 0.0))
            adjustment = max(-limit, min(limit, adjustment))

            return LLMAssessment(
                assessment=payload.get("assessment", "endorse_with_caution"),
                confidence_adjustment=adjustment,
                reasoning=payload.get("reasoning", fallback_reasoning),
                key_risks=list(payload.get("key_risks", [])),
                invalidation=payload.get("invalidation", ""),
                news_conflict=bool(payload.get("news_conflict", False)),
                used_llm=True,
            )

        except Exception as exc:
            log.warning("Reasoning layer failed, using deterministic fallback: %s", exc)
            return LLMAssessment(
                assessment="endorse_with_caution",
                confidence_adjustment=0.0,
                reasoning=fallback_reasoning,
                used_llm=False,
                error=str(exc)[:200],
            )

    @staticmethod
    def _build_prompt(evidence: dict) -> str:
        """Render the evidence packet as the analyst's briefing."""
        return (
            "Review this trade signal.\n\n"
            f"```json\n{json.dumps(evidence, indent=2, default=str)}\n```\n\n"
            "Consider specifically:\n"
            "1. Does the news flow contradict the technical direction?\n"
            "2. Is the stop placement sensible given the market structure "
            "and volatility described?\n"
            "3. Is the historical accuracy figure strong enough to justify "
            "the reward:risk on offer?\n"
            "4. Is there scheduled event risk that makes the timing poor?\n\n"
            "If the measured accuracy is absent or the edge is thin, say so "
            "plainly and reject rather than hedge."
        )

    def summarise_market(self, packet: dict) -> str:
        """An optional narrative overview of the whole run."""
        if not self.available:
            return ""
        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.config.model,
                max_tokens=2000,
                system=(
                    "You are a market strategist writing a short daily briefing "
                    "for one trader. Three sentences maximum. State what the "
                    "market is doing and what it implies for risk-taking today. "
                    "No preamble, no disclaimers."
                ),
                thinking={"type": "adaptive"},
                output_config={"effort": "low"},
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Market state:\n"
                            f"```json\n{json.dumps(packet, indent=2, default=str)}\n```"
                        ),
                    }
                ],
            )
            if response.stop_reason == "refusal":
                return ""
            return next((b.text for b in response.content if b.type == "text"), "")
        except Exception as exc:
            log.warning("Market summary failed: %s", exc)
            return ""


def build_evidence_packet(
    *,
    symbol: str,
    timeframe: str,
    direction: str,
    entry: float,
    stop_loss: float,
    take_profits: list[float],
    reward_risk: float,
    model_confidence: float,
    measured_accuracy: float | None,
    regime: dict,
    top_features: list[tuple[str, float]],
    news_sentiment: float,
    headlines: list[dict],
    event_risk: dict,
    anomaly: dict,
    backtest: dict,
) -> dict:
    """Assemble the compact evidence packet handed to the model.

    Kept small on purpose. Dumping 150 raw feature values in would cost tokens
    and bury the few things that actually inform a judgement.
    """
    return {
        "instrument": {"symbol": symbol, "timeframe": timeframe},
        "proposed_trade": {
            "direction": direction,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profits": take_profits,
            "reward_risk": round(reward_risk, 2),
        },
        "model": {
            "confidence": round(model_confidence, 4),
            "measured_out_of_sample_accuracy": measured_accuracy,
            "break_even_win_rate": round(1.0 / (1.0 + reward_risk), 4),
            "most_influential_features": [
                {"name": name, "importance": round(value, 4)}
                for name, value in top_features[:10]
            ],
        },
        "market_regime": regime,
        "news": {
            "aggregate_sentiment": round(news_sentiment, 3),
            "recent_headlines": headlines[:5],
        },
        "scheduled_events": event_risk,
        "unusual_activity": anomaly,
        "historical_performance": backtest,
    }
