"""Headline sentiment scoring.

A deliberately simple lexicon model rather than a transformer. Three reasons:
it runs in microseconds with no model download, its mistakes are inspectable,
and financial headline sentiment is dominated by a small vocabulary of moves,
directions and central-bank verbs that a lexicon captures well.

The output is intentionally treated as weak evidence. Headline sentiment is a
tiebreaker and a risk flag in this engine, never a trade trigger — published
news is in the price before a retail feed sees it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Weights are hand-set on a -1..+1 scale by how strongly the term implies
# direction in financial copy.
BULLISH_TERMS: dict[str, float] = {
    "surge": 0.8, "surges": 0.8, "soar": 0.9, "soars": 0.9, "rally": 0.7,
    "rallies": 0.7, "jump": 0.6, "jumps": 0.6, "gain": 0.5, "gains": 0.5,
    "rise": 0.5, "rises": 0.5, "climb": 0.5, "climbs": 0.5, "advance": 0.4,
    "boost": 0.6, "boosted": 0.6, "outperform": 0.6, "beat": 0.5, "beats": 0.5,
    "upgrade": 0.7, "upgraded": 0.7, "bullish": 0.8, "optimism": 0.6,
    "record high": 0.9, "all-time high": 0.9, "breakout": 0.7, "recovery": 0.5,
    "rebound": 0.6, "rebounds": 0.6, "strengthen": 0.5, "strengthens": 0.5,
    "hawkish": 0.4, "stimulus": 0.5, "inflow": 0.5, "inflows": 0.5,
    "accumulation": 0.5, "adoption": 0.5, "approval": 0.6, "approved": 0.6,
    "partnership": 0.4, "upside": 0.5, "momentum": 0.4, "strong": 0.4,
}

BEARISH_TERMS: dict[str, float] = {
    "plunge": -0.9, "plunges": -0.9, "crash": -1.0, "crashes": -1.0,
    "tumble": -0.8, "tumbles": -0.8, "slump": -0.7, "slumps": -0.7,
    "fall": -0.5, "falls": -0.5, "drop": -0.5, "drops": -0.5,
    "decline": -0.5, "declines": -0.5, "sink": -0.7, "sinks": -0.7,
    "slide": -0.6, "slides": -0.6, "loss": -0.5, "losses": -0.5,
    "downgrade": -0.7, "downgraded": -0.7, "bearish": -0.8, "pessimism": -0.6,
    "record low": -0.9, "selloff": -0.8, "sell-off": -0.8, "correction": -0.5,
    "weaken": -0.5, "weakens": -0.5, "dovish": -0.4, "recession": -0.8,
    "outflow": -0.5, "outflows": -0.5, "liquidation": -0.7, "hack": -0.9,
    "hacked": -0.9, "exploit": -0.8, "ban": -0.8, "banned": -0.8,
    "lawsuit": -0.6, "investigation": -0.5, "fraud": -0.9, "default": -0.9,
    "bankruptcy": -1.0, "collapse": -1.0, "fear": -0.6, "panic": -0.8,
    "warning": -0.5, "warns": -0.5, "risk": -0.3, "concerns": -0.4,
    "downside": -0.5, "weak": -0.4, "miss": -0.5, "misses": -0.5,
}

# Words that flip the polarity of whatever follows them.
NEGATIONS = {"not", "no", "never", "without", "fails", "failed", "unlikely", "denies"}

# Words that scale intensity.
INTENSIFIERS = {
    "very": 1.3, "extremely": 1.6, "sharply": 1.5, "massively": 1.7,
    "slightly": 0.5, "marginally": 0.4, "modestly": 0.6, "significantly": 1.4,
}

# Ticker and name aliases so a headline can be attached to an instrument.
SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "BTCUSDT": ("bitcoin", "btc", "$btc"),
    "ETHUSDT": ("ethereum", "ether", "eth", "$eth"),
    "SOLUSDT": ("solana", "sol"),
    "BNBUSDT": ("binance coin", "bnb"),
    "XRPUSDT": ("ripple", "xrp"),
    "ADAUSDT": ("cardano", "ada"),
    "DOGEUSDT": ("dogecoin", "doge"),
    "AVAXUSDT": ("avalanche", "avax"),
    "LINKUSDT": ("chainlink", "link"),
    "MATICUSDT": ("polygon", "matic"),
    "EURUSD": ("euro", "eur/usd", "eurusd", "ecb", "eurozone"),
    "GBPUSD": ("sterling", "pound", "gbp/usd", "gbpusd", "boe", "bank of england"),
    "USDJPY": ("yen", "usd/jpy", "usdjpy", "boj", "bank of japan"),
    "AUDUSD": ("aussie", "aud/usd", "audusd", "rba"),
    "USDCAD": ("loonie", "usd/cad", "usdcad", "boc"),
    "XAUUSD": ("gold", "bullion", "xau"),
    "XAGUSD": ("silver", "xag"),
    "USOIL": ("oil", "crude", "wti", "brent", "opec"),
    "US500": ("s&p 500", "s&p500", "spx", "sp500"),
    "NAS100": ("nasdaq", "ndx", "nas100"),
    "GER40": ("dax", "german stocks"),
    "JP225": ("nikkei", "japanese stocks"),
}


@dataclass
class SentimentScore:
    """Sentiment for a piece of text."""

    score: float  # -1..+1
    magnitude: float  # unnormalised strength, i.e. how loud the text is
    matched_terms: list[str]
    confidence: float  # 0..1, low when few terms matched

    def label(self) -> str:
        if self.confidence < 0.25:
            return "neutral"
        if self.score > 0.25:
            return "bullish"
        if self.score < -0.25:
            return "bearish"
        return "neutral"


_TOKEN_RE = re.compile(r"[a-z0-9$&/'-]+")


def score_text(text: str) -> SentimentScore:
    """Score a headline, handling negation and intensifiers."""
    if not text:
        return SentimentScore(0.0, 0.0, [], 0.0)

    lowered = text.lower()
    tokens = _TOKEN_RE.findall(lowered)
    matched: list[str] = []
    total = 0.0

    # Multi-word phrases first, since they carry the strongest signal.
    for phrase, weight in list(BULLISH_TERMS.items()) + list(BEARISH_TERMS.items()):
        if " " in phrase and phrase in lowered:
            total += weight
            matched.append(phrase)

    for i, token in enumerate(tokens):
        weight = BULLISH_TERMS.get(token) or BEARISH_TERMS.get(token)
        if weight is None:
            continue

        multiplier = 1.0
        # Look back two tokens for a negation or intensifier.
        for offset in (1, 2):
            if i - offset < 0:
                break
            previous = tokens[i - offset]
            if previous in NEGATIONS:
                multiplier *= -0.8
            elif previous in INTENSIFIERS:
                multiplier *= INTENSIFIERS[previous]

        total += weight * multiplier
        matched.append(token)

    if not matched:
        return SentimentScore(0.0, 0.0, [], 0.0)

    magnitude = abs(total)
    # Squash into -1..1 so a headline stuffed with adjectives cannot dominate.
    normalised = max(-1.0, min(1.0, total / max(len(matched), 1) / 0.8))
    # Confidence grows with the number of matches but saturates quickly.
    confidence = min(1.0, len(matched) / 4.0)

    return SentimentScore(
        score=round(normalised, 4),
        magnitude=round(magnitude, 4),
        matched_terms=matched[:10],
        confidence=round(confidence, 3),
    )


def relevant_symbols(text: str, watchlist: list[str] | None = None) -> list[str]:
    """Which instruments a headline plausibly concerns."""
    lowered = f" {text.lower()} "
    candidates = watchlist or list(SYMBOL_ALIASES)
    hits: list[str] = []

    for symbol in candidates:
        for alias in SYMBOL_ALIASES.get(symbol.upper(), ()):
            # Word-boundary match so "sol" does not fire on "solution".
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered):
                hits.append(symbol)
                break
    return hits


def aggregate(
    items: list[tuple[str, float]], half_life_hours: float = 6.0
) -> SentimentScore:
    """Blend several headlines, weighting recent ones more heavily.

    `items` is a list of (text, age_in_hours).
    """
    if not items:
        return SentimentScore(0.0, 0.0, [], 0.0)

    weighted_sum = 0.0
    weight_total = 0.0
    matched: list[str] = []

    for text, age_hours in items:
        result = score_text(text)
        if result.confidence == 0.0:
            continue
        decay = 0.5 ** (max(age_hours, 0.0) / half_life_hours)
        weight = result.confidence * decay
        weighted_sum += result.score * weight
        weight_total += weight
        matched.extend(result.matched_terms)

    if weight_total == 0.0:
        return SentimentScore(0.0, 0.0, [], 0.0)

    return SentimentScore(
        score=round(weighted_sum / weight_total, 4),
        magnitude=round(weight_total, 4),
        matched_terms=list(dict.fromkeys(matched))[:15],
        confidence=round(min(1.0, weight_total / 3.0), 3),
    )
