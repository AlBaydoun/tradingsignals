"""Signal explanation: deterministic rules plus an optional Claude review."""

from signalforge.reasoning.llm import (
    LLMAssessment,
    ReasoningEngine,
    build_evidence_packet,
)
from signalforge.reasoning.rules import (
    build_warnings,
    describe_evidence,
    market_summary,
    multi_timeframe_agreement,
)

__all__ = [
    "ReasoningEngine",
    "LLMAssessment",
    "build_evidence_packet",
    "describe_evidence",
    "build_warnings",
    "multi_timeframe_agreement",
    "market_summary",
]
