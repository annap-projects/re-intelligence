"""
processor.py - LLM Extraction Engine for the Strategic Market Intelligence Pipeline.

Design Philosophy
─────────────────
Raw text from the Bronze layer is inherently high-noise, low-signal.  This
module uses a structured LLM prompt + Pydantic schema enforcement to:

  1. FILTER noise  - strip irrelevant content (ads, boilerplate, unrelated topics)
  2. CLASSIFY      - assign each article to one of three strategic buckets
  3. EXTRACT       - pull named entities, sentiment, and a concise summary
  4. VALIDATE      - Pydantic guarantees the output schema before Silver insertion

The result is a measurable improvement in the signal-to-noise ratio (S/N ≫ 1)
and a dramatic reduction in time-to-insight (Δt → 0).

Signal Categories:
    Competitor_Strategy  - M&A, market positioning, product launches, leadership changes
    Broker_Dynamics      - Broker consolidation, capacity shifts, client relationship news
    Emerging_Risks       - New perils, nat-cat events, regulatory / ESG risk developments

LLM: Anthropic claude-sonnet-4-5 via structured tool_use (enforced JSON output).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Dict, List, Optional

import anthropic
from anthropic import APIError as AnthropicAPIError
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema - Signal Categories
# ---------------------------------------------------------------------------
class SignalCategory(str, Enum):
    """
    Three strategic intelligence buckets aligned with reinsurance business objectives.

    Competitor_Strategy:
        Tracks competitive landscape shifts - M&A activity, pricing strategy changes,
        new product launches, executive moves, or capital deployment by peers.

    Broker_Dynamics:
        Monitors the broker market - consolidation deals, capacity allocation trends,
        client relationship news, and distribution channel disruptions.

    Emerging_Risks:
        Identifies nascent threats - novel nat-cat perils (wildfire, flood, cyber),
        regulatory changes (Solvency II, ICS 2.0), ESG mandates, and parametric
        trigger innovations.
    """
    COMPETITOR_STRATEGY = "Competitor_Strategy"
    BROKER_DYNAMICS = "Broker_Dynamics"
    EMERGING_RISKS = "Emerging_Risks"


class Sentiment(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"


# ---------------------------------------------------------------------------
# Schema - LLM Output (Pydantic v2)
# ---------------------------------------------------------------------------
class IntelligenceSignal(BaseModel):
    """
    Pydantic schema enforced on every LLM response.

    This model is the contract between the LLM extraction engine and the
    Silver layer.  All fields are validated before database insertion,
    preventing malformed data from polluting downstream analytics.
    """

    is_relevant: bool = Field(
        description="True if the article is relevant to reinsurance market intelligence. "
                    "False = noise, skip this record."
    )
    signal_category: Optional[SignalCategory] = Field(
        default=None,
        description="Strategic bucket. Required when is_relevant=True.",
    )
    summary: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Concise 2-4 sentence summary of the key intelligence. "
                    "Write for a senior reinsurance executive. Required when is_relevant=True.",
    )
    key_entities: List[str] = Field(
        default_factory=list,
        description="Named entities: companies, people, geographies, perils, regulators.",
    )
    sentiment: Optional[Sentiment] = Field(
        default=None,
        description="Overall sentiment of the article relative to the reinsurance industry.",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model's self-assessed confidence in the classification (0.0–1.0).",
    )
    action_required: bool = Field(
        default=False,
        description="True if this signal warrants immediate human review "
                    "(e.g. major M&A announcement, catastrophic event, regulatory sanction).",
    )

    @field_validator("signal_category", mode="before")
    @classmethod
    def require_category_when_relevant(cls, v: Any, info: Any) -> Any:
        # Pydantic v2: info.data contains already-validated fields
        return v  # Cross-field validation handled in model_validator below

    def model_post_init(self, __context: Any) -> None:
        if self.is_relevant and self.signal_category is None:
            raise ValueError("signal_category is required when is_relevant=True")
        if self.is_relevant and not self.summary:
            raise ValueError("summary is required when is_relevant=True")


# ---------------------------------------------------------------------------
# Tool definition for Anthropic tool_use (structured JSON enforcement)
# ---------------------------------------------------------------------------
_EXTRACT_SIGNAL_TOOL: Dict[str, Any] = {
    "name": "extract_intelligence_signal",
    "description": (
        "Extract a structured intelligence signal from a reinsurance news article. "
        "Always call this tool with your analysis — do not respond with plain text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_relevant": {
                "type": "boolean",
                "description": "True if the article is relevant to reinsurance market intelligence.",
            },
            "signal_category": {
                "type": ["string", "null"],
                "enum": [None, "Competitor_Strategy", "Broker_Dynamics", "Emerging_Risks"],
                "description": "Strategic bucket. Required when is_relevant=True.",
            },
            "summary": {
                "type": ["string", "null"],
                "description": "Concise 2-4 sentence executive summary. Required when is_relevant=True.",
            },
            "key_entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Named entities: companies, people, geographies, perils, regulators.",
            },
            "sentiment": {
                "type": ["string", "null"],
                "enum": [None, "POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"],
                "description": "Overall sentiment relative to the reinsurance industry.",
            },
            "confidence_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Self-assessed confidence in the classification (0.0–1.0).",
            },
            "action_required": {
                "type": "boolean",
                "description": (
                    "True if this signal warrants immediate human review "
                    "(major M&A >$500M, catastrophic event, regulatory sanction, credit rating change)."
                ),
            },
        },
        "required": ["is_relevant", "key_entities", "confidence_score", "action_required"],
    },
}

# ---------------------------------------------------------------------------
# System prompt - instructs the LLM on its role and output contract
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """
You are a Senior Market Intelligence Analyst at a global reinsurance company.
Your job is to read raw article text and extract ONLY high-signal intelligence
relevant to the reinsurance industry.

CLASSIFICATION RULES:
- Competitor_Strategy: News about reinsurers (Munich Re, Swiss Re, Hannover Re,
  Gen Re, Berkshire Hathaway Re, Everest Re, RenaissanceRe, etc.), their capital
  actions, M&A, leadership changes, pricing strategy, product launches, or
  financial results.
- Broker_Dynamics: News about reinsurance/insurance brokers (Aon, Marsh, Gallagher,
  Guy Carpenter, Willis Towers Watson, etc.), capacity changes, client moves, or
  market structure shifts.
- Emerging_Risks: New or escalating perils (cyber, climate, nat-cat, pandemic,
  geopolitical), regulatory changes (IAIS, Solvency II, ICS), ESG mandates,
  parametric products, or ILS/cat bond market developments.

NOISE (is_relevant=false):
- General equity market commentary unrelated to insurance
- Sports, politics, entertainment, technology outside insurance
- Press releases about unrelated industries

ACTION_REQUIRED RULES — read these carefully before setting the flag:

Set action_required=TRUE only when ALL of the following hold:
  1. The event is CONFIRMED (not rumoured, not speculative)
  2. The event falls into one of these four categories EXACTLY:
       a) M&A transaction with disclosed deal value >= $500M involving a named reinsurer or broker
       b) A catastrophic physical event (hurricane, earthquake, flood, wildfire, pandemic)
          with estimated insured losses >= $1 billion OR declared a national/regional emergency
       c) A named regulatory sanction, licence revocation, or solvency intervention
          against a reinsurer by a named regulator (e.g. Lloyd's, BaFin, FINMA, IAIS)
       d) A credit rating DOWNGRADE (not affirmation, not outlook change) by Moody's, S&P,
          Fitch, or AM Best on a named reinsurer or major broker

FEW-SHOT EXAMPLES (action_required):
  TRUE  → "Munich Re acquires Torchmark Corp for $6.4 billion" (M&A >= $500M confirmed)
  TRUE  → "Swiss Re downgraded to A- by Moody's" (confirmed downgrade by named agency)
  TRUE  → "Hurricane Milton makes landfall; AIR estimates $18bn insured loss" (cat event >= $1bn)
  TRUE  → "Lloyd's of London suspends Syndicate 1234 licence" (regulatory sanction confirmed)
  FALSE → "Munich Re expected to report record profits next quarter" (speculative, not confirmed)
  FALSE → "Swiss Re outlook changed to Negative by S&P" (outlook change, not a downgrade)
  FALSE → "Catastrophic flooding in Germany causes significant losses" (loss quantum unknown/unconfirmed)
  FALSE → "Aon acquires small InsurTech for undisclosed sum" (deal value unknown, may be < $500M)

CRITICAL OUTPUT RULES:
1. Always call the extract_intelligence_signal tool with your analysis.
2. Keep summary under 200 words, written for a C-suite executive.
3. Set action_required=true ONLY per the explicit rules above - when in doubt, set FALSE.
4. If confidence_score < 0.70, you MUST set action_required=false regardless of content.
5. confidence_score must honestly reflect your certainty about the classification.
""".strip()


# ---------------------------------------------------------------------------
# Processor class
# ---------------------------------------------------------------------------
class IntelligenceProcessor:
    """
    LLM-powered extraction engine.

    Takes raw article text (from Bronze layer), runs it through claude-sonnet-4-5
    with a structured system prompt and tool_use enforcement, and returns a
    validated ``IntelligenceSignal``.

    Usage::

        processor = IntelligenceProcessor()
        signal = processor.extract(
            raw_text="Swiss Re reported Q1 2025 net income of $1.2bn …",
            title="Swiss Re Q1 2025 Results",
            source_url="https://swissre.com/investors/Q1-2025",
        )
        if signal.is_relevant:
            print(signal.signal_category, signal.summary)
    """

    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            logger.warning(
                "ANTHROPIC_API_KEY is not set. LLM extraction will fail. "
                "Set it in your .env file or as an environment variable."
            )
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
        logger.info("IntelligenceProcessor initialised (model=%s).", settings.anthropic_model)

    @retry(
        retry=retry_if_exception_type(AnthropicAPIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_llm(self, user_message: str) -> Dict[str, Any]:
        """
        Call the Anthropic Messages API with tool_use to enforce structured output.

        Returns the parsed tool input dict from the LLM response.
        """
        response = self._client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            system=_SYSTEM_PROMPT,
            tools=[_EXTRACT_SIGNAL_TOOL],
            tool_choice={"type": "any"},  # force the model to call the tool
            messages=[
                {"role": "user", "content": user_message},
            ],
        )

        # Extract the tool_use content block
        tool_use_block = next(
            (block for block in response.content if block.type == "tool_use"),
            None,
        )
        if tool_use_block is None:
            raise ProcessingError(
                f"Model did not call the extraction tool. "
                f"Stop reason: {response.stop_reason}. "
                f"Content: {response.content!r}"
            )

        logger.debug(
            "LLM response received: input_tokens=%s  output_tokens=%s  stop_reason=%s",
            response.usage.input_tokens if response.usage else "?",
            response.usage.output_tokens if response.usage else "?",
            response.stop_reason,
        )
        return tool_use_block.input  # type: ignore[return-value]

    def extract(
        self,
        raw_text: str,
        title: Optional[str] = None,
        source_url: Optional[str] = None,
        bronze_id: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> IntelligenceSignal:
        """
        Extract an intelligence signal from raw article text.

        Args:
            raw_text:    The full or truncated article body.
            title:       Optional article headline (helps LLM context).
            source_url:  Optional URL for audit trail.
            bronze_id:   Bronze layer record ID for lineage tracking.
            source_name: Connector identifier (e.g. ``"rss::Reuters"``).

        Returns:
            Validated ``IntelligenceSignal`` Pydantic model.

        Raises:
            ProcessingError: if the LLM returns unparseable or schema-violating output.
        """
        # Truncate to prevent context overflow (rough 3.5 chars/token estimate)
        max_chars = settings.anthropic_max_tokens * 3
        truncated_text = raw_text[:max_chars]
        if len(raw_text) > max_chars:
            logger.debug(
                "Text truncated from %d to %d chars for bronze_id=%s",
                len(raw_text), max_chars, bronze_id,
            )

        # Build user message
        headline_block = f"HEADLINE: {title}\n\n" if title else ""
        url_block = f"SOURCE URL: {source_url}\n\n" if source_url else ""
        user_message = (
            f"{headline_block}"
            f"{url_block}"
            f"ARTICLE TEXT:\n{truncated_text}"
        )

        logger.debug("Sending to LLM: bronze_id=%s  chars=%d", bronze_id, len(truncated_text))

        try:
            payload = self._call_llm(user_message)
            signal = IntelligenceSignal(**payload)
        except ProcessingError:
            raise
        except Exception as exc:
            raise ProcessingError(f"Signal extraction failed: {exc}") from exc

        logger.info(
            "Extracted signal: bronze_id=%s  relevant=%s  category=%s  confidence=%.2f",
            bronze_id,
            signal.is_relevant,
            signal.signal_category,
            signal.confidence_score,
        )
        return signal

    def extract_batch(
        self,
        bronze_records: List[Dict[str, Any]],
        max_workers: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of Bronze records concurrently using a thread pool,
        returning enriched dicts ready for Silver insertion.

        LLM calls are I/O-bound and independent of each other, making them
        a natural fit for thread-based concurrency.  The Anthropic SDK is
        thread-safe; each call gets its own HTTP request.

        Args:
            bronze_records: List of dicts as returned by ``database.fetch_unprocessed_bronze()``.
            max_workers:    Maximum concurrent LLM calls (default 5).  Increase
                            carefully — Anthropic rate limits apply per API key.

        Returns:
            List of Silver-ready signal dicts (only relevant signals included).
            Order is not guaranteed (futures complete as they resolve).
        """
        silver_signals: List[Dict[str, Any]] = []
        errors: List[tuple[str, str]] = []  # (bronze_id, error_message)

        def _process_one(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Extract one signal; return Silver dict or None (noise/error)."""
            bronze_id = record["id"]
            signal = self.extract(
                raw_text=record.get("raw_text", ""),
                title=record.get("title"),
                source_url=record.get("source_url"),
                bronze_id=bronze_id,
                source_name=record.get("source_name"),
            )
            if not signal.is_relevant:
                logger.debug("bronze_id=%s classified as noise - skipping.", bronze_id)
                return None
            return {
                "bronze_id": bronze_id,
                "source_name": record.get("source_name"),
                "source_url": record.get("source_url"),
                "title": record.get("title"),
                "signal_category": signal.signal_category.value,
                "summary": signal.summary,
                "key_entities": signal.key_entities,
                "sentiment": signal.sentiment.value if signal.sentiment else "NEUTRAL",
                "confidence_score": signal.confidence_score,
                "action_required": signal.action_required,
                "llm_model": settings.anthropic_model,
                "raw_llm_response": signal.model_dump(),
            }

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_id = {
                pool.submit(_process_one, record): record["id"]
                for record in bronze_records
            }
            for future in as_completed(future_to_id):
                bronze_id = future_to_id[future]
                try:
                    result = future.result()
                    if result is not None:
                        silver_signals.append(result)
                except ProcessingError as exc:
                    logger.error("ProcessingError for bronze_id=%s: %s", bronze_id, exc)
                    errors.append((bronze_id, str(exc)[:500]))
                except Exception as exc:
                    logger.error("Unexpected error for bronze_id=%s: %s", bronze_id, exc)
                    errors.append((bronze_id, f"Unexpected: {exc!s}"[:500]))

        # Route failed records to DLQ outside the thread pool
        if errors:
            from database import mark_bronze_processed
            for bronze_id, err_msg in errors:
                mark_bronze_processed(bronze_id, error=err_msg)

        logger.info(
            "Batch complete: %d records processed → %d relevant signals, %d errors (DLQ).",
            len(bronze_records),
            len(silver_signals),
            len(errors),
        )
        return silver_signals


class ProcessingError(RuntimeError):
    """Raised when the LLM extraction engine fails to produce a valid signal."""
