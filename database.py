"""
database.py - DuckDB Medallion Data Lake initialisation.

Implements a three-layer Medallion architecture:

  ┌──────────────────────────────────────────────────────────┐
  │  BRONZE  - Raw, immutable ingestion records (append-only) │
  │  SILVER  - Structured, LLM-validated intelligence signals │
  │  GOLD    - Aggregated KPIs, trend analytics + DLQ view    │
  └──────────────────────────────────────────────────────────┘

The module exposes a single ``get_connection()`` context manager and an
``initialise_lake()`` function that is idempotent (safe to call on every
startup).

Deduplication strategy
──────────────────────
Bronze uses a SHA-256 ``content_hash`` derived from the article URL (preferred)
or the full raw_text body (fallback).  ``INSERT OR IGNORE`` on the UNIQUE
constraint prevents duplicate rows from accumulating across pipeline runs even
after process restarts (which would clear any in-memory ``_seen_urls`` set).

Dead Letter Queue (DLQ)
────────────────────────
Records that fail LLM extraction or Pydantic validation are flagged with a
``processing_error`` message.  The ``gold_dead_letter_queue`` view surfaces
these for analyst review.  ``retry_dlq_record()`` clears the error flag so
the record re-enters the processing queue on the next run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

import duckdb

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL - Bronze Layer
# ---------------------------------------------------------------------------
_BRONZE_DDL = """
CREATE TABLE IF NOT EXISTS bronze_raw_articles (
    id              VARCHAR PRIMARY KEY,        -- UUID
    content_hash    VARCHAR UNIQUE NOT NULL,    -- SHA-256 of url OR raw_text (dedup key)
    source_name     VARCHAR NOT NULL,           -- connector identifier
    source_url      VARCHAR,                    -- original article / API URL
    title           VARCHAR,                    -- headline / title
    raw_text        TEXT,                       -- full raw content
    raw_metadata    JSON,                       -- connector-specific metadata blob
    ingested_at     TIMESTAMPTZ NOT NULL,       -- UTC ingest timestamp
    is_processed    BOOLEAN DEFAULT FALSE,      -- has the silver layer consumed this?
    processing_error VARCHAR                    -- last processing error message (DLQ flag)
);
"""

# ---------------------------------------------------------------------------
# DDL - Silver Layer
# ---------------------------------------------------------------------------
_SILVER_DDL = """
CREATE TABLE IF NOT EXISTS silver_intelligence_signals (
    id                  VARCHAR PRIMARY KEY,        -- UUID
    bronze_id           VARCHAR NOT NULL,           -- FK → bronze_raw_articles.id
    source_name         VARCHAR NOT NULL,
    source_url          VARCHAR,
    title               VARCHAR,
    signal_category     VARCHAR NOT NULL,           -- Competitor_Strategy | Broker_Dynamics | Emerging_Risks
    summary             TEXT NOT NULL,              -- LLM-generated concise summary
    key_entities        JSON,                       -- list of named entities
    sentiment           VARCHAR,                    -- POSITIVE | NEGATIVE | NEUTRAL | MIXED
    confidence_score    FLOAT,                      -- 0.0–1.0 LLM confidence
    action_required     BOOLEAN DEFAULT FALSE,      -- flag for human review
    extracted_at        TIMESTAMPTZ NOT NULL,       -- UTC extraction timestamp
    llm_model           VARCHAR,                    -- model version used
    raw_llm_response    JSON                        -- full LLM output for audit
);
"""

# ---------------------------------------------------------------------------
# DDL - Gold Layer (materialised view placeholder)
# ---------------------------------------------------------------------------
_GOLD_VIEW_DDL = """
CREATE VIEW IF NOT EXISTS gold_signal_summary AS
SELECT
    signal_category,
    COUNT(*)                                        AS total_signals,
    SUM(CASE WHEN action_required THEN 1 ELSE 0 END) AS action_items,
    AVG(confidence_score)                           AS avg_confidence,
    MAX(extracted_at)                               AS latest_signal_at
FROM silver_intelligence_signals
GROUP BY signal_category
ORDER BY total_signals DESC;
"""

# ---------------------------------------------------------------------------
# DDL - Dead Letter Queue view (Bronze records that failed LLM extraction)
# ---------------------------------------------------------------------------
_DLQ_VIEW_DDL = """
CREATE VIEW IF NOT EXISTS gold_dead_letter_queue AS
SELECT
    id,
    source_name,
    source_url,
    title,
    LEFT(raw_text, 200)  AS raw_text_preview,
    ingested_at,
    processing_error
FROM bronze_raw_articles
WHERE processing_error IS NOT NULL
ORDER BY ingested_at DESC;
"""


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------
@contextmanager
def get_connection() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    Context manager that yields a DuckDB connection and ensures it is
    cleanly closed afterwards.

    Example::

        with get_connection() as con:
            rows = con.execute("SELECT COUNT(*) FROM bronze_raw_articles").fetchall()
    """
    con = duckdb.connect(str(settings.duckdb_path))
    try:
        logger.debug("DuckDB connection opened: %s", settings.duckdb_path)
        yield con
    except Exception:
        logger.exception("Unhandled exception within DuckDB connection context")
        raise
    finally:
        con.close()
        logger.debug("DuckDB connection closed.")


# ---------------------------------------------------------------------------
# Lake initialisation (idempotent)
# ---------------------------------------------------------------------------
def initialise_lake() -> None:
    """
    Create all Medallion tables and views if they do not already exist.
    Safe to call on every application startup.
    """
    logger.info("Initialising Medallion Data Lake at: %s", settings.duckdb_path)
    with get_connection() as con:
        con.execute(_BRONZE_DDL)
        logger.debug("Bronze layer ready.")
        con.execute(_SILVER_DDL)
        logger.debug("Silver layer ready.")
        con.execute(_GOLD_VIEW_DDL)
        logger.debug("Gold view ready.")
        con.execute(_DLQ_VIEW_DDL)
        logger.debug("Dead Letter Queue view ready.")
    logger.info("Data Lake initialisation complete.")


def _compute_content_hash(source_url: Optional[str], raw_text: str) -> str:
    """
    Compute a stable SHA-256 deduplication key.

    Strategy: prefer the canonical URL (normalised, stripped of tracking params)
    as the dedup signal.  Fall back to hashing the full raw_text body if no URL
    is present (e.g. internal document connectors).

    This ensures that the same article fetched in two consecutive pipeline runs
    produces the same hash and is silently skipped on the second insertion.
    """
    seed = (source_url or "").strip() or raw_text.strip()
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Bronze - write helpers
# ---------------------------------------------------------------------------
def insert_bronze_record(
    source_name: str,
    raw_text: str,
    source_url: Optional[str] = None,
    title: Optional[str] = None,
    raw_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Persist a single raw article to the Bronze layer.

    Uses ``INSERT OR IGNORE`` on the ``content_hash`` UNIQUE constraint so that
    re-running the pipeline never inserts the same article twice, even across
    process restarts (where in-memory dedup sets would be lost).

    Returns:
        The generated UUID primary key if the record was inserted, or
        ``None`` if the article was already present (duplicate silently skipped).
    """
    record_id = str(uuid.uuid4())
    ingested_at = datetime.now(tz=timezone.utc).isoformat()
    metadata_json = json.dumps(raw_metadata or {})
    content_hash = _compute_content_hash(source_url, raw_text)

    with get_connection() as con:
        con.execute(
            """
            INSERT OR IGNORE INTO bronze_raw_articles
                (id, content_hash, source_name, source_url, title, raw_text, raw_metadata, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [record_id, content_hash, source_name, source_url, title, raw_text, metadata_json, ingested_at],
        )
        # Check if the row was actually inserted (vs silently ignored as duplicate)
        inserted = con.execute(
            "SELECT id FROM bronze_raw_articles WHERE content_hash = ?", [content_hash]
        ).fetchone()

    actual_id = inserted[0] if inserted else None
    if actual_id == record_id:
        logger.debug("Bronze record inserted: id=%s  source=%s", record_id, source_name)
    else:
        logger.debug("Bronze duplicate skipped (content_hash=%s  source=%s)", content_hash[:12], source_name)
    return actual_id if actual_id == record_id else None


def bulk_insert_bronze_records(records: List[Dict[str, Any]]) -> List[str]:
    """
    Bulk-insert multiple raw records to the Bronze layer in a single
    transaction for efficiency.

    Uses ``INSERT OR IGNORE`` on the ``content_hash`` UNIQUE constraint.
    Duplicate articles (same URL or identical body seen in a previous run)
    are silently skipped — only genuinely new records are inserted.

    Each dict in ``records`` must contain: source_name, raw_text
    Optional keys: source_url, title, raw_metadata

    Returns:
        List of UUIDs for records that were **actually inserted** (excludes
        duplicates).  Use ``len(result)`` vs ``len(records)`` to measure
        dedup effectiveness.
    """
    rows = []
    now = datetime.now(tz=timezone.utc).isoformat()
    candidate_ids: Dict[str, str] = {}  # content_hash → record_id

    for rec in records:
        record_id = str(uuid.uuid4())
        raw_text: str = rec["raw_text"]
        source_url: Optional[str] = rec.get("source_url")
        content_hash = _compute_content_hash(source_url, raw_text)
        candidate_ids[content_hash] = record_id
        rows.append(
            (
                record_id,
                content_hash,
                rec["source_name"],
                source_url,
                rec.get("title"),
                raw_text,
                json.dumps(rec.get("raw_metadata") or {}),
                now,
            )
        )

    with get_connection() as con:
        con.executemany(
            """
            INSERT OR IGNORE INTO bronze_raw_articles
                (id, content_hash, source_name, source_url, title, raw_text, raw_metadata, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        # Determine which hashes were actually inserted vs skipped
        placeholders = ", ".join(["?"] * len(candidate_ids))
        inserted_hashes = {
            row[0]
            for row in con.execute(
                f"SELECT content_hash FROM bronze_raw_articles WHERE content_hash IN ({placeholders})",
                list(candidate_ids.keys()),
            ).fetchall()
        }

    inserted_ids = [candidate_ids[h] for h in inserted_hashes if h in candidate_ids]
    duplicate_count = len(records) - len(inserted_ids)
    if duplicate_count:
        logger.info(
            "Bulk Bronze insert: %d new records saved, %d duplicates skipped.",
            len(inserted_ids), duplicate_count,
        )
    else:
        logger.info("Bulk Bronze insert: %d records saved.", len(inserted_ids))
    return inserted_ids


# ---------------------------------------------------------------------------
# Bronze - read helpers
# ---------------------------------------------------------------------------
def fetch_unprocessed_bronze(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Return up to ``limit`` Bronze records that have not yet been
    processed by the LLM extraction engine.
    """
    with get_connection() as con:
        rows = con.execute(
            """
            SELECT id, source_name, source_url, title, raw_text, raw_metadata, ingested_at
            FROM   bronze_raw_articles
            WHERE  is_processed = FALSE
              AND  processing_error IS NULL
            ORDER  BY ingested_at ASC
            LIMIT  ?
            """,
            [limit],
        ).fetchall()

    columns = ["id", "source_name", "source_url", "title", "raw_text", "raw_metadata", "ingested_at"]
    result = [dict(zip(columns, row)) for row in rows]
    logger.debug("Fetched %d unprocessed Bronze records.", len(result))
    return result


def mark_bronze_processed(record_id: str, error: Optional[str] = None) -> None:
    """
    Mark a Bronze record as processed (success) or route it to the DLQ (failure).

    On success:
        Sets ``is_processed = TRUE`` — record exits the processing queue.

    On failure (``error`` is provided):
        Writes the error message to ``processing_error``.  The record remains
        ``is_processed = FALSE`` but is excluded from ``fetch_unprocessed_bronze``
        (which filters ``WHERE processing_error IS NULL``), effectively routing
        it to the Dead Letter Queue view for analyst review or manual retry.
    """
    with get_connection() as con:
        if error:
            con.execute(
                "UPDATE bronze_raw_articles SET processing_error = ? WHERE id = ?",
                [error, record_id],
            )
            logger.warning("Bronze record routed to DLQ: id=%s  error=%s", record_id, error[:80])
        else:
            con.execute(
                "UPDATE bronze_raw_articles SET is_processed = TRUE WHERE id = ?",
                [record_id],
            )


def fetch_dlq_records(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Return Bronze records that failed LLM extraction (the Dead Letter Queue).

    These are records where ``processing_error IS NOT NULL``.  Use
    ``retry_dlq_record()`` to clear the error flag and re-queue a record.

    Returns:
        List of dicts with keys: id, source_name, source_url, title,
        raw_text_preview, ingested_at, processing_error.
    """
    with get_connection() as con:
        rows = con.execute(
            """
            SELECT id, source_name, source_url, title,
                   LEFT(raw_text, 200) AS raw_text_preview,
                   ingested_at, processing_error
            FROM   bronze_raw_articles
            WHERE  processing_error IS NOT NULL
            ORDER  BY ingested_at DESC
            LIMIT  ?
            """,
            [limit],
        ).fetchall()

    columns = ["id", "source_name", "source_url", "title",
                "raw_text_preview", "ingested_at", "processing_error"]
    result = [dict(zip(columns, row)) for row in rows]
    logger.debug("Fetched %d DLQ records.", len(result))
    return result


def retry_dlq_record(record_id: str) -> None:
    """
    Clear the ``processing_error`` flag on a Bronze DLQ record, returning
    it to the unprocessed queue so it will be retried on the next pipeline run.

    Use this after fixing a prompt issue or model configuration that caused
    the original extraction failure.
    """
    with get_connection() as con:
        con.execute(
            "UPDATE bronze_raw_articles SET processing_error = NULL WHERE id = ?",
            [record_id],
        )
    logger.info("DLQ record re-queued for retry: id=%s", record_id)


# ---------------------------------------------------------------------------
# Silver - write helpers
# ---------------------------------------------------------------------------
def insert_silver_signal(signal: Dict[str, Any]) -> str:
    """
    Persist a validated intelligence signal to the Silver layer.

    ``signal`` must match the schema enforced by ``processor.IntelligenceSignal``.
    Returns the generated UUID primary key.
    """
    signal_id = str(uuid.uuid4())
    extracted_at = datetime.now(tz=timezone.utc).isoformat()

    with get_connection() as con:
        con.execute(
            """
            INSERT INTO silver_intelligence_signals (
                id, bronze_id, source_name, source_url, title,
                signal_category, summary, key_entities, sentiment,
                confidence_score, action_required, extracted_at,
                llm_model, raw_llm_response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                signal_id,
                signal["bronze_id"],
                signal["source_name"],
                signal.get("source_url"),
                signal.get("title"),
                signal["signal_category"],
                signal["summary"],
                json.dumps(signal.get("key_entities") or []),
                signal.get("sentiment", "NEUTRAL"),
                signal.get("confidence_score", 0.0),
                signal.get("action_required", False),
                extracted_at,
                signal.get("llm_model"),
                json.dumps(signal.get("raw_llm_response") or {}),
            ],
        )
    logger.debug("Silver signal inserted: id=%s  category=%s", signal_id, signal["signal_category"])
    return signal_id


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def lake_stats() -> Dict[str, Any]:
    """Return row counts and Gold-layer summary for operational dashboards."""
    with get_connection() as con:
        bronze_count = con.execute("SELECT COUNT(*) FROM bronze_raw_articles").fetchone()[0]
        silver_count = con.execute("SELECT COUNT(*) FROM silver_intelligence_signals").fetchone()[0]
        gold_rows = con.execute(
            "SELECT signal_category, total_signals, action_items, avg_confidence FROM gold_signal_summary"
        ).fetchall()

    gold_summary = [
        {
            "signal_category": r[0],
            "total_signals": r[1],
            "action_items": r[2],
            "avg_confidence": round(r[3] or 0.0, 3),
        }
        for r in gold_rows
    ]

    return {
        "bronze_total": bronze_count,
        "silver_total": silver_count,
        "gold_summary": gold_summary,
    }
