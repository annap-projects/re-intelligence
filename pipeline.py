"""
pipeline.py - Orchestrator for the Strategic Market Intelligence Pipeline.

Execution Flow
──────────────
  ┌─────────────────────────────────────────────────────────────────────┐
  │  1. INGEST    - Connector.fetch() → yields RawArticle objects       │
  │  2. PERSIST   - Bulk-write RawArticles to Bronze layer (DuckDB)     │
  │  3. PROCESS   - IntelligenceProcessor.extract_batch() on Bronze     │
  │  4. SAVE      - Write validated IntelligenceSignals to Silver layer  │
  │  5. REPORT    - Print Gold-layer summary to console                  │
  └─────────────────────────────────────────────────────────────────────┘

Design Decisions:
  - The pipeline is stateless: each run is independent and idempotent.
  - Bronze-first: raw data is persisted BEFORE LLM processing so that
    no ingested article is ever lost due to a downstream failure.
  - Dry-run mode: set PIPELINE_DRY_RUN=true to test ingestion without
    consuming Anthropic API tokens.
  - Connector is swappable via CLI argument or code change.

Usage:
    python pipeline.py                     # default: RSS connector
    python pipeline.py --connector rss
    python pipeline.py --connector lexisnexis  # requires credentials
    python pipeline.py --dry-run           # skip LLM processing
    python pipeline.py --query "cat bond"  # keyword filter on ingest
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from config import settings
from connectors import DataConnector, RawArticle, get_connector
from database import (
    bulk_insert_bronze_records,
    fetch_dlq_records,
    fetch_unprocessed_bronze,
    initialise_lake,
    insert_silver_signal,
    lake_stats,
    mark_bronze_processed,
)
from processor import IntelligenceProcessor, ProcessingError

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

def step_ingest(
    connector: DataConnector,
    query: Optional[str] = None,
) -> List[RawArticle]:
    """
    Step 1: Run the connector and collect raw articles.

    Args:
        connector: Authenticated DataConnector instance.
        query: Optional keyword filter passed to connector.fetch().

    Returns:
        List of RawArticle objects.
    """
    console.print("[bold cyan]▶ Step 1/4:[/bold cyan] Ingesting articles …")
    articles: List[RawArticle] = []

    try:
        for article in connector.fetch(query=query):
            articles.append(article)
            console.print(
                f"  [green]✓[/green] [{article.source_name}] {article.title or '(no title)'!r:.80s}"
            )
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        console.print(f"[bold red]  ✗ Ingestion error:[/bold red] {exc}")

    console.print(f"\n  [bold]Total ingested:[/bold] {len(articles)} articles")
    return articles


def step_persist_bronze(articles: List[RawArticle]) -> List[str]:
    """
    Step 2: Bulk-insert raw articles into the Bronze (raw) layer.

    Args:
        articles: List of RawArticle objects from the ingestion step.

    Returns:
        List of Bronze record UUIDs.
    """
    console.print("\n[bold cyan]▶ Step 2/4:[/bold cyan] Persisting to Bronze layer …")

    if not articles:
        console.print("  [yellow]No articles to persist.[/yellow]")
        return []

    records = [
        {
            "source_name": a.source_name,
            "raw_text": a.raw_text,
            "source_url": a.source_url,
            "title": a.title,
            "raw_metadata": a.raw_metadata,
        }
        for a in articles
    ]

    ids = bulk_insert_bronze_records(records)
    console.print(f"  [bold]Bronze records saved:[/bold] {len(ids)}")
    return ids


def step_process_bronze(dry_run: bool = False) -> int:
    """
    Step 3: Fetch unprocessed Bronze records, run LLM extraction,
            and persist signals to the Silver layer.

    Args:
        dry_run: If True, skip LLM processing (useful for testing ingestion).

    Returns:
        Number of Silver signals created.
    """
    console.print("\n[bold cyan]▶ Step 3/4:[/bold cyan] Running LLM Extraction Engine …")

    if dry_run:
        console.print("  [yellow]DRY RUN - LLM processing skipped.[/yellow]")
        return 0

    if not settings.anthropic_api_key:
        console.print(
            "  [bold red]⚠ ANTHROPIC_API_KEY not set.[/bold red] "
            "Add it to your .env file to enable LLM extraction."
        )
        return 0

    bronze_records = fetch_unprocessed_bronze(limit=settings.max_bronze_batch_size)
    if not bronze_records:
        console.print("  [yellow]No unprocessed Bronze records found.[/yellow]")
        return 0

    console.print(f"  Processing {len(bronze_records)} Bronze records …")

    processor = IntelligenceProcessor()
    silver_signals = processor.extract_batch(bronze_records)

    signal_count = 0
    for signal in silver_signals:
        try:
            insert_silver_signal(signal)
            mark_bronze_processed(signal["bronze_id"])
            signal_count += 1
            console.print(
                f"  [green]✓[/green] [{signal['signal_category']}] "
                f"{signal.get('title', 'Untitled')!r:.60s} "
                f"(confidence={signal.get('confidence_score', 0):.0%})"
            )
        except Exception as exc:
            logger.error("Failed to save Silver signal for bronze_id=%s: %s", signal["bronze_id"], exc)
            mark_bronze_processed(signal["bronze_id"], error=str(exc)[:500])

    # Mark remaining Bronze records (noise) as processed
    silver_bronze_ids = {s["bronze_id"] for s in silver_signals}
    for record in bronze_records:
        if record["id"] not in silver_bronze_ids:
            mark_bronze_processed(record["id"])

    console.print(f"\n  [bold]Silver signals created:[/bold] {signal_count} / {len(bronze_records)} records")
    return signal_count


def step_report() -> None:
    """
    Step 4: Render a Rich-formatted summary from the Gold layer.
    """
    console.print("\n[bold cyan]▶ Step 4/4:[/bold cyan] Gold Layer Summary\n")

    stats = lake_stats()
    dlq_count = len(fetch_dlq_records(limit=1000))

    # Lake overview
    overview_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    overview_table.add_column("Metric", style="dim")
    overview_table.add_column("Value", justify="right")
    overview_table.add_row("Bronze Records (total)", str(stats["bronze_total"]))
    overview_table.add_row("Silver Signals (total)", str(stats["silver_total"]))
    dlq_style = "bold red" if dlq_count else "green"
    overview_table.add_row(f"[{dlq_style}]Dead Letter Queue[/{dlq_style}]", f"[{dlq_style}]{dlq_count}[/{dlq_style}]")
    console.print(overview_table)

    # Gold breakdown
    if stats["gold_summary"]:
        console.print()
        gold_table = Table(
            title="Signal Distribution (Gold View)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold yellow",
        )
        gold_table.add_column("Category")
        gold_table.add_column("Signals", justify="right")
        gold_table.add_column("Action Items", justify="right")
        gold_table.add_column("Avg. Confidence", justify="right")

        for row in stats["gold_summary"]:
            gold_table.add_row(
                row["signal_category"],
                str(row["total_signals"]),
                str(row["action_items"]),
                f"{row['avg_confidence']:.1%}",
            )
        console.print(gold_table)
    else:
        console.print("  [dim]No signals in Silver layer yet.[/dim]")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    connector_name: str = "rss",
    query: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """
    Execute the full Market Intelligence Pipeline end-to-end.

    Args:
        connector_name: Key from CONNECTOR_REGISTRY (default: ``"rss"``).
        query:          Optional keyword filter for ingestion.
        dry_run:        Skip LLM processing step.
    """
    start_time = time.time()
    run_ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    console.print(
        Panel(
            f"[bold white]Strategic Market Intelligence Pipeline[/bold white]\n"
            f"[dim]Run started: {run_ts}  |  Connector: {connector_name}  |  "
            f"Dry-run: {dry_run}[/dim]",
            style="bold blue",
            expand=False,
        )
    )

    # Initialise Data Lake
    initialise_lake()

    # Resolve & authenticate connector
    try:
        connector = get_connector(connector_name)
        connector.authenticate()
    except KeyError as exc:
        console.print(f"[bold red]Unknown connector: {exc}[/bold red]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[bold red]Connector authentication failed: {exc}[/bold red]")
        sys.exit(1)

    try:
        # Step 1 - Ingest
        articles = step_ingest(connector, query=query)

        # Step 2 - Bronze
        bronze_ids = step_persist_bronze(articles)

        # Step 3 - Silver
        signal_count = step_process_bronze(dry_run=dry_run)

        # Step 4 - Report
        step_report()

    finally:
        connector.close()

    elapsed = time.time() - start_time
    console.print(
        Panel(
            f"[bold green]Pipeline complete in {elapsed:.1f}s[/bold green]\n"
            f"[dim]Ingested: {len(articles)} articles  |  "
            f"Signals: {signal_count}[/dim]",
            style="green",
            expand=False,
        )
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Strategic Market Intelligence Pipeline - Reinsurance Data Lake",
    )
    parser.add_argument(
        "--connector",
        default="rss",
        choices=["rss", "lexisnexis"],
        help="Data connector to use (default: rss)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional keyword filter applied during ingestion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=settings.pipeline_dry_run,
        help="Ingest to Bronze but skip LLM processing",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_pipeline(
        connector_name=args.connector,
        query=args.query,
        dry_run=args.dry_run,
    )
