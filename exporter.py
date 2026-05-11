"""
exporter.py - Excel Report Exporter for the Strategic Market Intelligence Pipeline.

Export Flow
───────────
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  1. QUERY     - Pull Gold aggregate + Silver detail data from DuckDB    │
  │  2. TRANSFORM - Build two pandas DataFrames (dashboard + feed)          │
  │  3. EXPORT    - Write to .xlsx with xlsxwriter formatting engine        │
  │  4. FORMAT    - Apply conditional formatting, hyperlinks, auto-widths   │
  └──────────────────────────────────────────────────────────────────────────┘

Output File:  ``output/intelligence_report_<YYYYMMDD_HHMMSS>.xlsx``

Sheet 1: "Executive Dashboard" (Gold Layer)
    - Total signals by category (Competitor Strategy, Broker Dynamics, Emerging Risks)
    - Action-required count per category
    - Grand totals row
    → 5-second weekly overview for leadership.

Sheet 2: "Intelligence Feed" (Silver Layer)
    - Full categorized list of extracted signals
    - Columns: Date, Category, Summary, Sentiment, Source URL (clickable hyperlink)
    - Sorted with action_required = TRUE items at the top
    → Analyst's daily working tool.

Usage:
    from exporter import export_intelligence_report
    filepath = export_intelligence_report()
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend — no GUI needed
import matplotlib.pyplot as plt
import pandas as pd

from config import settings, PROJECT_ROOT
from database import get_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = PROJECT_ROOT / "output"


# ---------------------------------------------------------------------------
# Data extraction from DuckDB
# ---------------------------------------------------------------------------

def _fetch_gold_summary() -> pd.DataFrame:
    """
    Query the Gold-layer aggregate view and return a DataFrame with
    signal counts and action-required tallies grouped by category.

    Falls back to an empty DataFrame if no signals exist yet.
    """
    with get_connection() as con:
        df = con.execute(
            """
            SELECT
                signal_category                                        AS "Category",
                total_signals                                          AS "Total Signals",
                action_items                                           AS "Action Required",
                ROUND(avg_confidence * 100, 1)                         AS "Avg Confidence (%)",
                latest_signal_at                                       AS "Latest Signal"
            FROM gold_signal_summary
            ORDER BY total_signals DESC
            """
        ).fetchdf()

    if df.empty:
        logger.warning("Gold summary is empty — no signals in the Silver layer yet.")
        return pd.DataFrame(columns=[
            "Category", "Total Signals", "Action Required",
            "Avg Confidence (%)", "Latest Signal",
        ])

    # Human-readable category names
    _CATEGORY_LABELS = {
        "Competitor_Strategy": "Competitor Strategy",
        "Broker_Dynamics": "Broker Dynamics",
        "Emerging_Risks": "Emerging Risks",
    }
    df["Category"] = df["Category"].map(lambda c: _CATEGORY_LABELS.get(c, c))

    # Excel does not support timezone-aware datetimes — convert to naive string
    if "Latest Signal" in df.columns:
        df["Latest Signal"] = df["Latest Signal"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M") if pd.notna(x) else "—"
        )

    return df


def _fetch_intelligence_feed() -> pd.DataFrame:
    """
    Query the Silver layer for the full intelligence feed.

    Returns a DataFrame sorted so that action_required = TRUE rows
    appear first, then by extraction date descending.
    """
    with get_connection() as con:
        df = con.execute(
            """
            SELECT
                CAST(extracted_at AS DATE)                              AS "Date",
                signal_category                                        AS "Category",
                summary                                                AS "Summary",
                sentiment                                              AS "Sentiment",
                ROUND(confidence_score * 100, 1)                       AS "Confidence (%)",
                action_required                                        AS "Action Required",
                title                                                  AS "Title",
                source_url                                             AS "Source URL"
            FROM silver_intelligence_signals
            ORDER BY action_required DESC, extracted_at DESC
            """
        ).fetchdf()

    if df.empty:
        logger.warning("Intelligence feed is empty — no signals in the Silver layer yet.")
        return pd.DataFrame(columns=[
            "Date", "Category", "Summary", "Sentiment",
            "Confidence (%)", "Action Required", "Title", "Source URL",
        ])

    # Human-readable category names
    _CATEGORY_LABELS = {
        "Competitor_Strategy": "Competitor Strategy",
        "Broker_Dynamics": "Broker Dynamics",
        "Emerging_Risks": "Emerging Risks",
    }
    df["Category"] = df["Category"].map(lambda c: _CATEGORY_LABELS.get(c, c))

    # Convert Date column to plain strings for Excel compatibility
    if "Date" in df.columns:
        df["Date"] = df["Date"].apply(
            lambda x: str(x) if pd.notna(x) else "—"
        )

    return df


def _fetch_sentiment_distribution() -> pd.DataFrame:
    """Query sentiment counts for pie chart."""
    with get_connection() as con:
        df = con.execute(
            """
            SELECT sentiment AS "Sentiment", COUNT(*) AS "Count"
            FROM silver_intelligence_signals
            GROUP BY sentiment ORDER BY "Count" DESC
            """
        ).fetchdf()
    return df


def _fetch_action_breakdown() -> pd.DataFrame:
    """Query action-required breakdown for pie chart."""
    with get_connection() as con:
        df = con.execute(
            """
            SELECT
                CASE WHEN action_required THEN 'Action Required' ELSE 'Monitor' END AS "Status",
                COUNT(*) AS "Count"
            FROM silver_intelligence_signals
            GROUP BY action_required ORDER BY action_required DESC
            """
        ).fetchdf()
    return df


# ---------------------------------------------------------------------------
# Excel formatting helpers
# ---------------------------------------------------------------------------

# Brand colour palette (hex)
_COLORS = {
    "navy":        "#1B2A4A",
    "dark_blue":   "#2C3E6B",
    "accent_blue": "#4A90D9",
    "light_blue":  "#D6E4F0",
    "white":       "#FFFFFF",
    "light_gray":  "#F5F6FA",
    "text_dark":   "#1A1A2E",
    "text_gray":   "#6B7280",
    "green":       "#10B981",
    "red":         "#EF4444",
    "amber":       "#F59E0B",
    "red_bg":      "#FEE2E2",
    "green_bg":    "#D1FAE5",
    "amber_bg":    "#FEF3C7",
}

# Sentiment colour map for matplotlib
_SENT_COLORS = {
    "POSITIVE": _COLORS["green"],
    "NEGATIVE": _COLORS["red"],
    "MIXED":    _COLORS["amber"],
    "NEUTRAL":  _COLORS["text_gray"],
}


def _make_bar_chart(gold_df: pd.DataFrame) -> io.BytesIO:
    """Generate a bar chart image for signal counts by category."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=144)
    cats = gold_df["Category"].tolist()
    vals = gold_df["Total Signals"].astype(int).tolist()
    bars = ax.bar(cats, vals, color=_COLORS["accent_blue"],
                  edgecolor=_COLORS["navy"], linewidth=0.8, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(v), ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=_COLORS["navy"])
    ax.set_title("Signals by Category", fontsize=12, fontweight="bold",
                 color=_COLORS["navy"], pad=10)
    ax.set_ylabel("")
    ax.set_ylim(0, max(vals) * 1.3 if vals else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_COLORS["light_blue"])
    ax.spines["bottom"].set_color(_COLORS["light_blue"])
    ax.tick_params(colors=_COLORS["text_gray"], labelsize=9)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.patch.set_facecolor(_COLORS["light_gray"])
    ax.set_facecolor(_COLORS["white"])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_action_pie(action_df: pd.DataFrame) -> io.BytesIO:
    """Generate a pie chart image for action-required breakdown."""
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=144)
    labels = action_df["Status"].tolist()
    sizes = action_df["Count"].astype(int).tolist()
    colors = [_COLORS["red"] if "Action" in l else _COLORS["green"] for l in labels]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, textprops={"fontsize": 9, "color": _COLORS["text_dark"]},
        pctdistance=0.55, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_color("white")
    ax.set_title("Action Required", fontsize=12, fontweight="bold",
                 color=_COLORS["navy"], pad=10)
    fig.patch.set_facecolor(_COLORS["light_gray"])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_sentiment_pie(sentiment_df: pd.DataFrame) -> io.BytesIO:
    """Generate a pie chart image for sentiment distribution."""
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=144)
    labels = sentiment_df["Sentiment"].tolist()
    sizes = sentiment_df["Count"].astype(int).tolist()
    colors = [_SENT_COLORS.get(s.upper(), _COLORS["accent_blue"]) for s in labels]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, textprops={"fontsize": 9, "color": _COLORS["text_dark"]},
        pctdistance=0.55, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_color("white")
    ax.set_title("Sentiment Mix", fontsize=12, fontweight="bold",
                 color=_COLORS["navy"], pad=10)
    fig.patch.set_facecolor(_COLORS["light_gray"])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _apply_dashboard_formatting(
    workbook: "xlsxwriter.Workbook",
    worksheet: "xlsxwriter.Worksheet",
    gold_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    action_df: pd.DataFrame,
) -> None:
    """
    Build a full executive dashboard with KPI scorecards, charts, and a
    summary table. Layout:

      Row 0-1:   Title + subtitle
      Row 3-6:   KPI scorecards (Total Signals | Action Items | Avg Confidence)
      Row 8-24:  Charts row (Bar chart | Pie charts)
      Row 26+:   Gold summary data table
    """
    num_rows, num_cols = gold_df.shape

    # ── Shared Formats ───────────────────────────────────────────
    title_fmt = workbook.add_format({
        "bold": True, "font_size": 18, "font_color": _COLORS["navy"],
        "bottom": 3, "bottom_color": _COLORS["accent_blue"],
    })
    subtitle_fmt = workbook.add_format({
        "italic": True, "font_size": 10, "font_color": _COLORS["text_gray"],
    })
    section_title_fmt = workbook.add_format({
        "bold": True, "font_size": 13, "font_color": _COLORS["navy"],
        "bottom": 1, "bottom_color": _COLORS["light_blue"],
    })

    # KPI formats
    kpi_value_fmt = workbook.add_format({
        "bold": True, "font_size": 28, "font_color": _COLORS["navy"],
        "align": "center", "valign": "vcenter",
        "bg_color": _COLORS["light_gray"],
        "border": 1, "border_color": _COLORS["light_blue"],
    })
    kpi_label_fmt = workbook.add_format({
        "bold": True, "font_size": 10, "font_color": _COLORS["text_gray"],
        "align": "center", "valign": "vcenter",
        "bg_color": _COLORS["light_gray"],
        "border": 1, "border_color": _COLORS["light_blue"],
    })
    kpi_action_value_fmt = workbook.add_format({
        "bold": True, "font_size": 28, "font_color": _COLORS["red"],
        "align": "center", "valign": "vcenter",
        "bg_color": _COLORS["red_bg"],
        "border": 1, "border_color": _COLORS["light_blue"],
    })
    kpi_action_label_fmt = workbook.add_format({
        "bold": True, "font_size": 10, "font_color": _COLORS["red"],
        "align": "center", "valign": "vcenter",
        "bg_color": _COLORS["red_bg"],
        "border": 1, "border_color": _COLORS["light_blue"],
    })
    kpi_conf_value_fmt = workbook.add_format({
        "bold": True, "font_size": 28, "font_color": _COLORS["green"],
        "align": "center", "valign": "vcenter",
        "bg_color": _COLORS["green_bg"],
        "border": 1, "border_color": _COLORS["light_blue"],
    })
    kpi_conf_label_fmt = workbook.add_format({
        "bold": True, "font_size": 10, "font_color": _COLORS["green"],
        "align": "center", "valign": "vcenter",
        "bg_color": _COLORS["green_bg"],
        "border": 1, "border_color": _COLORS["light_blue"],
    })

    # Table formats
    header_fmt = workbook.add_format({
        "bold": True, "font_size": 11, "font_color": _COLORS["white"],
        "bg_color": _COLORS["navy"], "border": 1,
        "border_color": _COLORS["dark_blue"],
        "text_wrap": True, "align": "center", "valign": "vcenter",
    })
    data_fmt = workbook.add_format({
        "font_size": 11, "font_color": _COLORS["text_dark"],
        "border": 1, "border_color": _COLORS["light_blue"],
        "align": "center", "valign": "vcenter",
    })
    data_alt_fmt = workbook.add_format({
        "font_size": 11, "font_color": _COLORS["text_dark"],
        "bg_color": _COLORS["light_gray"], "border": 1,
        "border_color": _COLORS["light_blue"],
        "align": "center", "valign": "vcenter",
    })
    cat_fmt = workbook.add_format({
        "font_size": 11, "font_color": _COLORS["text_dark"], "bold": True,
        "border": 1, "border_color": _COLORS["light_blue"],
        "align": "left", "valign": "vcenter",
    })
    cat_alt_fmt = workbook.add_format({
        "font_size": 11, "font_color": _COLORS["text_dark"], "bold": True,
        "bg_color": _COLORS["light_gray"], "border": 1,
        "border_color": _COLORS["light_blue"],
        "align": "left", "valign": "vcenter",
    })
    action_hi_fmt = workbook.add_format({
        "font_size": 11, "font_color": _COLORS["red"], "bold": True,
        "bg_color": _COLORS["red_bg"], "border": 1,
        "border_color": _COLORS["light_blue"],
        "align": "center", "valign": "vcenter",
    })
    total_fmt = workbook.add_format({
        "bold": True, "font_size": 12, "font_color": _COLORS["white"],
        "bg_color": _COLORS["dark_blue"], "border": 1,
        "border_color": _COLORS["navy"],
        "align": "center", "valign": "vcenter",
    })
    total_label_fmt = workbook.add_format({
        "bold": True, "font_size": 12, "font_color": _COLORS["white"],
        "bg_color": _COLORS["dark_blue"], "border": 1,
        "border_color": _COLORS["navy"],
        "align": "left", "valign": "vcenter",
    })

    # ── Column widths ────────────────────────────────────────────
    worksheet.set_column(0, 0, 3)    # spacer
    worksheet.set_column(1, 1, 22)   # Category / KPI
    worksheet.set_column(2, 2, 16)
    worksheet.set_column(3, 3, 18)
    worksheet.set_column(4, 4, 20)
    worksheet.set_column(5, 5, 22)
    worksheet.set_column(6, 8, 14)   # chart data area (hidden later)
    worksheet.set_column(9, 9, 3)    # spacer

    # ── Row 0-1: Title ───────────────────────────────────────────
    worksheet.merge_range(0, 1, 0, 5, "📊 Executive Dashboard", title_fmt)
    run_ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    worksheet.merge_range(
        1, 1, 1, 5,
        f"Strategic Market Intelligence Overview  •  Generated {run_ts}",
        subtitle_fmt,
    )
    worksheet.set_row(0, 35)

    # ── Row 3-5: KPI Scorecards ──────────────────────────────────
    total_signals = int(gold_df["Total Signals"].sum()) if num_rows > 0 else 0
    action_items = int(gold_df["Action Required"].sum()) if num_rows > 0 else 0
    avg_conf = round(gold_df["Avg Confidence (%)"].mean(), 1) if num_rows > 0 else 0.0

    # KPI 1: Total Signals
    worksheet.merge_range(3, 1, 4, 2, str(total_signals), kpi_value_fmt)
    worksheet.merge_range(5, 1, 5, 2, "TOTAL SIGNALS", kpi_label_fmt)

    # KPI 2: Action Required
    worksheet.merge_range(3, 3, 4, 4, str(action_items), kpi_action_value_fmt)
    worksheet.merge_range(5, 3, 5, 4, "⚠  ACTION REQUIRED", kpi_action_label_fmt)

    # KPI 3: Avg Confidence
    worksheet.merge_range(3, 5, 4, 6, f"{avg_conf}%", kpi_conf_value_fmt)
    worksheet.merge_range(5, 5, 5, 6, "AVG CONFIDENCE", kpi_conf_label_fmt)

    worksheet.set_row(3, 30)
    worksheet.set_row(4, 30)
    worksheet.set_row(5, 20)

    # ── Row 7: Section title ─────────────────────────────────────
    worksheet.merge_range(7, 1, 7, 6, "Signal Distribution & Trends", section_title_fmt)

    # ── Charts as embedded PNG images (works in Numbers, Excel, Libre) ──
    if num_rows > 0:
        bar_buf = _make_bar_chart(gold_df)
        worksheet.insert_image("B9", "bar_chart.png",
                               {"image_data": bar_buf, "x_scale": 0.52, "y_scale": 0.52})

    if len(action_df) > 0:
        act_buf = _make_action_pie(action_df)
        worksheet.insert_image("E9", "action_pie.png",
                               {"image_data": act_buf, "x_scale": 0.52, "y_scale": 0.52})

    if len(sentiment_df) > 0:
        sent_buf = _make_sentiment_pie(sentiment_df)
        worksheet.insert_image("G9", "sentiment_pie.png",
                               {"image_data": sent_buf, "x_scale": 0.52, "y_scale": 0.52})


    # ── Row 26: Table section title ──────────────────────────────
    table_title_row = 26
    worksheet.merge_range(table_title_row, 1, table_title_row, 5, "Category Breakdown", section_title_fmt)

    # ── Table headers (row 28) ───────────────────────────────────
    tbl_header_row = 28
    for col_idx, col_name in enumerate(gold_df.columns):
        worksheet.write(tbl_header_row, col_idx + 1, col_name, header_fmt)

    # ── Table data rows ──────────────────────────────────────────
    for row_idx in range(num_rows):
        is_alt = row_idx % 2 == 1
        for col_idx, col_name in enumerate(gold_df.columns):
            value = gold_df.iloc[row_idx, col_idx]
            if col_idx == 0:
                fmt = cat_alt_fmt if is_alt else cat_fmt
            elif col_name == "Action Required" and value and int(value) > 0:
                fmt = action_hi_fmt
            else:
                fmt = data_alt_fmt if is_alt else data_fmt
            if pd.isna(value):
                worksheet.write(tbl_header_row + 1 + row_idx, col_idx + 1, "—", fmt)
            else:
                worksheet.write(tbl_header_row + 1 + row_idx, col_idx + 1, value, fmt)

    # ── Grand totals row ─────────────────────────────────────────
    if num_rows > 0:
        total_row = tbl_header_row + 1 + num_rows
        worksheet.write(total_row, 1, "GRAND TOTAL", total_label_fmt)
        for col_idx, col_name in enumerate(gold_df.columns):
            if col_name in ("Total Signals", "Action Required"):
                worksheet.write(total_row, col_idx + 1, int(gold_df[col_name].sum()), total_fmt)
            elif col_name == "Avg Confidence (%)":
                worksheet.write(total_row, col_idx + 1, round(gold_df[col_name].mean(), 1), total_fmt)
            elif col_idx > 0:
                worksheet.write(total_row, col_idx + 1, "", total_fmt)
        worksheet.set_row(tbl_header_row, 25)

    # Hide gridlines for clean dashboard look
    worksheet.hide_gridlines(2)




def _apply_feed_formatting(
    workbook: "xlsxwriter.Workbook",
    worksheet: "xlsxwriter.Worksheet",
    df: pd.DataFrame,
) -> None:
    """
    Apply professional formatting to the Intelligence Feed sheet.
    Includes: title, branded header, action-required row highlighting,
    clickable URL hyperlinks, text wrapping, and auto-sized columns.
    """
    num_rows, num_cols = df.shape

    # ── Formats ──────────────────────────────────────────────────
    title_fmt = workbook.add_format({
        "bold": True,
        "font_size": 16,
        "font_color": _COLORS["navy"],
        "bottom": 2,
        "bottom_color": _COLORS["accent_blue"],
    })
    subtitle_fmt = workbook.add_format({
        "italic": True,
        "font_size": 10,
        "font_color": _COLORS["text_gray"],
    })
    header_fmt = workbook.add_format({
        "bold": True,
        "font_size": 11,
        "font_color": _COLORS["white"],
        "bg_color": _COLORS["navy"],
        "border": 1,
        "border_color": _COLORS["dark_blue"],
        "text_wrap": True,
        "align": "center",
        "valign": "vcenter",
    })

    # Normal row
    data_fmt = workbook.add_format({
        "font_size": 10,
        "font_color": _COLORS["text_dark"],
        "border": 1,
        "border_color": _COLORS["light_blue"],
        "text_wrap": True,
        "valign": "vcenter",
    })
    data_center_fmt = workbook.add_format({
        "font_size": 10,
        "font_color": _COLORS["text_dark"],
        "border": 1,
        "border_color": _COLORS["light_blue"],
        "align": "center",
        "valign": "vcenter",
    })
    date_fmt = workbook.add_format({
        "font_size": 10,
        "font_color": _COLORS["text_dark"],
        "border": 1,
        "border_color": _COLORS["light_blue"],
        "align": "center",
        "valign": "vcenter",
        "num_format": "yyyy-mm-dd",
    })

    # Action-required row (urgent — light red background)
    action_data_fmt = workbook.add_format({
        "font_size": 10,
        "font_color": _COLORS["text_dark"],
        "bg_color": _COLORS["red_bg"],
        "border": 1,
        "border_color": _COLORS["light_blue"],
        "text_wrap": True,
        "valign": "vcenter",
    })
    action_center_fmt = workbook.add_format({
        "font_size": 10,
        "font_color": _COLORS["text_dark"],
        "bg_color": _COLORS["red_bg"],
        "border": 1,
        "border_color": _COLORS["light_blue"],
        "align": "center",
        "valign": "vcenter",
    })
    action_date_fmt = workbook.add_format({
        "font_size": 10,
        "font_color": _COLORS["text_dark"],
        "bg_color": _COLORS["red_bg"],
        "border": 1,
        "border_color": _COLORS["light_blue"],
        "align": "center",
        "valign": "vcenter",
        "num_format": "yyyy-mm-dd",
    })
    action_badge_fmt = workbook.add_format({
        "font_size": 10,
        "font_color": _COLORS["white"],
        "bg_color": _COLORS["red"],
        "bold": True,
        "border": 1,
        "border_color": _COLORS["light_blue"],
        "align": "center",
        "valign": "vcenter",
    })
    normal_badge_fmt = workbook.add_format({
        "font_size": 10,
        "font_color": _COLORS["green"],
        "border": 1,
        "border_color": _COLORS["light_blue"],
        "align": "center",
        "valign": "vcenter",
    })
    url_fmt = workbook.add_format({
        "font_size": 10,
        "font_color": _COLORS["accent_blue"],
        "underline": True,
        "border": 1,
        "border_color": _COLORS["light_blue"],
        "valign": "vcenter",
    })
    url_action_fmt = workbook.add_format({
        "font_size": 10,
        "font_color": _COLORS["accent_blue"],
        "underline": True,
        "bg_color": _COLORS["red_bg"],
        "border": 1,
        "border_color": _COLORS["light_blue"],
        "valign": "vcenter",
    })

    # Sentiment colour coding
    sentiment_fmts = {
        "POSITIVE": workbook.add_format({
            "font_size": 10, "font_color": _COLORS["green"], "bold": True,
            "border": 1, "border_color": _COLORS["light_blue"],
            "align": "center", "valign": "vcenter",
        }),
        "NEGATIVE": workbook.add_format({
            "font_size": 10, "font_color": _COLORS["red"], "bold": True,
            "border": 1, "border_color": _COLORS["light_blue"],
            "align": "center", "valign": "vcenter",
        }),
        "MIXED": workbook.add_format({
            "font_size": 10, "font_color": _COLORS["amber"], "bold": True,
            "border": 1, "border_color": _COLORS["light_blue"],
            "align": "center", "valign": "vcenter",
        }),
        "NEUTRAL": workbook.add_format({
            "font_size": 10, "font_color": _COLORS["text_gray"],
            "border": 1, "border_color": _COLORS["light_blue"],
            "align": "center", "valign": "vcenter",
        }),
    }
    sentiment_action_fmts = {
        "POSITIVE": workbook.add_format({
            "font_size": 10, "font_color": _COLORS["green"], "bold": True,
            "bg_color": _COLORS["red_bg"],
            "border": 1, "border_color": _COLORS["light_blue"],
            "align": "center", "valign": "vcenter",
        }),
        "NEGATIVE": workbook.add_format({
            "font_size": 10, "font_color": _COLORS["red"], "bold": True,
            "bg_color": _COLORS["red_bg"],
            "border": 1, "border_color": _COLORS["light_blue"],
            "align": "center", "valign": "vcenter",
        }),
        "MIXED": workbook.add_format({
            "font_size": 10, "font_color": _COLORS["amber"], "bold": True,
            "bg_color": _COLORS["red_bg"],
            "border": 1, "border_color": _COLORS["light_blue"],
            "align": "center", "valign": "vcenter",
        }),
        "NEUTRAL": workbook.add_format({
            "font_size": 10, "font_color": _COLORS["text_gray"],
            "bg_color": _COLORS["red_bg"],
            "border": 1, "border_color": _COLORS["light_blue"],
            "align": "center", "valign": "vcenter",
        }),
    }

    # ── Visible columns (exclude "Action Required" as a raw column,
    #    we use it only for row styling) ──────────────────────────
    display_cols = ["Date", "Category", "Title", "Summary", "Sentiment",
                    "Confidence (%)", "Action Required", "Source URL"]

    # ── Title block ──────────────────────────────────────────────
    worksheet.merge_range(0, 0, 0, len(display_cols) - 1, "📋 Intelligence Feed", title_fmt)
    run_ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    worksheet.merge_range(
        1, 0, 1, len(display_cols) - 1,
        f"Categorized signal feed — action-required items listed first  •  Generated {run_ts}",
        subtitle_fmt,
    )

    # ── Headers (row 3) ─────────────────────────────────────────
    header_row = 3
    for col_idx, col_name in enumerate(display_cols):
        worksheet.write(header_row, col_idx, col_name, header_fmt)

    # ── Data rows ────────────────────────────────────────────────
    for row_idx in range(num_rows):
        is_action = bool(df.iloc[row_idx].get("Action Required", False))
        excel_row = header_row + 1 + row_idx

        for col_idx, col_name in enumerate(display_cols):
            value = df.iloc[row_idx].get(col_name, "")

            # Handle NaT / NaN
            if pd.isna(value):
                value = "—"

            # Source URL → clickable hyperlink
            if col_name == "Source URL":
                fmt = url_action_fmt if is_action else url_fmt
                if value and value != "—" and str(value).startswith("http"):
                    worksheet.write_url(
                        excel_row, col_idx,
                        str(value),
                        fmt,
                        string="🔗 Open Article",
                    )
                else:
                    worksheet.write(excel_row, col_idx, "—", fmt)
                continue

            # Action Required → badge
            if col_name == "Action Required":
                if is_action:
                    worksheet.write(excel_row, col_idx, "⚠ ACTION", action_badge_fmt)
                else:
                    worksheet.write(excel_row, col_idx, "—", normal_badge_fmt)
                continue

            # Sentiment → colour-coded
            if col_name == "Sentiment":
                sent_str = str(value).upper() if value != "—" else "NEUTRAL"
                fmts = sentiment_action_fmts if is_action else sentiment_fmts
                fmt = fmts.get(sent_str, fmts["NEUTRAL"])
                worksheet.write(excel_row, col_idx, sent_str, fmt)
                continue

            # Date column
            if col_name == "Date":
                fmt = action_date_fmt if is_action else date_fmt
                worksheet.write(excel_row, col_idx, str(value), fmt)
                continue

            # Category, Confidence — centered
            if col_name in ("Category", "Confidence (%)"):
                fmt = action_center_fmt if is_action else data_center_fmt
                worksheet.write(excel_row, col_idx, value, fmt)
                continue

            # Default (Summary, Title) — wrapped text
            fmt = action_data_fmt if is_action else data_fmt
            worksheet.write(excel_row, col_idx, str(value), fmt)

        # Taller row for readability of summary text
        worksheet.set_row(excel_row, 50)

    # ── Column widths ────────────────────────────────────────────
    col_widths = {
        "Date": 13,
        "Category": 20,
        "Title": 35,
        "Summary": 60,
        "Sentiment": 13,
        "Confidence (%)": 15,
        "Action Required": 16,
        "Source URL": 18,
    }
    for col_idx, col_name in enumerate(display_cols):
        worksheet.set_column(col_idx, col_idx, col_widths.get(col_name, 18))

    # Row heights
    worksheet.set_row(0, 30)   # Title
    worksheet.set_row(header_row, 25)  # Header

    # ── Freeze panes: freeze title + header ──────────────────────
    worksheet.freeze_panes(header_row + 1, 0)

    # ── Autofilter on header row ─────────────────────────────────
    worksheet.autofilter(header_row, 0, header_row + num_rows, len(display_cols) - 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_intelligence_report(output_path: Optional[str] = None) -> Path:
    """
    Export DuckDB intelligence data to a professionally formatted Excel file.

    Creates a single ``.xlsx`` workbook with two sheets:
      - **Executive Dashboard** — Gold-layer aggregate summary
      - **Intelligence Feed** — Silver-layer detail with clickable URLs

    Args:
        output_path: Optional explicit file path for the output .xlsx.
                     If None, generates a timestamped file in ``output/``.

    Returns:
        Path to the generated .xlsx file.
    """
    # Resolve output path
    if output_path:
        filepath = Path(output_path)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = OUTPUT_DIR / f"intelligence_report_{timestamp}.xlsx"

    filepath.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting intelligence report to: %s", filepath)

    # Fetch data
    gold_df = _fetch_gold_summary()
    feed_df = _fetch_intelligence_feed()
    sentiment_df = _fetch_sentiment_distribution()
    action_df = _fetch_action_breakdown()

    # Write to Excel with xlsxwriter engine
    with pd.ExcelWriter(str(filepath), engine="xlsxwriter") as writer:
        workbook = writer.book

        # ── Sheet 1: Executive Dashboard ─────────────────────────
        # Create the sheet via an empty write (we control all layout)
        pd.DataFrame().to_excel(writer, sheet_name="Executive Dashboard", index=False)
        dashboard_ws = writer.sheets["Executive Dashboard"]

        _apply_dashboard_formatting(workbook, dashboard_ws, gold_df, sentiment_df, action_df)

        # Tab colour
        dashboard_ws.set_tab_color(_COLORS["navy"])

        # ── Sheet 2: Intelligence Feed ───────────────────────────
        # Write a minimal frame to create the sheet
        feed_df.head(0).to_excel(writer, sheet_name="Intelligence Feed", index=False, startrow=4)
        feed_ws = writer.sheets["Intelligence Feed"]

        _apply_feed_formatting(workbook, feed_ws, feed_df)

        # Tab colour
        feed_ws.set_tab_color(_COLORS["accent_blue"])

    logger.info(
        "Report exported: %s  (Dashboard: %d categories, Feed: %d signals)",
        filepath, len(gold_df), len(feed_df),
    )
    return filepath
