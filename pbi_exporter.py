"""
pbi_exporter.py - Power BI Dashboard Exporter for the Strategic Market Intelligence Pipeline.

Exports DuckDB Silver-layer data into:
  1. Parquet files (star schema) for Power BI Desktop import
  2. Interactive HTML dashboard (Plotly) for browser-based viewing

Dashboard Pages:
  - Competitor Overview: Stacked column chart + news feed
  - Emerging Risks: Heatmap + news feed
  - Broker Dynamics: Bubble chart + news feed
  - Sentiment Analysis: Stacked bars + trend lines
  - News Timeline: Month / Week / Day tables
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import PROJECT_ROOT
from database import get_connection

logger = logging.getLogger(__name__)

OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# Brand palette
_C = {
    "navy": "#1B2A4A", "blue": "#4A90D9", "light_blue": "#D6E4F0",
    "green": "#10B981", "red": "#EF4444", "amber": "#F59E0B",
    "gray": "#6B7280", "bg": "#F8FAFC", "white": "#FFFFFF",
    "cat_competitor": "#4A90D9", "cat_broker": "#6B7280", "cat_emerging": "#F59E0B",
}
_SENT_MAP = {"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0, "MIXED": 0.5}
_SENT_COLOR = {"POSITIVE": _C["green"], "NEGATIVE": _C["red"],
               "NEUTRAL": _C["gray"], "MIXED": _C["amber"]}
_CAT_COLOR = {"Competitor_Strategy": _C["cat_competitor"],
              "Broker_Dynamics": _C["cat_broker"],
              "Emerging_Risks": _C["cat_emerging"]}
_CAT_LABEL = {"Competitor_Strategy": "Competitor Strategy",
              "Broker_Dynamics": "Broker Dynamics",
              "Emerging_Risks": "Emerging Risks"}


# ═══════════════════════════════════════════════════════════════════════
# DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def _load_signals() -> pd.DataFrame:
    """Load all Silver-layer signals into a DataFrame with computed columns."""
    with get_connection() as con:
        df = con.execute("""
            SELECT id, signal_category, title, summary, key_entities,
                   sentiment, confidence_score, action_required,
                   extracted_at, source_url, source_name
            FROM silver_intelligence_signals
            ORDER BY extracted_at DESC
        """).fetchdf()

    if df.empty:
        return df

    df["category_label"] = df["signal_category"].map(lambda c: _CAT_LABEL.get(c, c))
    df["sentiment_score"] = df["sentiment"].map(lambda s: _SENT_MAP.get(s, 0.0))
    df["extracted_date"] = pd.to_datetime(df["extracted_at"]).dt.date
    df["year_month"] = pd.to_datetime(df["extracted_at"]).dt.to_period("M").astype(str)
    df["year_week"] = pd.to_datetime(df["extracted_at"]).dt.strftime("%Y-W%V")
    df["day"] = pd.to_datetime(df["extracted_at"]).dt.strftime("%Y-%m-%d")

    # Explode entities
    def _parse_entities(raw):
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return []
        if isinstance(raw, list):
            return raw
        return []

    df["entities_list"] = df["key_entities"].apply(_parse_entities)
    return df


def _build_entities_df(signals: pd.DataFrame) -> pd.DataFrame:
    """Explode key_entities into a flat entity table."""
    rows = []
    for _, r in signals.iterrows():
        cat = r["signal_category"]
        etype = ("Competitor" if cat == "Competitor_Strategy"
                 else "Broker" if cat == "Broker_Dynamics"
                 else "Risk")
        for ent in r["entities_list"]:
            rows.append({"signal_id": r["id"], "entity_name": str(ent),
                          "entity_type": etype, "category": r["signal_category"]})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["signal_id", "entity_name", "entity_type", "category"])


# ═══════════════════════════════════════════════════════════════════════
# PARQUET EXPORT
# ═══════════════════════════════════════════════════════════════════════

def _export_parquet(signals: pd.DataFrame, entities: pd.DataFrame, out_dir: Path) -> None:
    """Export star-schema Parquet files for Power BI Desktop."""
    parquet_dir = out_dir / "powerbi" / "data"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Fact table
    fact_cols = ["id", "signal_category", "category_label", "title", "summary",
                 "sentiment", "sentiment_score", "confidence_score",
                 "action_required", "source_url", "source_name",
                 "extracted_date", "year_month", "year_week", "day"]
    available = [c for c in fact_cols if c in signals.columns]
    signals[available].to_parquet(parquet_dir / "fact_signals.parquet", index=False)

    # Entities dimension
    entities.to_parquet(parquet_dir / "dim_entities.parquet", index=False)

    logger.info("Parquet files exported to %s", parquet_dir)


# ═══════════════════════════════════════════════════════════════════════
# PLOTLY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

def _make_news_table_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render a styled HTML news table from a signals DataFrame."""
    if df.empty:
        return "<p style='color:#6B7280;padding:20px;'>No signals in this period.</p>"

    rows_html = []
    for _, r in df.head(max_rows).iterrows():
        sent = str(r.get("sentiment", ""))
        sc = _SENT_COLOR.get(sent, _C["gray"])
        action = "⚠️" if r.get("action_required") else ""
        url = r.get("source_url", "")
        link = f'<a href="{url}" target="_blank" style="color:{_C["blue"]}">🔗</a>' if url else ""
        rows_html.append(f"""<tr>
            <td>{r.get('day','')}</td>
            <td>{_CAT_LABEL.get(r.get('signal_category',''), r.get('category_label',''))}</td>
            <td style="font-weight:600">{r.get('title','')}</td>
            <td>{r.get('summary','')[:200]}</td>
            <td style="color:{sc};font-weight:700">{sent}</td>
            <td style="text-align:center">{action}</td>
            <td style="text-align:center">{link}</td>
        </tr>""")

    return f"""<table class="news-tbl">
        <thead><tr><th>Date</th><th>Category</th><th>Title</th><th>Summary</th>
        <th>Sentiment</th><th>Action</th><th>Source</th></tr></thead>
        <tbody>{''.join(rows_html)}</tbody></table>"""


def _build_competitor_page(signals: pd.DataFrame, entities: pd.DataFrame) -> str:
    """Page 1: Stacked column chart + news feed."""
    comp_ent = entities[entities["entity_type"] == "Competitor"]
    if comp_ent.empty:
        return "<h2>Competitor Overview</h2><p>No competitor entities extracted yet.</p>"

    # Count by entity × category
    merged = comp_ent.merge(signals[["id", "signal_category"]], left_on="signal_id", right_on="id")
    pivot = merged.groupby(["entity_name", "signal_category"]).size().reset_index(name="count")

    fig = go.Figure()
    for cat in ["Competitor_Strategy", "Broker_Dynamics", "Emerging_Risks"]:
        sub = pivot[pivot["signal_category"] == cat]
        if not sub.empty:
            fig.add_trace(go.Bar(
                x=sub["entity_name"], y=sub["count"],
                name=_CAT_LABEL.get(cat, cat),
                marker_color=_CAT_COLOR.get(cat, _C["blue"]),
            ))
    fig.update_layout(
        barmode="stack", title="Competitor Strategic Activity by Category",
        template="plotly_white", height=400,
        font=dict(family="Inter, sans-serif", color=_C["navy"]),
        legend=dict(orientation="h", y=-0.15),
    )

    comp_signals = signals[signals["signal_category"] == "Competitor_Strategy"].copy()
    table_html = _make_news_table_html(comp_signals)
    return f'<h2>Competitor Overview</h2>{fig.to_html(full_html=False, include_plotlyjs=False)}<h3>Key Competitor News</h3>{table_html}'


def _build_emerging_risks_page(signals: pd.DataFrame, entities: pd.DataFrame) -> str:
    """Page 2: Heatmap + news feed."""
    risk = signals[signals["signal_category"] == "Emerging_Risks"].copy()
    if risk.empty:
        return "<h2>Emerging Risks</h2><p>No emerging risk signals yet.</p>"

    risk_ent = entities[entities["entity_type"] == "Risk"]
    if risk_ent.empty:
        # Fallback: use title as entity
        risk["entity"] = risk["title"].str[:40]
    else:
        merged = risk_ent.merge(risk[["id", "year_week", "sentiment_score"]],
                                left_on="signal_id", right_on="id")
        risk_agg = merged.groupby(["entity_name", "year_week"]).agg(
            count=("signal_id", "size"),
            avg_sent=("sentiment_score", "mean")
        ).reset_index()

        if risk_agg.empty:
            risk["entity"] = risk["title"].str[:40]
        else:
            # Build heatmap matrix
            heat_pivot = risk_agg.pivot_table(index="entity_name", columns="year_week",
                                              values="count", fill_value=0)
            fig = go.Figure(data=go.Heatmap(
                z=heat_pivot.values,
                x=heat_pivot.columns.tolist(),
                y=heat_pivot.index.tolist(),
                colorscale=[[0, _C["white"]], [0.5, _C["amber"]], [1, _C["red"]]],
                hovertemplate="Risk: %{y}<br>Period: %{x}<br>Signals: %{z}<extra></extra>",
            ))
            fig.update_layout(
                title="Emerging Risk Heat Map",
                template="plotly_white", height=max(300, len(heat_pivot) * 45 + 100),
                font=dict(family="Inter, sans-serif", color=_C["navy"]),
                yaxis=dict(autorange="reversed"),
            )
            table_html = _make_news_table_html(risk)
            return f'<h2>Emerging Risks</h2>{fig.to_html(full_html=False, include_plotlyjs=False)}<h3>Key Risk Signals</h3>{table_html}'

    # Simple fallback heatmap by title × week
    heat_data = risk.groupby(["title", "year_week"]).size().reset_index(name="count")
    heat_pivot = heat_data.pivot_table(index="title", columns="year_week", values="count", fill_value=0)
    fig = go.Figure(data=go.Heatmap(
        z=heat_pivot.values, x=heat_pivot.columns.tolist(), y=heat_pivot.index.tolist(),
        colorscale=[[0, _C["white"]], [0.5, _C["amber"]], [1, _C["red"]]],
    ))
    fig.update_layout(title="Emerging Risk Heat Map", template="plotly_white",
                      height=max(300, len(heat_pivot) * 45 + 100),
                      font=dict(family="Inter, sans-serif", color=_C["navy"]),
                      yaxis=dict(autorange="reversed"))
    table_html = _make_news_table_html(risk)
    return f'<h2>Emerging Risks</h2>{fig.to_html(full_html=False, include_plotlyjs=False)}<h3>Key Risk Signals</h3>{table_html}'


def _build_broker_page(signals: pd.DataFrame, entities: pd.DataFrame) -> str:
    """Page 3: Bubble chart + news feed."""
    broker = signals[signals["signal_category"] == "Broker_Dynamics"].copy()
    if broker.empty:
        return "<h2>Broker Dynamics</h2><p>No broker signals yet.</p>"

    broker_ent = entities[entities["entity_type"] == "Broker"]
    if broker_ent.empty:
        broker["entity"] = broker["title"].str[:30]
        agg = broker.groupby("entity").agg(
            count=("id", "size"),
            avg_sent=("sentiment_score", "mean"),
            actions=("action_required", "sum"),
        ).reset_index()
    else:
        merged = broker_ent.merge(
            broker[["id", "sentiment_score", "action_required"]],
            left_on="signal_id", right_on="id")
        agg = merged.groupby("entity_name").agg(
            count=("signal_id", "size"),
            avg_sent=("sentiment_score", "mean"),
            actions=("action_required", "sum"),
        ).reset_index().rename(columns={"entity_name": "entity"})

    agg["bubble_size"] = (agg["actions"] + 1) * 15

    fig = go.Figure(data=go.Scatter(
        x=agg["avg_sent"], y=agg["count"],
        mode="markers+text", text=agg["entity"], textposition="top center",
        marker=dict(size=agg["bubble_size"], color=_C["blue"],
                    opacity=0.7, line=dict(width=1, color=_C["navy"])),
        hovertemplate="<b>%{text}</b><br>Sentiment: %{x:.2f}<br>Signals: %{y}<br>Actions: %{marker.size}<extra></extra>",
    ))
    fig.update_layout(
        title="Broker Activity Matrix",
        xaxis_title="Market Sentiment (−1 Bearish → +1 Bullish)",
        yaxis_title="Signal Volume",
        template="plotly_white", height=450,
        font=dict(family="Inter, sans-serif", color=_C["navy"]),
    )
    fig.add_vline(x=0, line_dash="dash", line_color=_C["gray"], opacity=0.4)

    table_html = _make_news_table_html(broker)
    return f'<h2>Broker Dynamics</h2>{fig.to_html(full_html=False, include_plotlyjs=False)}<h3>Key Broker News</h3>{table_html}'


def _build_sentiment_page(signals: pd.DataFrame) -> str:
    """Page 4: Sentiment stacked bar + trend line."""
    if signals.empty:
        return "<h2>Sentiment Analysis</h2><p>No signals yet.</p>"

    # Stacked bar: sentiment distribution per category
    sent_counts = signals.groupby(["category_label", "sentiment"]).size().reset_index(name="count")
    fig1 = go.Figure()
    for sent in ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"]:
        sub = sent_counts[sent_counts["sentiment"] == sent]
        if not sub.empty:
            fig1.add_trace(go.Bar(
                x=sub["category_label"], y=sub["count"], name=sent,
                marker_color=_SENT_COLOR.get(sent, _C["gray"]),
            ))
    fig1.update_layout(
        barmode="stack", title="Sentiment Distribution by Category",
        template="plotly_white", height=380,
        font=dict(family="Inter, sans-serif", color=_C["navy"]),
        legend=dict(orientation="h", y=-0.15),
    )

    # Trend line: avg sentiment by date per category
    trend = signals.groupby(["day", "category_label"])["sentiment_score"].mean().reset_index()
    fig2 = go.Figure()
    for cat in trend["category_label"].unique():
        sub = trend[trend["category_label"] == cat].sort_values("day")
        cat_key = [k for k, v in _CAT_LABEL.items() if v == cat]
        color = _CAT_COLOR.get(cat_key[0], _C["blue"]) if cat_key else _C["blue"]
        fig2.add_trace(go.Scatter(
            x=sub["day"], y=sub["sentiment_score"], name=cat,
            mode="lines+markers", line=dict(color=color, width=2),
            marker=dict(size=6),
        ))
    fig2.update_layout(
        title="Sentiment Trend Over Time",
        xaxis_title="Date", yaxis_title="Avg Sentiment Score",
        template="plotly_white", height=350,
        font=dict(family="Inter, sans-serif", color=_C["navy"]),
        legend=dict(orientation="h", y=-0.2),
    )
    fig2.add_hline(y=0, line_dash="dash", line_color=_C["gray"], opacity=0.4)

    return (f'<h2>Sentiment Analysis</h2>'
            f'{fig1.to_html(full_html=False, include_plotlyjs=False)}'
            f'{fig2.to_html(full_html=False, include_plotlyjs=False)}')


def _build_timeline_page(signals: pd.DataFrame) -> str:
    """Page 5: News tables by month, week, day."""
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    this_week = datetime.now(tz=timezone.utc).strftime("%Y-W%V")
    this_month = datetime.now(tz=timezone.utc).strftime("%Y-%m")

    month_df = signals[signals["year_month"] == this_month]
    week_df = signals[signals["year_week"] == this_week]
    day_df = signals[signals["day"] == today]

    return (f'<h2>News Timeline</h2>'
            f'<h3>📅 This Month ({this_month})</h3>{_make_news_table_html(month_df)}'
            f'<h3>📆 This Week ({this_week})</h3>{_make_news_table_html(week_df)}'
            f'<h3>📌 Today ({today})</h3>{_make_news_table_html(day_df)}')


def _build_html_dashboard(signals: pd.DataFrame, entities: pd.DataFrame) -> str:
    """Assemble the full HTML dashboard with all 5 pages as tabs."""
    pages = [
        ("Competitor Overview", _build_competitor_page(signals, entities)),
        ("Emerging Risks", _build_emerging_risks_page(signals, entities)),
        ("Broker Dynamics", _build_broker_page(signals, entities)),
        ("Sentiment Analysis", _build_sentiment_page(signals)),
        ("News Timeline", _build_timeline_page(signals)),
    ]

    tabs_html = ""
    panels_html = ""
    for i, (label, content) in enumerate(pages):
        active = " active" if i == 0 else ""
        tabs_html += f'<button class="tab-btn{active}" onclick="showTab({i})">{label}</button>\n'
        display = "block" if i == 0 else "none"
        panels_html += f'<div class="tab-panel" id="panel-{i}" style="display:{display}">{content}</div>\n'

    run_ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Intelligence Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',sans-serif; background:{_C['bg']}; color:{_C['navy']}; }}
  .header {{ background:linear-gradient(135deg,{_C['navy']},{_C['blue']}); color:#fff;
             padding:24px 40px; }}
  .header h1 {{ font-size:24px; font-weight:700; }}
  .header p {{ font-size:12px; opacity:0.7; margin-top:4px; }}
  .tabs {{ display:flex; gap:0; background:{_C['white']}; border-bottom:2px solid {_C['light_blue']};
           padding:0 40px; position:sticky; top:0; z-index:10; }}
  .tab-btn {{ padding:14px 24px; border:none; background:transparent; cursor:pointer;
              font-family:inherit; font-size:14px; font-weight:600; color:{_C['gray']};
              border-bottom:3px solid transparent; transition:all 0.2s; }}
  .tab-btn:hover {{ color:{_C['navy']}; background:{_C['bg']}; }}
  .tab-btn.active {{ color:{_C['navy']}; border-bottom-color:{_C['blue']}; }}
  .tab-panel {{ padding:30px 40px; }}
  .tab-panel h2 {{ font-size:20px; margin-bottom:16px; padding-bottom:8px;
                   border-bottom:2px solid {_C['light_blue']}; }}
  .tab-panel h3 {{ font-size:15px; margin:20px 0 10px; color:{_C['gray']}; }}
  .news-tbl {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:8px; }}
  .news-tbl th {{ background:{_C['navy']}; color:#fff; padding:10px 8px; text-align:left;
                  font-weight:600; position:sticky; top:52px; }}
  .news-tbl td {{ padding:8px; border-bottom:1px solid {_C['light_blue']}; vertical-align:top; }}
  .news-tbl tr:nth-child(even) {{ background:{_C['bg']}; }}
  .news-tbl tr:hover {{ background:{_C['light_blue']}; }}
</style></head><body>
<div class="header">
  <h1>📊 Strategic Market Intelligence Dashboard</h1>
  <p>Generated {run_ts}  •  {len(signals)} signals across {signals['signal_category'].nunique()} categories</p>
</div>
<div class="tabs">{tabs_html}</div>
{panels_html}
<script>
function showTab(idx) {{
  document.querySelectorAll('.tab-panel').forEach((p,i) => p.style.display = i===idx ? 'block' : 'none');
  document.querySelectorAll('.tab-btn').forEach((b,i) => b.classList.toggle('active', i===idx));
  window.dispatchEvent(new Event('resize'));
}}
</script></body></html>"""


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def export_powerbi_report(output_path: Optional[str] = None) -> Path:
    """
    Export intelligence data as an interactive HTML dashboard + Parquet files.

    Returns:
        Path to the generated HTML dashboard file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    if output_path:
        html_path = Path(output_path)
    else:
        html_path = OUTPUT_DIR / f"dashboard_{timestamp}.html"

    html_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building Power BI dashboard export...")

    # Load data
    signals = _load_signals()
    entities = _build_entities_df(signals)

    if signals.empty:
        logger.warning("No signals to export.")
        html_path.write_text("<html><body><h1>No data available</h1></body></html>")
        return html_path

    # Export Parquet (for Power BI Desktop)
    _export_parquet(signals, entities, OUTPUT_DIR)

    # Build interactive HTML dashboard
    html = _build_html_dashboard(signals, entities)
    html_path.write_text(html, encoding="utf-8")

    logger.info("Dashboard exported: %s  (%d signals, %d entities)",
                html_path, len(signals), len(entities))
    return html_path
