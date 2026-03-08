from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.express as px


COLOR_MAP = {
    "positive": "#2ec4b6",
    "neutral": "#f4d35e",
    "negative": "#ff6b6b",
}
CHART_FONT_FAMILY = "IBM Plex Sans, sans-serif"
CHART_THEMES = {
    "dark": {
        "title": "#f5f7ff",
        "font": "#d9e5ff",
        "plot": "rgba(10, 22, 40, 0.78)",
        "grid": "rgba(138, 161, 198, 0.14)",
        "zero": "rgba(138, 161, 198, 0.10)",
        "axis": "#acc2df",
        "marker_line": "rgba(8, 16, 32, 0.95)",
    },
    "light": {
        "title": "#10253f",
        "font": "#23384e",
        "plot": "rgba(255, 255, 255, 0.9)",
        "grid": "rgba(75, 109, 143, 0.12)",
        "zero": "rgba(75, 109, 143, 0.08)",
        "axis": "#35516f",
        "marker_line": "rgba(236, 243, 250, 0.95)",
    },
}


def apply_chart_chrome(figure, title: str, theme_mode: str = "dark"):
    chrome = CHART_THEMES.get(str(theme_mode).lower(), CHART_THEMES["dark"])
    figure.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 20, "family": "Space Grotesk, sans-serif", "color": chrome["title"]},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=chrome["plot"],
        font={"family": CHART_FONT_FAMILY, "color": chrome["font"], "size": 13},
        margin={"l": 24, "r": 24, "t": 72, "b": 24},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "title": {"text": ""},
            "font": {"size": 12},
        },
    )
    figure.update_xaxes(showgrid=False, color=chrome["axis"], tickfont={"size": 12})
    figure.update_yaxes(
        gridcolor=chrome["grid"],
        zerolinecolor=chrome["zero"],
        color=chrome["axis"],
        tickfont={"size": 12},
    )
    return figure


def sentiment_pie_chart(sentiment_distribution: dict[str, float], theme_mode: str = "dark"):
    chrome = CHART_THEMES.get(str(theme_mode).lower(), CHART_THEMES["dark"])
    frame = pd.DataFrame(
        {
            "sentiment": list(sentiment_distribution.keys()),
            "share": [value * 100 for value in sentiment_distribution.values()],
        }
    )
    figure = px.pie(
        frame,
        names="sentiment",
        values="share",
        color="sentiment",
        color_discrete_map=COLOR_MAP,
        hole=0.66,
    )
    figure.update_traces(
        sort=False,
        textposition="inside",
        textinfo="percent",
        marker={"line": {"color": chrome["marker_line"], "width": 2}},
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    )
    figure.add_annotation(
        text="Sentiment<br>Mix",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"family": "Space Grotesk, sans-serif", "size": 18, "color": chrome["title"]},
    )
    return apply_chart_chrome(figure, "Sentiment Distribution", theme_mode=theme_mode)


def sentiment_trend_chart(frame: pd.DataFrame, theme_mode: str = "dark"):
    trend = (
        frame.assign(date=pd.to_datetime(frame["date"], errors="coerce"))
        .dropna(subset=["date"])
        .groupby(["date", "sentiment"])
        .size()
        .reset_index(name="review_count")
    )
    if trend.empty:
        return None

    figure = px.line(
        trend,
        x="date",
        y="review_count",
        color="sentiment",
        markers=True,
        color_discrete_map=COLOR_MAP,
    )
    figure.update_traces(line={"width": 3}, marker={"size": 8}, mode="lines+markers")
    figure.update_yaxes(title_text="Reviews")
    figure.update_xaxes(title_text="")
    return apply_chart_chrome(figure, "Sentiment Trend Over Time", theme_mode=theme_mode)


def theme_importance_chart(topics: list[dict], title: str, color: str, theme_mode: str = "dark"):
    rows = [
        {
            "theme": topic.get("label") or f"Topic {index}",
            "prevalence": topic.get("prevalence", 0.0) * 100,
        }
        for index, topic in enumerate(topics, start=1)
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None

    frame = frame.sort_values("prevalence", ascending=True)
    figure = px.bar(
        frame,
        x="prevalence",
        y="theme",
        orientation="h",
        text="prevalence",
    )
    figure.update_traces(
        marker={"color": color},
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    )
    figure.update_layout(showlegend=False)
    figure.update_xaxes(title_text="Theme prevalence (%)")
    figure.update_yaxes(title_text="")
    return apply_chart_chrome(figure, title, theme_mode=theme_mode)


def keyword_cluster_chart(
    topics: list[dict],
    title: str,
    color_sequence: Sequence[str] | None = None,
    theme_mode: str = "dark",
):
    chrome = CHART_THEMES.get(str(theme_mode).lower(), CHART_THEMES["dark"])
    rows: list[dict[str, object]] = []
    for topic in topics:
        label = topic.get("label") or f"Topic {topic.get('topic_id', 0) + 1}"
        prevalence = float(topic.get("prevalence", 0.0))
        for rank, keyword in enumerate(topic.get("keywords", [])[:6], start=1):
            weight = prevalence * (7 - rank) * 100
            rows.append({"theme": label, "keyword": keyword, "weight": max(weight, 0.1)})

    frame = pd.DataFrame(rows)
    if frame.empty:
        return None

    figure = px.treemap(
        frame,
        path=["theme", "keyword"],
        values="weight",
        color="theme",
        color_discrete_sequence=list(color_sequence) if color_sequence else None,
    )
    figure.update_traces(
        textinfo="label+value",
        hovertemplate="%{label}<br>Weighted importance: %{value:.1f}<extra></extra>",
        marker={"line": {"color": chrome["marker_line"], "width": 1}},
    )
    figure.update_layout(margin={"l": 12, "r": 12, "t": 72, "b": 12})
    return apply_chart_chrome(figure, title, theme_mode=theme_mode)


def topic_keyword_chart(topics: list[dict], title: str, theme_mode: str = "dark"):
    return theme_importance_chart(topics, title, "#7cc4ff", theme_mode=theme_mode)
