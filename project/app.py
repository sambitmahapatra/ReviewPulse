from __future__ import annotations

import html
import io
import json
from textwrap import fill
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

from project.pipeline.ingestion import (
    UNMAPPED_COLUMN,
    normalize_play_store_frame,
    normalize_uploaded_frame,
    suggest_column_mapping,
)
from project.pipeline.llm_analysis import LLMAnalyzer
from project.pipeline.orchestrator import build_analysis_bundle
from project.pipeline.preprocess import preprocess_reviews_frame
from project.pipeline.scraper import MAX_ALLOWED_REVIEWS, fetch_reviews_dataframe
from project.pipeline.sentiment import (
    DEFAULT_SENTIMENT_MODEL,
    DEFAULT_SENTIMENT_MODEL_KEY,
    SENTIMENT_MODEL_PROFILES,
    SentimentAnalyzer,
    get_sentiment_model_profile,
)
from project.utils.helpers import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_OPENAI_MODEL,
    extract_app_id,
    format_percentage,
    generic_review_noise_stopwords,
    get_secret_value,
    safe_filename,
)
from project.visualization.charts import (
    keyword_cluster_chart,
    sentiment_pie_chart,
    sentiment_trend_chart,
    theme_importance_chart,
)
from project.visualization.wordcloud import generate_wordcloud_figure


APP_NAME = "ReviewPulse"
APP_TAGLINE = "Real-time user sentiment signals & Review Intelligence"
SAMPLE_URL = "https://play.google.com/store/apps/details?id=in.startv.hotstar"
SOURCE_PLAY_STORE = "Google Play Store"
SOURCE_CSV_UPLOAD = "CSV Upload"
OUR_DEFAULT_LLM = "Our Default LLM"
YOUR_OWN_LLM = "Your Own LLM"
DISABLE_LLM = "Disable LLM"
UI_THEME_DARK = "Dark"
UI_THEME_LIGHT = "Light"
PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False, "responsive": True}
CSV_MAPPING_FIELDS = ["review_text", "date", "rating", "likes", "product_name", "product_id"]
CSV_MAPPING_LABELS = {
    "review_text": "Review text column",
    "date": "Review date column",
    "rating": "Rating column",
    "likes": "Helpful votes / likes column",
    "product_name": "Product name column",
    "product_id": "Product id column",
}
OWN_LLM_MODELS = {
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ],
    "openai": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"],
}


def get_ui_palette(theme_mode: str) -> dict[str, str]:
    if str(theme_mode).lower() == "light":
        return {
            "scheme": "light",
            "page_background": """
                radial-gradient(circle at top right, rgba(66, 196, 184, 0.16), transparent 28%),
                radial-gradient(circle at top left, rgba(79, 127, 209, 0.14), transparent 30%),
                linear-gradient(180deg, #f5f9ff 0%, #eef4fb 100%)
            """,
            "sidebar_background": "linear-gradient(180deg, rgba(244, 248, 253, 0.98) 0%, rgba(236, 243, 250, 0.98) 100%)",
            "text_primary": "#13263c",
            "text_secondary": "#53697f",
            "panel_background": "linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(247, 251, 255, 0.96) 100%)",
            "hero_background": "linear-gradient(135deg, rgba(255, 255, 255, 0.97) 0%, rgba(240, 247, 252, 0.97) 60%, rgba(233, 244, 250, 0.98) 100%)",
            "panel_border": "rgba(83, 109, 136, 0.16)",
            "panel_shadow": "0 16px 38px rgba(55, 84, 117, 0.08)",
            "hero_shadow": "0 18px 48px rgba(55, 84, 117, 0.10)",
            "accent_warm": "#b57900",
            "accent_warm_bg": "rgba(181, 121, 0, 0.10)",
            "accent_warm_border": "rgba(181, 121, 0, 0.18)",
            "tab_background": "rgba(255, 255, 255, 0.92)",
            "tab_selected_background": "linear-gradient(135deg, rgba(46, 196, 182, 0.16) 0%, rgba(22, 137, 154, 0.12) 100%)",
            "tab_text": "#53697f",
            "tab_selected_text": "#13263c",
            "input_background": "rgba(255, 255, 255, 0.95)",
            "input_border": "rgba(90, 121, 153, 0.20)",
            "plot_background": "rgba(255, 255, 255, 0.92)",
            "muted_line": "linear-gradient(90deg, rgba(110, 141, 174, 0.28), rgba(110, 141, 174, 0.04))",
            "info_text": "#5d7690",
            "chip_positive_bg": "rgba(46, 196, 182, 0.14)",
            "chip_positive_text": "#1f7b78",
            "chip_positive_border": "rgba(46, 196, 182, 0.28)",
            "chip_negative_bg": "rgba(255, 107, 107, 0.12)",
            "chip_negative_text": "#b64b4b",
            "chip_negative_border": "rgba(255, 107, 107, 0.24)",
            "widget_label": "#314c68",
            "button_text": "#13263c",
        }

    return {
        "scheme": "dark",
        "page_background": """
            radial-gradient(circle at top right, rgba(17, 93, 97, 0.34), transparent 28%),
            radial-gradient(circle at top left, rgba(22, 74, 133, 0.28), transparent 28%),
            linear-gradient(180deg, #041120 0%, #03111c 100%)
        """,
        "sidebar_background": "linear-gradient(180deg, rgba(4, 14, 29, 0.98) 0%, rgba(3, 14, 26, 0.98) 100%)",
        "text_primary": "#f5f7ff",
        "text_secondary": "#a8b9d4",
        "panel_background": "linear-gradient(180deg, rgba(7, 18, 33, 0.96) 0%, rgba(7, 22, 38, 0.92) 100%)",
        "hero_background": "linear-gradient(135deg, rgba(6, 17, 33, 0.94) 0%, rgba(4, 24, 47, 0.94) 60%, rgba(5, 28, 42, 0.96) 100%)",
        "panel_border": "rgba(112, 160, 203, 0.16)",
        "panel_shadow": "0 16px 38px rgba(0, 0, 0, 0.16)",
        "hero_shadow": "0 18px 48px rgba(0, 0, 0, 0.28)",
        "accent_warm": "#ffd67f",
        "accent_warm_bg": "rgba(255, 214, 127, 0.12)",
        "accent_warm_border": "rgba(255, 214, 127, 0.18)",
        "tab_background": "rgba(8, 22, 39, 0.92)",
        "tab_selected_background": "linear-gradient(135deg, rgba(46, 196, 182, 0.18) 0%, rgba(22, 137, 154, 0.18) 100%)",
        "tab_text": "#a8b9d4",
        "tab_selected_text": "#f5f7ff",
        "input_background": "rgba(8, 22, 39, 0.95)",
        "input_border": "rgba(112, 160, 203, 0.18)",
        "plot_background": "rgba(7, 18, 33, 0.88)",
        "muted_line": "linear-gradient(90deg, rgba(112, 160, 203, 0.25), rgba(112, 160, 203, 0.03))",
        "info_text": "#87a6cc",
        "chip_positive_bg": "rgba(46, 196, 182, 0.12)",
        "chip_positive_text": "#9ef0e6",
        "chip_positive_border": "rgba(46, 196, 182, 0.20)",
        "chip_negative_bg": "rgba(255, 107, 107, 0.11)",
        "chip_negative_text": "#ffb8b8",
        "chip_negative_border": "rgba(255, 107, 107, 0.22)",
        "widget_label": "#d3def0",
        "button_text": "#04111d",
    }


def read_secret_value(secret_name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(secret_name, default)
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    return get_secret_value(secret_name, default)


def resolve_platform_llm_config() -> dict[str, str | bool]:
    provider = read_secret_value("PLATFORM_LLM_PROVIDER")
    model = read_secret_value("PLATFORM_LLM_MODEL")
    api_key = read_secret_value("PLATFORM_LLM_API_KEY")

    groq_key = read_secret_value("GROQ_API_KEY")
    openai_key = read_secret_value("OPENAI_API_KEY")

    if not provider:
        if groq_key:
            provider = "groq"
        elif openai_key:
            provider = "openai"

    if not api_key:
        if provider == "groq":
            api_key = groq_key
        elif provider == "openai":
            api_key = openai_key

    if not model:
        if provider == "groq":
            model = DEFAULT_GROQ_MODEL
        elif provider == "openai":
            model = DEFAULT_OPENAI_MODEL

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "configured": bool(provider and api_key),
    }


@st.cache_resource(show_spinner=False)
def load_sentiment_analyzer(profile_key: str) -> SentimentAnalyzer:
    profile = get_sentiment_model_profile(profile_key)
    return SentimentAnalyzer(
        model_name=str(profile["model_name"]),
        max_length=int(profile["max_length"]),
        label_map=dict(profile["label_map"]),
    )


@st.cache_data(show_spinner=False)
def fetch_play_store_reviews_cached(
    app_id: str,
    max_reviews: int,
    lang: str,
    country: str,
    sort_by: str,
) -> pd.DataFrame:
    return fetch_reviews_dataframe(
        app_id=app_id,
        max_reviews=max_reviews,
        lang=lang,
        country=country,
        sort_by=sort_by,
    )


@st.cache_data(show_spinner=False)
def preprocess_reviews_cached(frame: pd.DataFrame, preserve_terms: tuple[str, ...]) -> pd.DataFrame:
    return preprocess_reviews_frame(frame, preserve_terms=set(preserve_terms))


@st.cache_data(show_spinner=False)
def score_reviews_cached(frame: pd.DataFrame, profile_key: str, batch_size: int) -> tuple[pd.DataFrame, dict[str, float]]:
    analyzer = load_sentiment_analyzer(profile_key)
    return analyzer.score_frame(frame, batch_size=batch_size)


def build_llm_analyzer(
    access_mode: str,
    own_provider: str,
    own_model: str,
    own_api_key: str,
    timeout_seconds: float,
) -> tuple[LLMAnalyzer, dict[str, str | bool]]:
    if access_mode == OUR_DEFAULT_LLM:
        config = resolve_platform_llm_config()
        return (
            LLMAnalyzer(
                provider=str(config["provider"] or "none"),
                api_key=str(config["api_key"] or ""),
                model_name=str(config["model"] or ""),
                timeout_seconds=timeout_seconds,
            ),
            config,
        )

    if access_mode == YOUR_OWN_LLM:
        return (
            LLMAnalyzer(
                provider=own_provider,
                api_key=own_api_key,
                model_name=own_model,
                timeout_seconds=timeout_seconds,
            ),
            {"provider": own_provider, "model": own_model, "api_key": own_api_key, "configured": bool(own_api_key)},
        )

    return (
        LLMAnalyzer(provider="none", api_key="", model_name="", timeout_seconds=timeout_seconds),
        {"provider": "none", "model": "", "api_key": "", "configured": False},
    )


def format_llm_display_label(access_mode: str, provider: str, model: str) -> str:
    provider_value = str(provider or "").strip()
    model_value = str(model or "").strip()
    provider_label_map = {"openai": "OpenAI", "groq": "Groq"}
    provider_label = provider_label_map.get(provider_value.lower(), provider_value.title())

    if access_mode == DISABLE_LLM or provider_value.lower() == "none":
        return "Disabled"
    if provider_value and model_value:
        return f"{provider_label} {model_value}"
    if model_value:
        return model_value
    if provider_value:
        return provider_label
    return access_mode


def render_styles(theme_mode: str) -> None:
    palette = get_ui_palette(theme_mode)
    theme_variables = f"""
        :root {{
            --text-primary: {palette["text_primary"]};
            --text-secondary: {palette["text_secondary"]};
            --teal: #2ec4b6;
            --gold: #f4d35e;
            --coral: #ff6b6b;
            --shadow: {palette["hero_shadow"]};
            --card-shadow: {palette["panel_shadow"]};
            --radius: 18px;
            --page-bg: {palette["page_background"]};
            --sidebar-bg: {palette["sidebar_background"]};
            --panel-bg: {palette["panel_background"]};
            --hero-bg: {palette["hero_background"]};
            --panel-border: {palette["panel_border"]};
            --accent-warm: {palette["accent_warm"]};
            --accent-warm-bg: {palette["accent_warm_bg"]};
            --accent-warm-border: {palette["accent_warm_border"]};
            --tab-bg: {palette["tab_background"]};
            --tab-selected-bg: {palette["tab_selected_background"]};
            --tab-text: {palette["tab_text"]};
            --tab-selected-text: {palette["tab_selected_text"]};
            --input-bg: {palette["input_background"]};
            --input-border: {palette["input_border"]};
            --plot-bg: {palette["plot_background"]};
            --divider-line: {palette["muted_line"]};
            --meta-text: {palette["info_text"]};
            --chip-positive-bg: {palette["chip_positive_bg"]};
            --chip-positive-text: {palette["chip_positive_text"]};
            --chip-positive-border: {palette["chip_positive_border"]};
            --chip-negative-bg: {palette["chip_negative_bg"]};
            --chip-negative-text: {palette["chip_negative_text"]};
            --chip-negative-border: {palette["chip_negative_border"]};
            --widget-label: {palette["widget_label"]};
            --button-text: {palette["button_text"]};
        }}
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
        """
        + theme_variables
        + """

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
            color-scheme: """
        + palette["scheme"]
        + """;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--page-bg);
            color: var(--text-primary);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            right: 0.75rem;
            top: 0.35rem;
        }

        [data-testid="stDecoration"],
        footer {
            display: none !important;
        }

        [data-testid="stSidebar"] {
            background: var(--sidebar-bg);
            border-right: 1px solid var(--panel-border);
        }

        [data-testid="stSidebar"] * {
            color: var(--text-primary);
        }

        .block-container {
            max-width: 1380px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
            padding-left: 2.4rem;
            padding-right: 2.4rem;
        }

        h1, h2, h3, h4 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.02em;
        }

        .hero-shell {
            padding: 2.4rem 2.4rem 2rem 2.4rem;
            border: 1px solid var(--panel-border);
            border-radius: 26px;
            background: var(--hero-bg);
            box-shadow: var(--shadow);
            margin-bottom: 1.5rem;
            overflow: hidden;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.5rem 0.95rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--accent-warm);
            background: var(--accent-warm-bg);
            border: 1px solid var(--accent-warm-border);
        }

        .hero-title {
            margin: 1.35rem 0 0.5rem 0;
            font-size: clamp(3.1rem, 5.8vw, 5.2rem);
            line-height: 0.96;
        }

        .hero-copy {
            max-width: 58rem;
            font-size: 1.22rem;
            line-height: 1.8;
            color: var(--text-secondary);
            margin-bottom: 1.55rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }

        .hero-tile,
        .kpi-card,
        .story-card,
        .insight-card,
        .theme-card,
        .explorer-shell,
        .input-shell,
        .download-card {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: var(--radius);
            box-shadow: var(--card-shadow);
        }

        .hero-tile {
            padding: 1.25rem 1.3rem;
        }

        .hero-tile-label,
        .section-kicker,
        .kpi-label,
        .insight-banner-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--accent-warm);
            font-weight: 700;
        }

        .hero-tile-value {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0.5rem 0 0.35rem 0;
        }

        .hero-tile-copy,
        .section-copy,
        .kpi-copy,
        .story-copy,
        .microcopy,
        .download-copy {
            color: var(--text-secondary);
            line-height: 1.68;
        }

        .section-title {
            font-size: 2.2rem;
            line-height: 1.08;
            margin-bottom: 0.5rem;
        }

        .section-copy {
            font-size: 1.02rem;
            margin-bottom: 1.1rem;
            max-width: 54rem;
        }

        .divider {
            height: 1px;
            background: var(--divider-line);
            margin: 1.35rem 0 1.4rem 0;
        }

        .kpi-card {
            min-height: 168px;
            padding: 1.35rem 1.35rem 1.15rem 1.35rem;
        }

        .kpi-label {
            color: var(--text-secondary);
            margin-bottom: 1rem;
        }

        .kpi-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.3rem;
            line-height: 1;
            margin-bottom: 0.7rem;
            color: var(--text-primary);
        }

        .insight-banner {
            padding: 1.1rem 1.3rem;
            border-radius: 18px;
            border: 1px solid var(--accent-warm-border);
            background: linear-gradient(135deg, color-mix(in srgb, var(--panel-bg) 82%, var(--teal) 18%) 0%, color-mix(in srgb, var(--panel-bg) 76%, var(--gold) 14%) 100%);
            box-shadow: var(--shadow);
            margin: 0.4rem 0 1.5rem 0;
        }

        .insight-banner-copy {
            font-size: 1.08rem;
            line-height: 1.75;
            color: var(--text-primary);
            margin-top: 0.45rem;
        }

        .summary-lead-card,
        .summary-engine-card {
            padding: 1.35rem 1.4rem;
            border-radius: var(--radius);
            border: 1px solid var(--panel-border);
            box-shadow: var(--card-shadow);
            min-height: 220px;
        }

        .summary-lead-card {
            background: linear-gradient(135deg, color-mix(in srgb, var(--panel-bg) 82%, var(--teal) 18%) 0%, var(--panel-bg) 100%);
        }

        .summary-engine-card {
            background: linear-gradient(135deg, color-mix(in srgb, var(--panel-bg) 88%, var(--gold) 12%) 0%, var(--panel-bg) 100%);
        }

        .summary-overline {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--accent-warm);
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .summary-headline {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.55rem;
            line-height: 1.28;
            margin-bottom: 0.75rem;
            color: var(--text-primary);
        }

        .summary-body {
            color: var(--text-secondary);
            line-height: 1.72;
            font-size: 0.98rem;
            margin-bottom: 1rem;
        }

        .summary-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }

        .summary-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.45rem 0.72rem;
            border-radius: 999px;
            font-size: 0.86rem;
            background: color-mix(in srgb, var(--panel-bg) 78%, white 22%);
            border: 1px solid var(--panel-border);
            color: var(--text-primary);
        }

        .story-card,
        .insight-card,
        .theme-card,
        .explorer-shell,
        .input-shell {
            padding: 1.25rem 1.35rem;
        }

        .theme-card {
            min-height: 280px;
            margin-bottom: 1rem;
        }

        .theme-card.positive {
            border-color: rgba(46, 196, 182, 0.24);
        }

        .theme-card.negative {
            border-color: rgba(255, 107, 107, 0.24);
        }

        .insight-card {
            position: relative;
            overflow: hidden;
        }

        .insight-card::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, var(--teal), var(--gold));
            opacity: 0.9;
        }

        .theme-meta {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.9rem;
            color: var(--text-secondary);
            font-size: 0.88rem;
        }

        .theme-title {
            font-size: 1.45rem;
            margin: 0 0 0.85rem 0;
            line-height: 1.2;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.86rem;
            border: 1px solid rgba(112, 160, 203, 0.14);
        }

        .chip.positive {
            background: var(--chip-positive-bg);
            color: var(--chip-positive-text);
            border-color: var(--chip-positive-border);
        }

        .chip.negative {
            background: var(--chip-negative-bg);
            color: var(--chip-negative-text);
            border-color: var(--chip-negative-border);
        }

        .theme-quotes {
            margin: 0;
            padding-left: 1.2rem;
            color: var(--text-secondary);
            line-height: 1.68;
        }

        .theme-quotes li {
            margin-bottom: 0.62rem;
        }

        .input-title,
        .download-title {
            font-size: 1.18rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .input-copy {
            color: var(--text-secondary);
            margin-bottom: 0.85rem;
            line-height: 1.6;
        }

        .download-card {
            padding: 1.1rem 1.15rem;
            min-height: 122px;
        }

        button[kind="primary"], .stButton > button {
            width: 100%;
            min-height: 46px;
            border-radius: 14px;
            font-weight: 700;
            box-shadow: none !important;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2ec4b6 0%, #16899a 100%);
            color: var(--button-text);
        }

        .stButton > button[kind="secondary"] {
            background: var(--panel-bg);
            color: var(--text-primary);
            border: 1px solid var(--panel-border);
        }

        .stDownloadButton > button {
            width: 100%;
            min-height: 46px;
            border-radius: 14px;
            font-weight: 700;
            background: linear-gradient(135deg, #2ec4b6 0%, #16899a 100%);
            color: var(--button-text);
            border: none;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stNumberInputContainer"] > div,
        [data-testid="stDateInputField"] > div {
            background: var(--input-bg) !important;
            border: 1px solid var(--input-border) !important;
            border-radius: 14px !important;
            color: var(--text-primary) !important;
        }

        input, textarea, [data-baseweb="select"] * {
            color: var(--text-primary) !important;
        }

        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span,
        [data-testid="stWidgetLabel"] label,
        label,
        .stMarkdown,
        .stCaptionContainer,
        .stTextInput label,
        .stSelectbox label,
        .stMultiSelect label,
        .stDateInput label,
        .stNumberInput label {
            color: var(--widget-label) !important;
        }

        div[role="radiogroup"] label,
        div[role="radiogroup"] label p,
        div[role="radiogroup"] label span,
        div[role="radiogroup"] [data-testid="stMarkdownContainer"] p,
        div[role="radiogroup"] [data-testid="stMarkdownContainer"] span,
        [data-baseweb="radio"] label,
        [data-baseweb="radio"] label p,
        [data-baseweb="radio"] label span,
        .stRadio label,
        .stRadio p,
        .stRadio [data-testid="stMarkdownContainer"] p {
            color: var(--text-primary) !important;
            opacity: 1 !important;
        }

        [data-testid="stAlertContainer"] p,
        [data-testid="stAlertContainer"] div,
        [data-testid="stAlertContentInfo"],
        [data-testid="stAlertContentWarning"],
        [data-testid="stAlertContentError"],
        [data-testid="stAlertContentSuccess"] {
            color: var(--text-primary) !important;
        }

        [data-testid="stAlertContainer"] {
            border-radius: 14px;
        }

        [data-testid="stStatusWidget"] {
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            overflow: hidden;
            background: var(--panel-bg);
        }

        [data-testid="stStatusWidget"] summary,
        [data-testid="stStatusWidget"] summary * {
            background: color-mix(in srgb, var(--panel-bg) 92%, black 8%);
            color: var(--text-primary) !important;
        }

        [data-testid="stStatusWidget"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stStatusWidget"] div,
        [data-testid="stStatusWidget"] span {
            color: var(--text-primary) !important;
        }

        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {
            background: var(--panel-bg);
            border-radius: 16px;
            border: 1px solid var(--panel-border);
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {
            color: var(--text-primary) !important;
            opacity: 1 !important;
        }

        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stExpander"] div,
        [data-testid="stExpander"] span {
            color: var(--text-primary) !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: color-mix(in srgb, var(--widget-label) 78%, white 22%) !important;
            opacity: 1 !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid rgba(112, 160, 203, 0.12);
            border-radius: 18px;
            overflow: hidden;
        }

        [data-testid="stPlotlyChart"] {
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            overflow: hidden;
            background: var(--plot-bg);
        }

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.6rem;
            background: transparent;
            margin-bottom: 1rem;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-border"] {
            display: none;
        }

        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            background: var(--tab-bg);
            border: 1px solid var(--panel-border);
            border-radius: 999px;
            padding: 0.62rem 1rem;
            color: var(--tab-text);
            font-weight: 600;
        }

        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            background: var(--tab-selected-bg);
            color: var(--tab-selected-text);
            border-color: rgba(46, 196, 182, 0.28);
        }

        .review-meta {
            color: var(--meta-text);
            font-size: 0.92rem;
            margin-top: -0.15rem;
            margin-bottom: 0.75rem;
        }

        @media (max-width: 960px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }

            .hero-shell {
                padding: 1.65rem;
            }

            .hero-title {
                font-size: 3rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="eyebrow">Built for live review intelligence</div>
            <h1 class="hero-title">{html.escape(APP_NAME)}</h1>
            <p class="hero-copy">
                {html.escape(APP_TAGLINE)}. Turn raw customer feedback from apps, marketplaces, and partner datasets
                into executive-ready sentiment movement, complaint themes, and product signals.
            </p>
            <div class="hero-grid">
                <div class="hero-tile">
                    <div class="hero-tile-label">Ingestion</div>
                    <div class="hero-tile-value">Google Play + CSV</div>
                    <div class="hero-tile-copy">One canonical review schema keeps downstream analysis consistent across sources.</div>
                </div>
                <div class="hero-tile">
                    <div class="hero-tile-label">Intelligence Layer</div>
                    <div class="hero-tile-value">RoBERTa + LDA + LLM</div>
                    <div class="hero-tile-copy">Transformer sentiment, topic discovery, and semantic interpretation power the dashboard narrative.</div>
                </div>
                <div class="hero-tile">
                    <div class="hero-tile-label">Delivery</div>
                    <div class="hero-tile-value">Insights, trends, exports</div>
                    <div class="hero-tile-copy">Built for product, CX, and operations teams that need signals fast without manual review triage.</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_divider() -> None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def render_section_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-kicker">{html.escape(kicker)}</div>
        <div class="section-title">{html.escape(title)}</div>
        <div class="section-copy">{html.escape(copy)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_banner(insight_text: str) -> None:
    st.markdown(
        f"""
        <div class="insight-banner">
            <div class="insight-banner-label">AI Insight</div>
            <div class="insight-banner-copy">{html.escape(insight_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_story(text: str) -> None:
    st.markdown(
        f"""
        <div class="story-card">
            <p class="story-copy">{html.escape(text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_snapshot_cards(snapshot_items: list[dict[str, str]]) -> None:
    columns = st.columns(4, gap="medium")
    for column, item in zip(columns, snapshot_items):
        with column:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{html.escape(item["label"])}</div>
                    <div class="kpi-value">{html.escape(item["value"])}</div>
                    <div class="kpi-copy">{html.escape(item["copy"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def truncate_text(value: str, limit: int = 220) -> str:
    cleaned = " ".join(str(value or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def render_theme_panel(title: str, description: str, topics: list[dict[str, Any]], tone: str) -> None:
    render_section_header("Topic Analysis", title, description)
    if not topics:
        st.info("Not enough reviews were available to surface reliable themes.")
        return

    for index, topic in enumerate(topics, start=1):
        chips = "".join(f'<span class="chip {tone}">{html.escape(keyword)}</span>' for keyword in topic.get("keywords", [])[:6])
        quotes = topic.get("representative_reviews", [])[:3]
        quote_markup = "".join(f"<li>{html.escape(truncate_text(quote, 260))}</li>" for quote in quotes)
        if not quote_markup:
            quote_markup = "<li>No representative review samples were available for this theme.</li>"

        st.markdown(
            f"""
            <div class="theme-card {tone}">
                <div class="theme-meta">
                    <span>Theme {index}</span>
                    <span>{topic.get("prevalence", 0.0) * 100:.1f}% prevalence</span>
                </div>
                <h3 class="theme-title">{html.escape(topic.get("label", f"Topic {index}"))}</h3>
                <div class="chip-row">{chips}</div>
                <ul class="theme-quotes">{quote_markup}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_summary_showcase(
    insights: dict[str, str],
    positive_topics: list[dict[str, Any]],
    negative_topics: list[dict[str, Any]],
    vocabulary_profile: dict[str, Any],
    llm_display_label: str,
) -> None:
    lead_positive = positive_topics[0].get("label", "No positive theme yet") if positive_topics else "No positive theme yet"
    lead_negative = negative_topics[0].get("label", "No negative theme yet") if negative_topics else "No negative theme yet"
    strategy = str(vocabulary_profile.get("strategy", "heuristic")).upper()
    pills = [
        f"Lead complaint: {lead_negative}",
        f"Lead strength: {lead_positive}",
        f"Vocabulary mode: {strategy}",
        f"Insight mode: {llm_display_label}",
    ]
    pill_markup = "".join(f'<span class="summary-pill">{html.escape(item)}</span>' for item in pills)

    summary_columns = st.columns([1.35, 0.95], gap="medium")
    with summary_columns[0]:
        st.markdown(
            f"""
            <div class="summary-lead-card">
                <div class="summary-overline">Executive Readout</div>
                <div class="summary-headline">{html.escape(insights.get("customer_complaints_summary", ""))}</div>
                <div class="summary-body">{html.escape(insights.get("business_implications", ""))}</div>
                <div class="summary-pill-row">{pill_markup}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with summary_columns[1]:
        st.markdown(
            f"""
            <div class="summary-engine-card">
                <div class="summary-overline">Signal Framing</div>
                <div class="summary-headline">What the model stack is saying</div>
                <div class="summary-body">{html.escape(insights.get("product_strengths", ""))}</div>
                <div class="summary-body">{html.escape(insights.get("product_improvement_suggestions", ""))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_insight_cards(insights: dict[str, str]) -> None:
    card_specs = [
        ("Customer Complaints", insights.get("customer_complaints_summary", "")),
        ("Product Strengths", insights.get("product_strengths", "")),
        ("Business Implications", insights.get("business_implications", "")),
        ("Improvement Suggestions", insights.get("product_improvement_suggestions", "")),
    ]

    first_row = st.columns(2, gap="medium")
    second_row = st.columns(2, gap="medium")

    for column, (title, body) in zip(first_row, card_specs[:2]):
        with column:
            st.markdown(
                f"""
                <div class="insight-card">
                    <h4>{html.escape(title)}</h4>
                    <p>{html.escape(body)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    for column, (title, body) in zip(second_row, card_specs[2:]):
        with column:
            st.markdown(
                f"""
                <div class="insight-card">
                    <h4>{html.escape(title)}</h4>
                    <p>{html.escape(body)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_vocabulary_profile(vocabulary_profile: dict[str, Any]) -> None:
    strategy = str(vocabulary_profile.get("strategy", "heuristic")).upper()
    suppressed = vocabulary_profile.get("additional_stopwords", [])
    preserved = vocabulary_profile.get("preserve_terms", [])

    with st.expander("Topic Vocabulary Policy", expanded=False):
        st.caption(f"Strategy: {strategy}")
        if suppressed:
            st.write("Suppressed topic-noise terms")
            st.code(", ".join(suppressed[:80]), language="text")
        if preserved:
            st.write("Preserved domain terms")
            st.code(", ".join(preserved[:80]), language="text")
        if not suppressed and not preserved:
            st.info("No extra vocabulary adjustments were applied for this run.")


def serialize_topics(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for topic in topics:
        serialized.append(
            {
                "label": topic.get("label"),
                "prevalence": topic.get("prevalence"),
                "keywords": topic.get("keywords", []),
                "review_count": topic.get("review_count"),
            }
        )
    return serialized


def build_ai_summary_pdf(
    analysis_label: str,
    insights: dict[str, str],
    positive_topics: list[dict[str, Any]],
    negative_topics: list[dict[str, Any]],
    llm_display_label: str,
) -> bytes:
    buffer = io.BytesIO()
    positive_labels = ", ".join(topic.get("label", "Unlabeled") for topic in positive_topics[:3]) or "No positive themes"
    negative_labels = ", ".join(topic.get("label", "Unlabeled") for topic in negative_topics[:3]) or "No negative themes"

    with PdfPages(buffer) as pdf:
        figure = plt.figure(figsize=(8.27, 11.69))
        figure.patch.set_facecolor("white")
        axis = figure.add_subplot(111)
        axis.axis("off")
        axis.text(0.05, 0.96, f"{APP_NAME} AI Summary", fontsize=22, fontweight="bold", color="#10253f")
        axis.text(0.05, 0.92, f"Analysis target: {analysis_label}", fontsize=11, color="#35516f")
        axis.text(0.05, 0.89, f"LLM: {llm_display_label}", fontsize=11, color="#35516f")

        y_position = 0.82
        sections = [
            ("Customer Complaints", insights.get("customer_complaints_summary", "")),
            ("Product Strengths", insights.get("product_strengths", "")),
            ("Business Implications", insights.get("business_implications", "")),
            ("Improvement Suggestions", insights.get("product_improvement_suggestions", "")),
            ("Lead Positive Themes", positive_labels),
            ("Lead Negative Themes", negative_labels),
        ]
        for title, body in sections:
            axis.text(0.05, y_position, title, fontsize=13, fontweight="bold", color="#b57900")
            axis.text(0.05, y_position - 0.035, fill(str(body), width=92), fontsize=10.5, color="#23384e", va="top")
            y_position -= 0.14

        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    return buffer.getvalue()


def build_visuals_pdf(
    analysis_label: str,
    sentiment_distribution: dict[str, float],
    scored_reviews: pd.DataFrame,
    positive_topics: list[dict[str, Any]],
    negative_topics: list[dict[str, Any]],
) -> bytes:
    buffer = io.BytesIO()
    positive_rows = [(topic.get("label", "Unlabeled"), float(topic.get("prevalence", 0.0)) * 100) for topic in positive_topics[:5]]
    negative_rows = [(topic.get("label", "Unlabeled"), float(topic.get("prevalence", 0.0)) * 100) for topic in negative_topics[:5]]
    trend_frame = (
        scored_reviews.assign(date=pd.to_datetime(scored_reviews["date"], errors="coerce"))
        .dropna(subset=["date"])
        .groupby(["date", "sentiment"])
        .size()
        .reset_index(name="review_count")
    )

    with PdfPages(buffer) as pdf:
        figure, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        figure.patch.set_facecolor("white")
        figure.suptitle(f"{APP_NAME} Visual Report: {analysis_label}", fontsize=18, fontweight="bold", color="#10253f")

        pie_axis = axes[0, 0]
        pie_axis.pie(
            [sentiment_distribution.get("positive", 0.0), sentiment_distribution.get("neutral", 0.0), sentiment_distribution.get("negative", 0.0)],
            labels=["Positive", "Neutral", "Negative"],
            colors=["#2ec4b6", "#f4d35e", "#ff6b6b"],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": "#23384e", "fontsize": 10},
        )
        pie_axis.set_title("Sentiment Distribution", color="#10253f", fontsize=13, fontweight="bold")

        trend_axis = axes[0, 1]
        if not trend_frame.empty:
            for sentiment, color in [("positive", "#2ec4b6"), ("neutral", "#d6b13f"), ("negative", "#ff6b6b")]:
                subset = trend_frame[trend_frame["sentiment"] == sentiment]
                if not subset.empty:
                    trend_axis.plot(subset["date"], subset["review_count"], label=sentiment.title(), color=color, linewidth=2.2)
            trend_axis.legend(frameon=False)
        trend_axis.set_title("Sentiment Trend", color="#10253f", fontsize=13, fontweight="bold")
        trend_axis.tick_params(colors="#35516f")
        trend_axis.grid(alpha=0.2)

        positive_axis = axes[1, 0]
        if positive_rows:
            labels, values = zip(*positive_rows)
            positive_axis.barh(list(labels), list(values), color="#2ec4b6")
        positive_axis.set_title("Top Positive Themes", color="#10253f", fontsize=13, fontweight="bold")
        positive_axis.tick_params(colors="#35516f")
        positive_axis.grid(axis="x", alpha=0.2)

        negative_axis = axes[1, 1]
        if negative_rows:
            labels, values = zip(*negative_rows)
            negative_axis.barh(list(labels), list(values), color="#ff6b6b")
        negative_axis.set_title("Top Complaint Drivers", color="#10253f", fontsize=13, fontweight="bold")
        negative_axis.tick_params(colors="#35516f")
        negative_axis.grid(axis="x", alpha=0.2)

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    return buffer.getvalue()


def render_downloads(result: dict[str, Any]) -> None:
    analysis_label = str(result["analysis_label"])
    export_name = safe_filename(analysis_label.lower())
    scored_reviews = result["scored_reviews"].copy()
    review_buffer = io.StringIO()
    scored_reviews.to_csv(review_buffer, index=False)

    summary = {
        "analysis_label": analysis_label,
        "source_name": result["source_name"],
        "sentiment_distribution": result["sentiment_distribution"],
        "positive_topics": serialize_topics(result["analysis_bundle"]["positive_topics"]),
        "negative_topics": serialize_topics(result["analysis_bundle"]["negative_topics"]),
        "insights": result["analysis_bundle"]["insights"],
        "vocabulary_profile": result["analysis_bundle"]["vocabulary_profile"],
    }

    summary_buffer = io.StringIO()
    json.dump(summary, summary_buffer, indent=2)

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            """
            <div class="download-card">
                <div class="download-title">Scored review dataset</div>
                <div class="download-copy">Download the cleaned and sentiment-scored review-level dataset for further slicing or BI work.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download scored reviews",
            review_buffer.getvalue().encode("utf-8"),
            file_name=f"{export_name}_scored_reviews.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with right:
        st.markdown(
            """
            <div class="download-card">
                <div class="download-title">Executive summary artifact</div>
                <div class="download-copy">Export the sentiment mix, dominant themes, vocabulary policy, and AI summaries as JSON.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download analysis summary",
            summary_buffer.getvalue().encode("utf-8"),
            file_name=f"{export_name}_analysis_summary.json",
            mime="application/json",
            use_container_width=True,
        )


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("Upload a CSV file to analyze.")

    content = uploaded_file.getvalue()
    if not content:
        raise ValueError("The uploaded file is empty.")

    parse_attempts = [
        {"sep": None, "engine": "python", "encoding": "utf-8"},
        {"sep": ",", "engine": "python", "encoding": "utf-8"},
        {"sep": ",", "engine": "python", "encoding": "latin-1"},
    ]
    for attempt in parse_attempts:
        try:
            return pd.read_csv(io.BytesIO(content), **attempt)
        except Exception:
            continue

    raise ValueError("The uploaded file could not be parsed as CSV.")


def mapping_option_label(value: str) -> str:
    return "Not provided" if value == UNMAPPED_COLUMN else value


def render_csv_mapping_ui(columns: list[str]) -> dict[str, str]:
    options = [UNMAPPED_COLUMN] + list(columns)
    suggested_mapping = suggest_column_mapping(columns)
    mapping: dict[str, str] = {}

    map_columns = st.columns(2, gap="medium")
    for index, field in enumerate(CSV_MAPPING_FIELDS):
        default_value = suggested_mapping.get(field, UNMAPPED_COLUMN)
        try:
            default_index = options.index(default_value)
        except ValueError:
            default_index = 0

        with map_columns[index % 2]:
            mapping[field] = st.selectbox(
                CSV_MAPPING_LABELS[field],
                options=options,
                index=default_index,
                format_func=mapping_option_label,
                key=f"mapping_{field}",
            )

    return mapping


def dominant_complaint_theme(negative_topics: list[dict[str, Any]]) -> str:
    if not negative_topics:
        return "No dominant complaint theme"
    return str(negative_topics[0].get("label", "No dominant complaint theme"))


def build_banner_text(negative_topics: list[dict[str, Any]], positive_topics: list[dict[str, Any]]) -> str:
    if negative_topics:
        lead_labels = [topic.get("label", "negative drivers") for topic in negative_topics[:2]]
        combined_share = sum(float(topic.get("prevalence", 0.0)) for topic in negative_topics[:2]) * 100
        return (
            f"{', '.join(lead_labels)} account for {combined_share:.0f}% of dominant negative themes, "
            "signaling concentrated retention and trust risk if these issues remain unresolved."
        )
    if positive_topics:
        lead_label = positive_topics[0].get("label", "positive engagement")
        return f"The strongest positive theme is {lead_label}, suggesting a clear value proposition to reinforce in product messaging."
    return "No dominant theme signal emerged from the current review set."


def build_sentiment_story(scored_reviews: pd.DataFrame, sentiment_distribution: dict[str, float]) -> str:
    positive_share = sentiment_distribution.get("positive", 0.0) * 100
    negative_share = sentiment_distribution.get("negative", 0.0) * 100
    neutral_share = sentiment_distribution.get("neutral", 0.0) * 100

    trend = (
        scored_reviews.assign(date=pd.to_datetime(scored_reviews["date"], errors="coerce"))
        .dropna(subset=["date"])
        .groupby(["date", "sentiment"])
        .size()
        .reset_index(name="review_count")
    )
    negative_trend = trend[trend["sentiment"] == "negative"]["review_count"]
    if not negative_trend.empty and negative_trend.max() >= max(5, negative_trend.mean() * 1.6):
        return (
            f"Sentiment is led by positive reviews at {positive_share:.1f}%, but negative feedback still represents "
            f"{negative_share:.1f}% of the conversation and shows identifiable spikes around specific review periods."
        )

    return (
        f"Sentiment has remained relatively stable with {positive_share:.1f}% positive, {negative_share:.1f}% negative, "
        f"and {neutral_share:.1f}% neutral reviews. The current signal suggests persistent but not runaway dissatisfaction."
    )


def limit_reviews(frame: pd.DataFrame, max_reviews: int) -> pd.DataFrame:
    if len(frame) <= max_reviews:
        return frame

    working = frame.copy()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        working = working.sort_values("date", ascending=False, na_position="last")
    return working.head(max_reviews).reset_index(drop=True)


def render_review_explorer(scored_reviews: pd.DataFrame) -> None:
    st.markdown('<div class="explorer-shell">', unsafe_allow_html=True)
    filter_columns = st.columns([1.1, 1.2, 1.35, 0.85], gap="medium")
    available_sentiments = sorted(scored_reviews["sentiment"].dropna().unique().tolist())
    default_dates = pd.to_datetime(scored_reviews["date"], errors="coerce").dropna()

    with filter_columns[0]:
        selected_sentiments = st.multiselect(
            "Sentiment filter",
            options=available_sentiments,
            default=available_sentiments,
        )

    with filter_columns[1]:
        keyword_filter = st.text_input("Keyword search", placeholder="login, ads, pricing...")

    with filter_columns[2]:
        if not default_dates.empty:
            date_values = st.date_input(
                "Date range",
                value=(default_dates.min().date(), default_dates.max().date()),
            )
        else:
            date_values = ()

    with filter_columns[3]:
        row_limit = st.number_input("Rows displayed", min_value=10, max_value=500, value=100, step=10)

    filtered = scored_reviews.copy()
    if selected_sentiments:
        filtered = filtered[filtered["sentiment"].isin(selected_sentiments)]

    if keyword_filter.strip():
        filtered = filtered[filtered["review_text"].str.contains(keyword_filter.strip(), case=False, na=False)]

    if date_values and isinstance(date_values, (tuple, list)) and len(date_values) == 2:
        start_date, end_date = date_values
        date_series = pd.to_datetime(filtered["date"], errors="coerce").dt.date
        filtered = filtered[(date_series >= start_date) & (date_series <= end_date)]

    filtered = filtered.sort_values("date", ascending=False, na_position="last").head(int(row_limit))
    display_frame = filtered[["date", "rating", "sentiment", "sentiment_confidence", "review_text", "source"]].copy()
    display_frame = display_frame.rename(
        columns={
            "date": "Date",
            "rating": "Rating",
            "sentiment": "Sentiment",
            "sentiment_confidence": "Confidence",
            "review_text": "Review",
            "source": "Source",
        }
    )
    st.dataframe(display_frame, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_input_shell(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="input-shell">
            <div class="input-title">{html.escape(title)}</div>
            <div class="input-copy">{html.escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_analysis_pipeline(
    source_mode: str,
    play_store_url: str,
    uploaded_frame: pd.DataFrame | None,
    csv_mapping: dict[str, str],
    csv_source_name: str,
    csv_product_name: str,
    csv_product_id: str,
    max_reviews: int,
    review_language: str,
    country: str,
    sentiment_batch_size: int,
    llm_access_mode: str,
    own_provider: str,
    own_model: str,
    own_api_key: str,
    llm_timeout: float,
    sentiment_model_key: str,
) -> dict[str, Any]:
    sentiment_model_profile = get_sentiment_model_profile(sentiment_model_key)
    llm_analyzer, llm_config = build_llm_analyzer(
        access_mode=llm_access_mode,
        own_provider=own_provider,
        own_model=own_model,
        own_api_key=own_api_key,
        timeout_seconds=llm_timeout,
    )

    if llm_access_mode == OUR_DEFAULT_LLM and not bool(llm_config.get("configured")):
        raise ValueError("No backend-managed LLM key is configured in this environment. Use Your Own LLM for local testing.")
    if llm_access_mode == YOUR_OWN_LLM and not own_api_key.strip():
        raise ValueError("Enter your own API key to use the selected LLM provider.")

    with st.status("Running review intelligence pipeline...", expanded=True) as status:
        try:
            status.write("Resolving review source...")
            if source_mode == SOURCE_PLAY_STORE:
                app_id = extract_app_id(play_store_url)
                status.write("Fetching Google Play reviews...")
                raw_frame = fetch_play_store_reviews_cached(
                    app_id=app_id,
                    max_reviews=max_reviews,
                    lang=review_language,
                    country=country,
                    sort_by="newest",
                )
                ingestion_result = normalize_play_store_frame(raw_frame, app_id)
                source_name = SOURCE_PLAY_STORE
            else:
                if uploaded_frame is None:
                    raise ValueError("Upload a CSV file before starting analysis.")
                limited_frame = limit_reviews(uploaded_frame, max_reviews)
                ingestion_result = normalize_uploaded_frame(
                    frame=limited_frame,
                    mapping=csv_mapping,
                    source_name=csv_source_name or "csv_upload",
                    product_name=csv_product_name,
                    product_id=csv_product_id,
                )
                source_name = csv_source_name or SOURCE_CSV_UPLOAD

            status.write("Generating topic vocabulary policy...")
            vocabulary_profile = llm_analyzer.generate_topic_vocabulary_profile(
                topic_context=ingestion_result.topic_context or ingestion_result.analysis_label,
                source_name=source_name,
                reviews=ingestion_result.frame["review_text"].tolist(),
            )

            status.write("Preprocessing review text...")
            processed_reviews = preprocess_reviews_cached(
                ingestion_result.frame,
                tuple(vocabulary_profile.get("preserve_terms", [])),
            )

            status.write(f"Running {sentiment_model_profile['summary_label']} sentiment analysis...")
            scored_reviews, sentiment_distribution = score_reviews_cached(
                processed_reviews,
                profile_key=sentiment_model_key,
                batch_size=sentiment_batch_size,
            )

            status.write("Modeling topics and generating insights...")
            analysis_bundle = build_analysis_bundle(
                scored_reviews=scored_reviews,
                sentiment_distribution=sentiment_distribution,
                topic_context=ingestion_result.topic_context or ingestion_result.analysis_label,
                vocabulary_profile=vocabulary_profile,
                source_name=source_name,
                llm_analyzer=llm_analyzer,
            )

            status.update(label="Analysis complete", state="complete")

            return {
                "analysis_label": ingestion_result.analysis_label,
                "topic_context": ingestion_result.topic_context,
                "source_name": source_name,
                "raw_reviews": ingestion_result.frame,
                "scored_reviews": scored_reviews,
                "sentiment_distribution": sentiment_distribution,
                "analysis_bundle": analysis_bundle,
                "llm_access_mode": llm_access_mode,
                "llm_provider": str(llm_config.get("provider") or "none"),
                "llm_model": str(llm_config.get("model") or ""),
                "sentiment_model_key": sentiment_model_key,
                "sentiment_model_name": str(sentiment_model_profile["model_name"]),
                "sentiment_model_label": str(sentiment_model_profile["summary_label"]),
            }
        except Exception as exc:
            status.update(label="Analysis failed", state="error")
            raise


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="Pulse", layout="wide", initial_sidebar_state="expanded")

    if "ui_theme_mode" not in st.session_state:
        st.session_state.ui_theme_mode = UI_THEME_DARK
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    render_styles(st.session_state.ui_theme_mode)
    top_bar_left, top_bar_right = st.columns([5.2, 1.5], gap="medium")
    with top_bar_right:
        st.radio(
            "App theme",
            options=[UI_THEME_DARK, UI_THEME_LIGHT],
            horizontal=True,
            key="ui_theme_mode",
        )
    render_hero()

    with st.sidebar:
        st.markdown("## Control Room")
        st.caption("Configure runtime, LLM access, and notifications for the current analysis session.")
        st.caption(f"Theme: {st.session_state.ui_theme_mode}")

        sentiment_batch_size = st.slider("Sentiment batch size", min_value=8, max_value=64, value=16, step=8)
        sentiment_model_key = st.selectbox(
            "Sentiment model",
            options=list(SENTIMENT_MODEL_PROFILES.keys()),
            index=list(SENTIMENT_MODEL_PROFILES.keys()).index(DEFAULT_SENTIMENT_MODEL_KEY),
        )
        active_sentiment_profile = get_sentiment_model_profile(sentiment_model_key)
        st.caption(str(active_sentiment_profile["description"]))
        max_reviews = st.slider("Maximum reviews", min_value=100, max_value=MAX_ALLOWED_REVIEWS, value=2000, step=100)
        review_language = st.text_input("Review language", value="en")
        country = st.text_input("Country", value="us")

        render_divider()
        st.markdown("### LLM access")
        llm_access_mode = st.selectbox(
            "LLM access mode",
            options=[OUR_DEFAULT_LLM, YOUR_OWN_LLM, DISABLE_LLM],
            index=0,
            label_visibility="collapsed",
        )
        own_provider = "groq"
        own_model = OWN_LLM_MODELS["groq"][0]
        own_api_key = ""
        platform_config = resolve_platform_llm_config()

        if llm_access_mode == OUR_DEFAULT_LLM:
            if platform_config["configured"]:
                st.caption(
                    f"Using the platform-managed {str(platform_config['provider']).title()} model "
                    f"`{platform_config['model']}` for this session."
                )
            else:
                st.warning("No backend-managed LLM key is available in this local environment. Use Your Own LLM for testing.")

        if llm_access_mode == YOUR_OWN_LLM:
            own_provider = st.selectbox("Custom provider", options=list(OWN_LLM_MODELS.keys()), index=0)
            own_model = st.selectbox("Model", options=OWN_LLM_MODELS[own_provider], index=0)
            own_api_key = st.text_input("Your API key", type="password")
            st.caption(f"Using a user-supplied {own_provider.title()} key for this session.")

        if llm_access_mode == DISABLE_LLM:
            st.caption("LLM disabled. Topic labels, vocabulary policy, and business insights will use deterministic fallbacks.")

        llm_timeout = st.number_input(
            "LLM timeout (seconds)",
            min_value=5,
            max_value=120,
            value=30,
            step=5,
            disabled=llm_access_mode == DISABLE_LLM,
        )

    render_divider()
    render_section_header(
        "Start Analysis",
        "Bring in a review source",
        "Scrape Google Play reviews or upload your own review export, then run the full review intelligence pipeline.",
    )

    source_mode = st.radio(
        "Select source",
        options=[SOURCE_PLAY_STORE, SOURCE_CSV_UPLOAD],
        horizontal=True,
        label_visibility="collapsed",
    )

    play_store_url = SAMPLE_URL
    uploaded_file = None
    uploaded_frame: pd.DataFrame | None = None
    csv_mapping = {field: UNMAPPED_COLUMN for field in CSV_MAPPING_FIELDS}
    csv_source_name = "partner_csv"
    csv_product_name = ""
    csv_product_id = ""

    if source_mode == SOURCE_PLAY_STORE:
        render_input_shell(
            "Google Play review source",
            "Paste a Google Play Store app URL. ReviewPulse will extract the app id automatically and pull fresh reviews.",
        )
        input_columns = st.columns([5, 1.6], gap="medium")
        with input_columns[0]:
            play_store_url = st.text_input("Google Play URL", value=SAMPLE_URL)
        with input_columns[1]:
            st.markdown('<div class="review-meta">App id will be extracted automatically during analysis.</div>', unsafe_allow_html=True)
    else:
        render_input_shell(
            "Universal CSV review source",
            "Upload a review export from any marketplace, product, or partner system and map the columns to the canonical schema.",
        )
        upload_columns = st.columns([1.5, 1.5, 1.2], gap="medium")
        with upload_columns[0]:
            uploaded_file = st.file_uploader("Review CSV", type=["csv"])
        with upload_columns[1]:
            csv_source_name = st.text_input("Source label", value="partner_csv")
            csv_product_name = st.text_input("Product name", value="")
        with upload_columns[2]:
            csv_product_id = st.text_input("Product id", value="")

        if uploaded_file is not None:
            try:
                uploaded_frame = read_uploaded_csv(uploaded_file)
                st.caption(f"Detected {len(uploaded_frame):,} rows and {len(uploaded_frame.columns)} columns in the uploaded file.")
                st.dataframe(uploaded_frame.head(8), use_container_width=True, hide_index=True)
                render_divider()
                st.markdown("#### Column mapping")
                csv_mapping = render_csv_mapping_ui(uploaded_frame.columns.tolist())
            except Exception as exc:
                st.error(str(exc))
                uploaded_frame = None

    action_columns = st.columns([1.2, 1.2, 4], gap="medium")
    with action_columns[0]:
        analyze_clicked = st.button("Analyze Reviews", type="primary", use_container_width=True)
    with action_columns[1]:
        reset_clicked = st.button("Clear Results", use_container_width=True)
    with action_columns[2]:
        st.markdown(
            '<div class="microcopy">Local testing should typically use <strong>Your Own LLM</strong>. '
            'In deployment, <strong>Our Default LLM</strong> will use backend-managed secrets without exposing an API key field to end users.</div>',
            unsafe_allow_html=True,
        )

    if reset_clicked:
        st.session_state.analysis_result = None
        st.cache_data.clear()
        st.rerun()

    if analyze_clicked:
        try:
            result = run_analysis_pipeline(
                source_mode=source_mode,
                play_store_url=play_store_url,
                uploaded_frame=uploaded_frame,
                csv_mapping=csv_mapping,
                csv_source_name=csv_source_name,
                csv_product_name=csv_product_name,
                csv_product_id=csv_product_id,
                max_reviews=max_reviews,
                review_language=review_language,
                country=country,
                sentiment_batch_size=sentiment_batch_size,
                llm_access_mode=llm_access_mode,
                own_provider=own_provider,
                own_model=own_model,
                own_api_key=own_api_key,
                llm_timeout=float(llm_timeout),
                sentiment_model_key=sentiment_model_key,
            )
            st.session_state.analysis_result = result
        except Exception as exc:
            st.error(str(exc))

    result = st.session_state.analysis_result
    if not result:
        st.info("Run an analysis to populate the dashboard.")
        return

    analysis_bundle = result["analysis_bundle"]
    scored_reviews = result["scored_reviews"]
    sentiment_distribution = result["sentiment_distribution"]
    positive_topics = analysis_bundle["positive_topics"]
    negative_topics = analysis_bundle["negative_topics"]
    insights = analysis_bundle["insights"]
    vocabulary_profile = analysis_bundle["vocabulary_profile"]
    active_theme_mode = st.session_state.ui_theme_mode
    llm_display_label = format_llm_display_label(
        result["llm_access_mode"],
        result.get("llm_provider", ""),
        result.get("llm_model", ""),
    )
    ai_summary_pdf = build_ai_summary_pdf(
        analysis_label=result["analysis_label"],
        insights=insights,
        positive_topics=positive_topics,
        negative_topics=negative_topics,
        llm_display_label=llm_display_label,
    )
    visuals_pdf = build_visuals_pdf(
        analysis_label=result["analysis_label"],
        sentiment_distribution=sentiment_distribution,
        scored_reviews=scored_reviews,
        positive_topics=positive_topics,
        negative_topics=negative_topics,
    )

    render_divider()
    render_banner(build_banner_text(negative_topics, positive_topics))
    st.markdown(
        f"""
        <div class="review-meta">
            Source: {html.escape(result["source_name"])} |
            Analysis target: {html.escape(result["analysis_label"])} |
            LLM: {html.escape(llm_display_label)} |
            Sentiment model: {html.escape(str(result.get("sentiment_model_label") or result.get("sentiment_model_name") or DEFAULT_SENTIMENT_MODEL))}
        </div>
        """,
        unsafe_allow_html=True,
    )
    pulse_tab, themes_tab, summary_tab, visuals_tab, explorer_tab, downloads_tab = st.tabs(
        ["Pulse", "Themes", "AI Summary", "Visuals", "Review Explorer", "Downloads"]
    )

    with pulse_tab:
        render_section_header(
            "Section 1",
            "Product Snapshot",
            "A fast read on review volume, sentiment balance, and the single complaint theme most likely to affect product perception.",
        )
        render_snapshot_cards(
            [
                {
                    "label": "Total Reviews",
                    "value": f"{len(scored_reviews):,}",
                    "copy": "Reviews retained after ingestion and preprocessing.",
                },
                {
                    "label": "Positive Sentiment",
                    "value": format_percentage(sentiment_distribution.get("positive", 0.0)),
                    "copy": "Share of reviews classified as positive by RoBERTa.",
                },
                {
                    "label": "Negative Sentiment",
                    "value": format_percentage(sentiment_distribution.get("negative", 0.0)),
                    "copy": "Share of reviews classified as negative by RoBERTa.",
                },
                {
                    "label": "Dominant Complaint Theme",
                    "value": dominant_complaint_theme(negative_topics),
                    "copy": "Most prevalent negative theme surfaced by topic modeling.",
                },
            ]
        )

        render_divider()
        render_section_header(
            "Section 2",
            "Customer Sentiment Overview",
            "Read the overall mix on the left, then inspect how review sentiment evolves across time on the right.",
        )
        sentiment_columns = st.columns([0.9, 1.4], gap="medium")
        with sentiment_columns[0]:
            pie_figure = sentiment_pie_chart(sentiment_distribution, theme_mode=active_theme_mode)
            st.plotly_chart(pie_figure, use_container_width=True, config=PLOTLY_CONFIG, theme=None)
        with sentiment_columns[1]:
            trend_figure = sentiment_trend_chart(scored_reviews, theme_mode=active_theme_mode)
            if trend_figure is not None:
                st.plotly_chart(trend_figure, use_container_width=True, config=PLOTLY_CONFIG, theme=None)
            else:
                st.info("Not enough dated reviews were available to render the sentiment trend.")
        render_story(build_sentiment_story(scored_reviews, sentiment_distribution))

    with themes_tab:
        positive_column, negative_column = st.columns(2, gap="large")
        with positive_column:
            render_theme_panel(
                "Top Positive Themes",
                "These themes represent what users consistently value and what is worth protecting in the product experience.",
                positive_topics,
                "positive",
            )
        with negative_column:
            render_theme_panel(
                "Top Negative Themes",
                "These themes capture friction, reliability concerns, and signals of churn or support escalation pressure.",
                negative_topics,
                "negative",
            )

    with summary_tab:
        summary_download_columns = st.columns([4.6, 1.4], gap="medium")
        with summary_download_columns[1]:
            st.download_button(
                "Download AI Summary PDF",
                ai_summary_pdf,
                file_name=f"{safe_filename(result['analysis_label'].lower())}_ai_summary.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        render_section_header(
            "Section 4",
            "AI Product Insights",
            "Executive-ready synthesis grounded in topic prevalence and sentiment movement rather than anecdotal examples.",
        )
        render_summary_showcase(
            insights=insights,
            positive_topics=positive_topics,
            negative_topics=negative_topics,
            vocabulary_profile=vocabulary_profile,
            llm_display_label=llm_display_label,
        )
        render_divider()
        render_insight_cards(insights)
        render_vocabulary_profile(vocabulary_profile)

    with visuals_tab:
        visuals_download_columns = st.columns([4.6, 1.4], gap="medium")
        with visuals_download_columns[1]:
            st.download_button(
                "Download Visuals PDF",
                visuals_pdf,
                file_name=f"{safe_filename(result['analysis_label'].lower())}_visuals.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        render_section_header(
            "Section 5",
            "Visual Analysis",
            "These views emphasize theme importance and complaint drivers so the story stays focused on action rather than chart volume.",
        )
        visual_top = st.columns(2, gap="medium")
        with visual_top[0]:
            positive_chart = theme_importance_chart(
                positive_topics,
                "Top Positive Themes by Prevalence",
                "#2ec4b6",
                theme_mode=active_theme_mode,
            )
            if positive_chart is not None:
                st.plotly_chart(positive_chart, use_container_width=True, config=PLOTLY_CONFIG, theme=None)
            else:
                st.info("Positive theme prevalence will appear here once reliable topics are available.")
        with visual_top[1]:
            negative_chart = theme_importance_chart(
                negative_topics,
                "Top Complaint Drivers",
                "#ff6b6b",
                theme_mode=active_theme_mode,
            )
            if negative_chart is not None:
                st.plotly_chart(negative_chart, use_container_width=True, config=PLOTLY_CONFIG, theme=None)
            else:
                st.info("Negative driver ranking will appear here once reliable topics are available.")

        visual_bottom = st.columns(2, gap="medium")
        with visual_bottom[0]:
            positive_clusters = keyword_cluster_chart(
                positive_topics,
                "Positive Keyword Clusters",
                color_sequence=["#0f766e", "#16a34a", "#0ea5e9", "#22c55e", "#14b8a6"],
                theme_mode=active_theme_mode,
            )
            if positive_clusters is not None:
                st.plotly_chart(positive_clusters, use_container_width=True, config=PLOTLY_CONFIG, theme=None)
            else:
                st.info("Positive keyword clusters were not available for this run.")
        with visual_bottom[1]:
            negative_clusters = keyword_cluster_chart(
                negative_topics,
                "Negative Keyword Clusters",
                color_sequence=["#be123c", "#ef4444", "#f97316", "#fb7185", "#dc2626"],
                theme_mode=active_theme_mode,
            )
            if negative_clusters is not None:
                st.plotly_chart(negative_clusters, use_container_width=True, config=PLOTLY_CONFIG, theme=None)
            else:
                st.info("Negative keyword clusters were not available for this run.")

        with st.expander("Language clouds", expanded=False):
            cloud_columns = st.columns(2, gap="medium")
            shared_stopwords = generic_review_noise_stopwords() | set(vocabulary_profile.get("additional_stopwords", []))
            with cloud_columns[0]:
                positive_cloud = generate_wordcloud_figure(
                    analysis_bundle["sentiment_frames"]["positive"]["clean_text"].tolist(),
                    "Positive Language Cloud",
                    stopwords=shared_stopwords,
                    theme_mode=active_theme_mode,
                )
                if positive_cloud is not None:
                    st.pyplot(positive_cloud, use_container_width=True)
                else:
                    st.info("Positive word cloud unavailable.")
            with cloud_columns[1]:
                negative_cloud = generate_wordcloud_figure(
                    analysis_bundle["sentiment_frames"]["negative"]["clean_text"].tolist(),
                    "Negative Language Cloud",
                    stopwords=shared_stopwords,
                    theme_mode=active_theme_mode,
                )
                if negative_cloud is not None:
                    st.pyplot(negative_cloud, use_container_width=True)
                else:
                    st.info("Negative word cloud unavailable.")

    with explorer_tab:
        render_section_header(
            "Section 6",
            "Review Explorer",
            "Filter by sentiment, search terms, and dates to move from dashboard summaries into concrete review evidence.",
        )
        render_review_explorer(scored_reviews)

    with downloads_tab:
        render_section_header(
            "Exports",
            "Download analysis artifacts",
            "Take the scored dataset and the structured AI summary into downstream reporting, issue triage, or stakeholder communication.",
        )
        render_downloads(result)


if __name__ == "__main__":
    main()
