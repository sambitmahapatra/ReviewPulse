from __future__ import annotations

from collections import Counter
from typing import Iterable

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def generate_wordcloud_figure(
    texts: list[str],
    title: str,
    stopwords: Iterable[str] | None = None,
    theme_mode: str = "dark",
):
    excluded_terms = {str(term).lower() for term in (stopwords or []) if str(term).strip()}
    tokens = [
        token
        for token in " ".join(texts).split()
        if token.strip() and token.lower() not in excluded_terms
    ]
    if not tokens:
        return None

    theme = str(theme_mode).lower()
    background = "#08121f" if theme == "dark" else "#f8fbff"
    title_color = "#f5f7ff" if theme == "dark" else "#13263c"

    frequencies = Counter(tokens)
    cloud = WordCloud(
        width=900,
        height=450,
        background_color=background,
        colormap="viridis",
    ).generate_from_frequencies(frequencies)

    figure, axis = plt.subplots(figsize=(10, 5))
    figure.patch.set_facecolor(background)
    axis.set_facecolor(background)
    axis.imshow(cloud, interpolation="bilinear")
    axis.set_title(title, color=title_color)
    axis.axis("off")
    return figure
