from project.visualization.wordcloud import generate_wordcloud_figure


def test_wordcloud_excludes_configured_stopwords():
    figure = generate_wordcloud_figure(
        ["good good streaming smooth cricket", "good match smooth"],
        "Demo",
        stopwords={"good"},
    )

    assert figure is not None
