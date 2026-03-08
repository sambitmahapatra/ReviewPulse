from project.pipeline import preprocess
from project.pipeline.preprocess import ConservativeLemmatizer, preprocess_review_text


def test_preprocess_removes_html_urls_and_keeps_negation():
    text = "This app is <b>not</b> good. Visit https://example.com now."
    cleaned = preprocess_review_text(text)
    assert "not" in cleaned
    assert "https" not in cleaned
    assert "<b>" not in cleaned


def test_preprocess_can_preserve_context_terms():
    text = "This is not on time and on point."
    cleaned = preprocess_review_text(text, preserve_terms={"on"})
    assert "on" in cleaned.split()


def test_conservative_lemmatizer_singularizes_simple_plurals():
    lemmatizer = ConservativeLemmatizer()
    assert lemmatizer.lemmatize("reviews") == "review"
    assert lemmatizer.lemmatize("stories") == "story"


def test_stopword_fallback_works_without_nltk_corpora(monkeypatch):
    preprocess.get_stop_words.cache_clear()
    preprocess.get_nltk_resource_status.cache_clear()
    monkeypatch.setattr(
        preprocess,
        "get_nltk_resource_status",
        lambda: preprocess.NltkResourceStatus(stopwords=False, wordnet=False),
    )

    stop_words = preprocess.get_stop_words()
    assert "the" in stop_words
    assert "not" not in stop_words
