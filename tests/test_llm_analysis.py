from project.pipeline.llm_analysis import LLMAnalyzer


def test_vocabulary_profile_falls_back_to_heuristics_without_llm():
    analyzer = LLMAnalyzer(provider="none")
    profile = analyzer.generate_topic_vocabulary_profile(
        topic_context="Amazon Prime Video",
        source_name="amazon_csv",
        reviews=["Prime membership is expensive but video quality is good."],
    )

    assert profile["strategy"] == "heuristic"
    assert "amazon" in profile["additional_stopwords"]
    assert "prime" in profile["additional_stopwords"]
    assert "good" in profile["additional_stopwords"]
    assert profile["preserve_terms"] == []


def test_label_topics_accepts_json_llm_response(monkeypatch):
    analyzer = LLMAnalyzer(provider="groq", api_key="demo-key")
    monkeypatch.setattr(
        analyzer,
        "_chat_completion",
        lambda system_prompt, user_prompt: '{"label":"Playback Stability Issues"}',
    )

    topics = analyzer.label_topics(
        [
            {
                "topic_id": 0,
                "keywords": ["buffering", "lag", "playback"],
                "representative_reviews": ["Video keeps buffering during live matches."],
            }
        ],
        topic_context="hotstar",
        source_name="google_play",
    )

    assert topics[0]["label"] == "Playback Stability Issues"


def test_parse_json_response_accepts_fenced_or_wrapped_json():
    analyzer = LLMAnalyzer(provider="none")

    fenced = "```json\n{\"label\":\"Subscription Value Concerns\"}\n```"
    wrapped = 'Here is the result:\n{"label":"Login and Access Issues"}\nThanks.'

    assert analyzer._parse_json_response(fenced)["label"] == "Subscription Value Concerns"
    assert analyzer._parse_json_response(wrapped)["label"] == "Login and Access Issues"
