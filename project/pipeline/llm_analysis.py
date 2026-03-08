from __future__ import annotations

import json
import re
from typing import Any

from project.utils.helpers import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_OPENAI_MODEL,
    collapse_whitespace,
    generic_review_noise_stopwords,
    make_topic_fallback_label,
    text_identifier_stopwords,
)


TERM_PATTERN = re.compile(r"[a-z][a-z']+")
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class LLMAnalyzer:
    def __init__(
        self,
        provider: str = "none",
        api_key: str = "",
        model_name: str = "",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.provider = (provider or "none").lower()
        self.api_key = (api_key or "").strip()
        self.model_name = model_name.strip() or self._default_model_name()
        self.timeout_seconds = float(timeout_seconds)

    def _default_model_name(self) -> str:
        if self.provider == "groq":
            return DEFAULT_GROQ_MODEL
        if self.provider == "openai":
            return DEFAULT_OPENAI_MODEL
        return ""

    @property
    def is_configured(self) -> bool:
        return self.provider in {"groq", "openai"} and bool(self.api_key)

    @staticmethod
    def _parse_json_response(response_text: str) -> dict[str, Any]:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = JSON_OBJECT_PATTERN.search(cleaned)
            if not match:
                raise
            return json.loads(match.group(0))

    def _chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "groq":
            from groq import Groq

            client = Groq(api_key=self.api_key, timeout=self.timeout_seconds)
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=self.timeout_seconds,
            )
            return response.choices[0].message.content.strip()

        if self.provider == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=self.timeout_seconds,
            )
            return response.choices[0].message.content.strip()

        raise RuntimeError("LLM provider is not configured.")

    @staticmethod
    def _clean_term_list(values: Any) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()

        if not isinstance(values, list):
            return cleaned

        for value in values:
            for token in TERM_PATTERN.findall(str(value).lower()):
                if len(token) <= 1 or token in seen:
                    continue
                cleaned.append(token)
                seen.add(token)

        return cleaned

    @staticmethod
    def _clean_label(value: Any) -> str:
        label = collapse_whitespace(str(value or ""))
        label = re.sub(r"[/|]+", " ", label)
        label = re.sub(r"^[\"'`]+|[\"'`]+$", "", label)
        label = re.sub(r"\s+", " ", label).strip()
        return label[:80]

    @staticmethod
    def _looks_like_keyword_dump(label: str, keywords: list[str]) -> bool:
        normalized_label = re.sub(r"[^a-zA-Z0-9 ]+", " ", label.lower())
        label_tokens = [token for token in normalized_label.split() if token]
        if not label_tokens:
            return True

        keyword_tokens: set[str] = set()
        for keyword in keywords:
            keyword_tokens.update(TERM_PATTERN.findall(str(keyword).lower()))

        weak_theme_terms = {
            "good",
            "great",
            "nice",
            "best",
            "wonderful",
            "amazing",
            "service",
            "experience",
            "app",
            "ott",
            "time",
            "video",
            "watch",
        }
        if all(token in keyword_tokens for token in label_tokens):
            if len(label_tokens) <= 4:
                return True
            if sum(token in weak_theme_terms for token in label_tokens) >= 2:
                return True

        return False

    def _build_vocabulary_policy_prompt(
        self,
        topic_context: str,
        source_name: str,
        sampled_reviews: list[str],
    ) -> tuple[str, str]:
        payload = {
            "topic_context": topic_context,
            "source_name": source_name,
            "sample_reviews": sampled_reviews,
        }
        system_prompt = (
            "You are designing a vocabulary policy for review topic modeling. "
            "Work semantically, not by frequency alone. Separate generic review noise from domain-bearing terms. "
            "Suppress source names, brand mentions that do not define a theme, generic praise/complaint adjectives, "
            "polite filler, and non-topical verbs. Preserve meaningful product nouns, issue terms, payment/account/device "
            "terms, performance terms, and domain-specific feature language. Return strict JSON only."
        )
        user_prompt = (
            "Return JSON with exactly these keys:\n"
            "- `additional_stopwords`: lowercase single-token terms to suppress in topic modeling\n"
            "- `preserve_terms`: lowercase single-token terms that must be kept because they carry domain meaning\n\n"
            "Rules:\n"
            "- prefer semantic usefulness over raw frequency\n"
            "- do not include multi-word phrases\n"
            "- do not include duplicates\n"
            "- keep the lists concise\n"
            "- if a token is in `preserve_terms`, do not also include it in `additional_stopwords`\n\n"
            f"{json.dumps(payload, ensure_ascii=True)}"
        )
        return system_prompt, user_prompt

    def _build_topic_label_prompt(
        self,
        topic: dict[str, Any],
        topic_context: str,
        source_name: str,
        rejected_label: str = "",
    ) -> tuple[str, str]:
        payload = {
            "topic_context": topic_context,
            "source_name": source_name,
            "keywords": topic.get("keywords", []),
            "representative_reviews": topic.get("representative_reviews", [])[:3],
        }
        system_prompt = (
            "You convert review-topic evidence into concise business labels. "
            "Infer the underlying theme from semantics, not just the most obvious words. "
            "Prefer a 2 to 5 word noun phrase. Avoid slashes, avoid generic sentiment words, "
            "and avoid source or brand names unless they are the core issue. "
            "Do not return a raw keyword list. Return strict JSON only."
        )
        retry_instruction = ""
        if rejected_label:
            retry_instruction = (
                f'The previous label "{rejected_label}" was rejected because it looked like a raw keyword list. '
                "Rewrite it as a business theme name.\n\n"
            )
        user_prompt = (
            "Return JSON with exactly one key: `label`.\n"
            "The label should capture the business theme behind the topic.\n"
            "If keywords are noisy, rely more on the representative reviews.\n\n"
            f"{retry_instruction}"
            "Bad labels:\n"
            '- "Ads / Watch / Time"\n'
            '- "Wonderful Service Ott"\n'
            '- "Login No Account"\n\n'
            "Good labels:\n"
            '- "Ad Frequency Issues"\n'
            '- "Streaming Performance Problems"\n'
            '- "Login and Access Issues"\n'
            '- "Subscription Value Concerns"\n'
            '- "Content Quality Appreciation"\n\n'
            f"{json.dumps(payload, ensure_ascii=True)}"
        )
        return system_prompt, user_prompt

    def _build_business_insight_prompt(
        self,
        negative_topics: list[dict[str, Any]],
        positive_topics: list[dict[str, Any]],
        sentiment_distribution: dict[str, float],
    ) -> tuple[str, str]:
        prompt_payload = {
            "sentiment_distribution": sentiment_distribution,
            "top_negative_topics": [
                {
                    "label": topic.get("label"),
                    "keywords": topic.get("keywords", []),
                    "prevalence": topic.get("prevalence"),
                }
                for topic in negative_topics
            ],
            "top_positive_topics": [
                {
                    "label": topic.get("label"),
                    "keywords": topic.get("keywords", []),
                    "prevalence": topic.get("prevalence"),
                }
                for topic in positive_topics
            ],
        }
        system_prompt = (
            "You are a senior product analyst. Generate grounded review-analysis insights only from the supplied evidence. "
            "Do not invent features, causes, or metrics that are not present. "
            "If evidence is limited, speak cautiously. Return strict JSON only."
        )
        user_prompt = (
            "Return JSON with exactly these keys:\n"
            "- `customer_complaints_summary`\n"
            "- `product_strengths`\n"
            "- `business_implications`\n"
            "- `product_improvement_suggestions`\n\n"
            "Each field should be a short paragraph grounded in the provided topics and sentiment mix.\n\n"
            f"{json.dumps(prompt_payload, ensure_ascii=True)}"
        )
        return system_prompt, user_prompt

    def generate_topic_vocabulary_profile(
        self,
        topic_context: str,
        source_name: str,
        reviews: list[str],
        sample_size: int = 25,
    ) -> dict[str, Any]:
        heuristic_stopwords = text_identifier_stopwords(topic_context, source_name) | generic_review_noise_stopwords()
        fallback = {
            "additional_stopwords": sorted(heuristic_stopwords),
            "preserve_terms": [],
            "strategy": "heuristic",
        }
        if not self.is_configured:
            return fallback

        sampled_reviews = [
            collapse_whitespace(str(review))[:280]
            for review in reviews
            if collapse_whitespace(str(review))
        ][:sample_size]
        if not sampled_reviews:
            return fallback

        system_prompt, user_prompt = self._build_vocabulary_policy_prompt(
            topic_context=topic_context,
            source_name=source_name,
            sampled_reviews=sampled_reviews,
        )

        try:
            response_text = self._chat_completion(system_prompt, user_prompt)
            parsed = self._parse_json_response(response_text)
            llm_stopwords = set(self._clean_term_list(parsed.get("additional_stopwords", [])))
            preserve_terms = set(self._clean_term_list(parsed.get("preserve_terms", [])))
            additional_stopwords = sorted((heuristic_stopwords | llm_stopwords) - preserve_terms)
            return {
                "additional_stopwords": additional_stopwords,
                "preserve_terms": sorted(preserve_terms),
                "strategy": "llm",
            }
        except Exception:
            return fallback

    def label_topics(
        self,
        topics: list[dict[str, Any]],
        topic_context: str = "",
        source_name: str = "",
    ) -> list[dict[str, Any]]:
        labeled_topics = []

        for topic in topics:
            label = make_topic_fallback_label(topic.get("keywords", []))
            if self.is_configured:
                try:
                    candidate_label = ""
                    rejected_label = ""
                    for _ in range(2):
                        system_prompt, user_prompt = self._build_topic_label_prompt(
                            topic=topic,
                            topic_context=topic_context,
                            source_name=source_name,
                            rejected_label=rejected_label,
                        )
                        response_text = self._chat_completion(system_prompt, user_prompt)
                        parsed = self._parse_json_response(response_text)
                        candidate_label = self._clean_label(parsed.get("label", ""))
                        if candidate_label and not self._looks_like_keyword_dump(candidate_label, topic.get("keywords", [])):
                            break
                        rejected_label = candidate_label
                        candidate_label = ""
                    label = candidate_label or make_topic_fallback_label(topic.get("keywords", []))
                except Exception:
                    label = make_topic_fallback_label(topic.get("keywords", []))

            updated_topic = topic.copy()
            updated_topic["label"] = label
            labeled_topics.append(updated_topic)

        return labeled_topics

    def generate_business_insights(
        self,
        negative_topics: list[dict[str, Any]],
        positive_topics: list[dict[str, Any]],
        sentiment_distribution: dict[str, float],
    ) -> dict[str, str]:
        fallback = self._fallback_business_insights(
            negative_topics=negative_topics,
            positive_topics=positive_topics,
            sentiment_distribution=sentiment_distribution,
        )
        if not self.is_configured:
            return fallback

        system_prompt, user_prompt = self._build_business_insight_prompt(
            negative_topics=negative_topics,
            positive_topics=positive_topics,
            sentiment_distribution=sentiment_distribution,
        )

        try:
            response_text = self._chat_completion(system_prompt, user_prompt)
            parsed = self._parse_json_response(response_text)
            return {
                "customer_complaints_summary": str(parsed.get("customer_complaints_summary", fallback["customer_complaints_summary"])),
                "product_strengths": str(parsed.get("product_strengths", fallback["product_strengths"])),
                "business_implications": str(parsed.get("business_implications", fallback["business_implications"])),
                "product_improvement_suggestions": str(
                    parsed.get("product_improvement_suggestions", fallback["product_improvement_suggestions"])
                ),
            }
        except Exception:
            return fallback

    @staticmethod
    def _fallback_business_insights(
        negative_topics: list[dict[str, Any]],
        positive_topics: list[dict[str, Any]],
        sentiment_distribution: dict[str, float],
    ) -> dict[str, str]:
        top_negative_labels = ", ".join(topic.get("label", "Unlabeled Topic") for topic in negative_topics[:3]) or "no dominant negative topics"
        top_positive_labels = ", ".join(topic.get("label", "Unlabeled Topic") for topic in positive_topics[:3]) or "no dominant positive topics"
        negative_share = sentiment_distribution.get("negative", 0.0)
        positive_share = sentiment_distribution.get("positive", 0.0)

        return {
            "customer_complaints_summary": (
                "The main complaints cluster around "
                f"{top_negative_labels}. Negative sentiment represents {negative_share * 100:.1f}% of analyzed reviews."
            ),
            "product_strengths": (
                "Users most often praise "
                f"{top_positive_labels}. Positive sentiment represents {positive_share * 100:.1f}% of analyzed reviews."
            ),
            "business_implications": (
                "Recurring negative topics indicate where churn risk and support pressure are likely to increase, "
                "while repeated positive themes highlight the features that support retention and brand preference."
            ),
            "product_improvement_suggestions": (
                "Prioritize fixes for the top negative themes, validate those changes against fresh reviews, "
                "and reinforce the strongest positive themes in roadmap and product messaging."
            ),
        }
