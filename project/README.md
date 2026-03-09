# ReviewPulse

Real-time user sentiment signals & Review Intelligence.

ReviewPulse is a Streamlit application for universal review analytics, with built-in Google Play support and CSV upload for any review dataset that can be mapped into the canonical schema.

## Features
- Google Play Store URL input with automatic app id extraction
- CSV upload mode for non-Play-Store review datasets
- Schema mapping into a unified review model
- Review scraping with `google-play-scraper`
- Text preprocessing with stopword removal, lemmatization, and negation preservation
- LLM-assisted vocabulary refinement for context-aware topic stopwords and preserve terms
- Transformer sentiment analysis with `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Positive and negative topic modeling with `CountVectorizer` and LDA
- Platform-managed topic labeling and business insights using Groq or OpenAI, with optional user key override
- Optional TrainWatcher email notifications for milestone, completion, and failure updates
- Plotly sentiment and topic charts
- Positive and negative word clouds
- Downloadable scored-review CSV, analysis-summary JSON, AI-summary PDF, and visuals PDF exports

## Project Structure
```text
project/
  app.py
  pipeline/
    ingestion.py
    scraper.py
    preprocess.py
    sentiment.py
    topic_model.py
    llm_analysis.py
  visualization/
    charts.py
    wordcloud.py
  utils/
    helpers.py
  requirements.txt
  README.md
```

## Setup
1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

3. Optional LLM configuration:

```bash
set GROQ_API_KEY=your_key_here
set OPENAI_API_KEY=your_key_here
```

Only one provider key is needed if you want LLM topic labels and business insights.
You can also define secrets in `.streamlit/secrets.toml` for local Streamlit runs and for Streamlit Community Cloud.
Use `.streamlit/secrets.toml.example` as the template.
If a key is stored in environment variables or Streamlit secrets, the deployed app will use it server-side and end users will not need to paste their own key.
If the platform-managed quota is exhausted, users can switch to a personal provider key for that session with `Your Own LLM`.

## Run
From the repository root:

```bash
streamlit run app.py
```

Or from the `project` directory:

```bash
streamlit run app.py
```

## Input Modes
### 1. Google Play Store
- Paste a Play Store app URL
- Reviews are scraped automatically
- App id is used as the analysis label and topic context

### 2. CSV Upload
- Upload a CSV containing review data
- Map your columns to the canonical schema in the UI
- Minimum required field: `review_text`
- Optional fields: `date`, `rating`, `likes`, `product_name`, `product_id`

Canonical schema used downstream:

```text
date
rating
review_text
likes
source
product_id
product_name
```

## Notes
- Review scraping is capped at 10000 reviews.
- Streamlit caching is used for scraping, preprocessing, topic modeling, and model loading.
- If no LLM API key is configured, the app falls back to heuristic topic labels and template business insights.
- Topic modeling is run only on positive and negative reviews. Neutral reviews are excluded from topic analysis.

## Pipeline Flow
```text
source input
  -> canonical ingestion
  -> vocabulary policy generation
  -> preprocessing
  -> transformer sentiment
  -> positive/negative split
  -> LDA topic modeling
  -> topic labels
  -> business insights
  -> charts, word clouds, exports
```

## Module Responsibilities
- `project/app.py`: Streamlit UI, caching, user flow, downloads
- `project/pipeline/ingestion.py`: source normalization into the canonical review schema
- `project/pipeline/scraper.py`: Google Play review fetching and schema cleanup
- `project/pipeline/preprocess.py`: deterministic text normalization plus preserve-term handling
- `project/pipeline/sentiment.py`: RoBERTa-based sentiment inference
- `project/pipeline/topic_model.py`: CountVectorizer and LDA topic extraction
- `project/pipeline/llm_analysis.py`: topic labels, business insights, vocabulary-policy generation
- `project/pipeline/orchestrator.py`: positive/negative split and end-to-end analysis assembly
- `project/visualization/*`: Plotly and word-cloud rendering

## Deployment
### Local
```bash
streamlit run app.py
```

### Streamlit Community Cloud
1. Push the repository to GitHub.
2. Set the app entrypoint to `app.py`.
3. In `Advanced settings`, choose a supported Python version and paste your secrets if you want LLM support:
```toml
PLATFORM_LLM_PROVIDER = "groq"
PLATFORM_LLM_MODEL = "llama-3.3-70b-versatile"
PLATFORM_LLM_API_KEY = "your_platform_managed_llm_key"
```
4. Alternatively, you can omit `PLATFORM_LLM_*` and provide only `GROQ_API_KEY` or `OPENAI_API_KEY`.
5. Keep review counts conservative for hosted CPU inference. Recommended interactive range: `500-2000`.

### Recommended Runtime Model
- Streamlit Cloud: UI + scraping + preprocessing + RoBERTa + topic modeling on CPU
- Groq/OpenAI: topic labels and business insights over API
- Colab or a GPU VM: optional benchmarking or heavy offline runs, not the main deployed backend

## Limitations and Risks
- Large review counts can make transformer inference slow on CPU-only hosts.
- Topic quality degrades when there are too few positive or negative reviews.
- CSV uploads still depend on good column mapping and reasonably clean text.
- Google Play scraping behavior can change with upstream service changes or rate limits.
- LLM-generated vocabulary policies and business insights are helpful but not deterministic.
- Community-cloud deployments are not appropriate for large batch workloads.

## Tests
```bash
pytest
```
