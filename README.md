# ReviewPulse

Real-time user sentiment signals & Review Intelligence.

ReviewPulse is a review analytics platform built from a research notebook into a deployable Streamlit product. It combines transformer sentiment analysis, topic modeling, and LLM-based semantic interpretation to turn raw user reviews into product, CX, and business signals.

This repository contains both:

- a research notebook: [project_notebook.ipynb](project_notebook.ipynb)
- a deployable Streamlit application: [app.py](app.py)

## Project Story

The work started as a research notebook focused on the **JioHotstar Android app** and then evolved into a reusable application that supports:

- Google Play review scraping
- CSV upload for external review datasets
- one canonical schema for downstream analysis
- semantic topic interpretation with Groq or OpenAI
- interactive dashboards and exportable outputs

The notebook is the research artifact. The app is the production-style reimplementation of the validated path.

## Research Foundation

The notebook [project_notebook.ipynb](project_notebook.ipynb) documents the original JioHotstar review study.

### Research objective

The core goal was to replace traditional questionnaire-based feedback analysis with automated review mining so management can understand:

- overall customer sentiment
- recurring complaints
- product strengths
- business implications
- improvement opportunities

### Research data source

The notebook uses Google Play Store reviews for:

- product: `JioHotstar Android App`
- app id: `in.startv.hotstar`

The dataset includes fields such as:

- `date`
- `rating`
- `review_text`
- `likes`

The notebook documents that some fields requested in the academic assignment were not available through the Google Play review source.

### Research methodology

The notebook follows a phased methodology:

1. **Phase 1: Full dataset sentiment and topic modeling**
   - scrape large-scale Play Store reviews
   - clean and normalize text
   - run transformer sentiment analysis
   - split positive and negative review groups
   - run topic modeling on each group
   - interpret themes and summarize business implications

2. **Phase 2: Recent 25% reanalysis**
   - isolate the most recent portion of reviews
   - compare more recent customer sentiment and topics against the broader dataset

3. **Phase 3: Robust binary sentiment classifier**
   - explore stronger supervised classification behavior beyond the core app pipeline

4. **Phase 4: Helpfulness prediction**
   - experiment with modeling review helpfulness as a separate research problem

### What was productized

Not every notebook experiment belongs in a deployable app.

The production app keeps the strongest, most reusable core:

- ingestion
- cleaning and preprocessing
- RoBERTa sentiment analysis
- positive/negative topic discovery
- LLM topic labeling
- LLM business insight generation
- executive dashboarding and exports

The app intentionally does **not** ship the extra research-only branches such as the helpfulness model or notebook-specific exploratory workflows.

## Product Overview

ReviewPulse is positioned as:

**Universal review analytics for any review dataset, with built-in Google Play support.**

It can analyze:

- Google Play Store reviews from a pasted app URL
- uploaded CSV exports from marketplaces, apps, or partner systems

## Core Product Features

- Google Play Store URL input with automatic app id extraction
- CSV upload mode for external review datasets
- canonical schema mapping for unified downstream processing
- transformer sentiment analysis with `cardiffnlp/twitter-roberta-base-sentiment-latest`
- LDA-based topic modeling for positive and negative reviews
- LLM-generated semantic topic labels
- LLM-generated product and business insights
- dark/light executive dashboard UI
- downloadable CSV, JSON, and PDF outputs

## End-to-End Pipeline

```text
source input
  -> canonical ingestion
  -> vocabulary policy generation
  -> preprocessing
  -> transformer sentiment
  -> positive/negative split
  -> topic modeling
  -> topic labeling
  -> business insights
  -> dashboard visuals and exports
```

## Technical Stack

### Frontend

- Streamlit

### Backend and analysis

- Python
- pandas
- NumPy

### NLP and ML

- HuggingFace Transformers
- PyTorch
- scikit-learn
- NLTK

### Visualization

- Plotly
- Matplotlib
- WordCloud

### LLM integration

- Groq API
- OpenAI API

### Scraping

- `google-play-scraper`

## App Design

The deployed app is designed as an executive-friendly analytics dashboard rather than a raw notebook output.

### UX structure

- Product snapshot
- sentiment overview
- positive and negative theme panels
- AI summary
- visuals
- review explorer
- downloads

### LLM access modes

The app supports three LLM modes:

- `Our Default LLM`
- `Your Own LLM`
- `Disable LLM`

In deployment, `Our Default LLM` is expected to use backend-managed secrets. During local testing, `Your Own LLM` is the practical option if no backend key is configured.

### Supported outputs

- sentiment distribution
- sentiment trend
- positive themes
- negative themes
- LLM insight cards
- review explorer filters
- scored review CSV
- analysis summary JSON
- AI summary PDF
- visuals PDF

## Repository Structure

```text
E:\A_6
├── app.py
├── project_notebook.ipynb
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── wbs.md
├── .streamlit/
│   └── secrets.toml.example
├── project/
│   ├── app.py
│   ├── README.md
│   ├── pipeline/
│   │   ├── ingestion.py
│   │   ├── scraper.py
│   │   ├── preprocess.py
│   │   ├── sentiment.py
│   │   ├── topic_model.py
│   │   ├── llm_analysis.py
│   │   └── orchestrator.py
│   ├── utils/
│   │   └── helpers.py
│   └── visualization/
│       ├── charts.py
│       └── wordcloud.py
└── tests/
```

## Module Roles

- [project/app.py](project/app.py): Streamlit UI, theming, layout, downloads, orchestration entry
- [project/pipeline/ingestion.py](project/pipeline/ingestion.py): source normalization into the canonical schema
- [project/pipeline/scraper.py](project/pipeline/scraper.py): Google Play review scraping
- [project/pipeline/preprocess.py](project/pipeline/preprocess.py): text cleaning, stopwords, negation preservation, lemmatization fallback logic
- [project/pipeline/sentiment.py](project/pipeline/sentiment.py): RoBERTa sentiment inference
- [project/pipeline/topic_model.py](project/pipeline/topic_model.py): CountVectorizer + LDA topic extraction
- [project/pipeline/llm_analysis.py](project/pipeline/llm_analysis.py): vocabulary policy, topic labeling, business insights
- [project/pipeline/orchestrator.py](project/pipeline/orchestrator.py): end-to-end analysis assembly
- [project/visualization/charts.py](project/visualization/charts.py): Plotly chart builders
- [project/visualization/wordcloud.py](project/visualization/wordcloud.py): themed word cloud rendering
- [tests/](tests): automated coverage for ingestion, preprocessing, sentiment, topics, LLM parsing, and visuals

## Local Setup

Use your Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For tests:

```bash
pip install -r requirements-dev.txt
```

## Run Locally

From the repository root:

```bash
streamlit run app.py
```

In your environment, the validated Python interpreter has been:

```text
C:\Users\sambi\miniconda3\envs\myenv\python.exe
```

So the explicit command is:

```powershell
cd E:\A_6
C:\Users\sambi\miniconda3\envs\myenv\python.exe -m streamlit run app.py
```

## LLM Configuration

For local testing, you can use environment variables or `.streamlit/secrets.toml`.

Example environment variables:

```bash
set GROQ_API_KEY=your_key_here
set OPENAI_API_KEY=your_key_here
```

Example Streamlit secrets template:

```toml
PLATFORM_LLM_PROVIDER = "groq"
PLATFORM_LLM_MODEL = "llama-3.3-70b-versatile"
PLATFORM_LLM_API_KEY = "your_platform_managed_llm_key"
```

See [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example).

## Testing

Run:

```bash
pytest
```

The current codebase has local automated coverage for:

- helpers
- ingestion
- preprocessing
- sentiment
- topic modeling
- LLM response handling
- word cloud generation

## Deployment

The intended deployment target is **Streamlit Community Cloud**.

### Deployment flow

1. Push this repository to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Set the entrypoint to `app.py`.
4. Add backend-managed secrets in `Advanced settings`.
5. Deploy.

### Recommended secrets

```toml
PLATFORM_LLM_PROVIDER = "groq"
PLATFORM_LLM_MODEL = "llama-3.3-70b-versatile"
PLATFORM_LLM_API_KEY = "your_platform_managed_llm_key"
```

Alternative:

- use `GROQ_API_KEY`
- or use `OPENAI_API_KEY`

### Recommended runtime usage

- Streamlit Cloud for UI and CPU-based analysis
- Groq or OpenAI for semantic LLM calls
- keep hosted interactive review counts moderate, ideally `500-2000`

## Limitations

- transformer inference can be slow for large review volumes on CPU-only deployment
- topic quality depends on having enough reviews in positive and negative segments
- Google Play scraping behavior can change over time
- LLM outputs are useful but not deterministic
- the research notebook contains exploratory branches that are intentionally not part of the deployed product flow

## Notebook and App Relationship

This repository should be read in two layers:

1. [project_notebook.ipynb](project_notebook.ipynb)
   - research artifact
   - problem framing
   - exploratory analysis
   - methodology validation

2. [app.py](app.py) + [project/](project)
   - deployable product
   - modular architecture
   - reusable pipeline
   - production-style UI and exports

## Additional Documentation

Detailed app-level documentation is available in [project/README.md](project/README.md).
