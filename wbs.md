# Product Review Intelligence Platform WBS

## 1.0 Project Initiation
- 1.1 Define production scope from the research notebook
- 1.2 Confirm in-scope features for V1
- 1.3 Exclude research-only components from app scope
- 1.4 Define success criteria for deployment and usability

## 2.0 Research-to-Product Mapping
- 2.1 Review notebook and identify reusable logic
- 2.2 Extract validated preprocessing approach
- 2.3 Extract validated transformer sentiment workflow
- 2.4 Extract validated topic modeling workflow
- 2.5 Document gaps between notebook and production app

## 3.0 Solution Architecture
- 3.1 Finalize folder structure
- 3.2 Define module responsibilities
- 3.3 Define end-to-end pipeline contract
- 3.4 Define app configuration and secrets strategy
- 3.5 Define caching strategy for Streamlit

## 4.0 Project Setup
- 4.1 Create project directories
- 4.2 Initialize Python entrypoint and modules
- 4.3 Prepare `requirements.txt`
- 4.4 Prepare environment variable and secrets handling
- 4.5 Prepare base README structure

## 5.0 Utilities Layer
- 5.1 Build Play Store URL parser
- 5.2 Build app-id extraction helper
- 5.3 Build reusable validation helpers
- 5.4 Build error formatting and logging helpers
- 5.5 Build config/constants helper module

## 6.0 Review Scraper Module
- 6.1 Integrate `google-play-scraper`
- 6.2 Implement paginated review fetching
- 6.3 Enforce review limit up to 10000
- 6.4 Normalize scraper output schema
- 6.5 Return DataFrame with `date`, `rating`, `review_text`, `likes`
- 6.6 Add scraper error handling and edge-case handling

## 7.0 Preprocessing Module
- 7.1 Build text cleaning pipeline
- 7.2 Implement lowercase conversion
- 7.3 Remove URLs and HTML
- 7.4 Implement tokenization
- 7.5 Implement stopword removal with negation preservation
- 7.6 Implement lemmatization
- 7.7 Create `clean_text` output column
- 7.8 Add review quality filtering for empty/short text

## 8.0 Sentiment Analysis Module
- 8.1 Load HuggingFace sentiment model
- 8.2 Implement batched inference
- 8.3 Map predictions to `positive`, `neutral`, `negative`
- 8.4 Add sentiment column to dataset
- 8.5 Compute sentiment percentages
- 8.6 Optimize inference for Streamlit runtime constraints

## 9.0 Dataset Split Module
- 9.1 Split positive reviews
- 9.2 Split negative reviews
- 9.3 Exclude or park neutral reviews from topic modeling
- 9.4 Validate minimum review thresholds for modeling

## 10.0 Topic Modeling Module
- 10.1 Build vectorization pipeline with `CountVectorizer`
- 10.2 Build LDA modeling flow
- 10.3 Support configurable topic count
- 10.4 Extract top keywords per topic
- 10.5 Rank and select top 5 positive topics
- 10.6 Rank and select top 5 negative topics
- 10.7 Capture representative review samples for each topic

## 11.0 LLM Topic Labeling Module
- 11.1 Design prompt for topic labeling
- 11.2 Pass keywords and sample reviews to LLM
- 11.3 Generate concise human-readable topic labels
- 11.4 Handle API failures and fallback labels
- 11.5 Support Groq or OpenAI provider selection

## 12.0 LLM Business Insights Module
- 12.1 Design prompt for business insight generation
- 12.2 Provide negative topics, positive topics, and sentiment mix
- 12.3 Generate complaints summary
- 12.4 Generate strengths summary
- 12.5 Generate business implications
- 12.6 Generate improvement suggestions
- 12.7 Add structured output formatting for UI display

## 13.0 Visualization Module
- 13.1 Build sentiment distribution pie chart
- 13.2 Build sentiment trend over time chart
- 13.3 Build topic keyword bar charts
- 13.4 Build positive word cloud
- 13.5 Build negative word cloud
- 13.6 Standardize chart styling and display interfaces

## 14.0 Streamlit Frontend
- 14.1 Create app layout and branding
- 14.2 Add Play Store URL input section
- 14.3 Add Analyze button and validation flow
- 14.4 Add progress/status messaging
- 14.5 Add sentiment overview section
- 14.6 Add topic analysis section
- 14.7 Add AI insights section
- 14.8 Add visual analytics section
- 14.9 Add download/export options if needed

## 15.0 Performance and Reliability
- 15.1 Add `st.cache_data` for scrape and processing outputs
- 15.2 Add `st.cache_resource` for model loading
- 15.3 Limit review count for practical runtime
- 15.4 Add timeout and exception handling around LLM calls
- 15.5 Add handling for apps with sparse reviews or one-sided sentiment

## 16.0 Testing and Validation
- 16.1 Test URL parsing with multiple Play Store URL formats
- 16.2 Test scraper with valid and invalid app ids
- 16.3 Test preprocessing outputs on noisy reviews
- 16.4 Test sentiment predictions on sample reviews
- 16.5 Test topic modeling outputs on positive and negative subsets
- 16.6 Test LLM labeling and insight generation
- 16.7 Test Streamlit flow end to end
- 16.8 Validate Streamlit Cloud compatibility

## 17.0 Documentation
- 17.1 Write setup instructions
- 17.2 Write environment variable and secrets instructions
- 17.3 Write local run instructions
- 17.4 Write deployment instructions
- 17.5 Document module responsibilities and pipeline flow
- 17.6 Document limitations and known risks

## 18.0 Deployment Readiness
- 18.1 Finalize `app.py` entrypoint
- 18.2 Finalize dependency list
- 18.3 Verify `streamlit run app.py` works locally
- 18.4 Verify secrets configuration for hosted deployment
- 18.5 Prepare repository for Streamlit Cloud deployment

## 19.0 Final Deliverables
- 19.1 Modular codebase
- 19.2 Streamlit application
- 19.3 Requirements file
- 19.4 README
- 19.5 Deployment-ready project structure

## Recommended V1 Scope
- Include: scraper, preprocessing, transformer sentiment, topic modeling, LLM labeling, LLM insights, charts, word clouds, Streamlit UI
- Exclude: notebook-only experiments, Phase 2 reanalysis, custom classifier training, helpfulness prediction
