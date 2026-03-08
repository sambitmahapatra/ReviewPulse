# ReviewPulse

Real-time user sentiment signals & Review Intelligence.

Deployable Streamlit application for review analytics across Google Play and uploaded CSV datasets.

Main documentation: [project/README.md](project/README.md)

## Run locally

```bash
streamlit run app.py
```

## Publish on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, click `Create app`.
3. Select the repository, branch, and root entrypoint file: `app.py`.
4. In `Advanced settings`, paste the contents of `.streamlit/secrets.toml.example` after replacing the placeholder key values.
5. Deploy.
