# NSF NRT AI Challenge Starter Kit

## Project idea
**Dynamic Multimodal Addiction Risk Graph (DMARG)**

This starter kit is built for the UMKC NSF NRT AI challenge on substance abuse risk detection. It is designed to work with **API-based data access** so you do not need to download large datasets locally.

### What it does
- Pulls overdose and public-health signals from the **CDC Socrata API**
- Pulls demographic context from the **U.S. Census API**
- Optionally pulls social-signal data from the **Reddit API** if you have credentials
- Builds an **Early Warning Score (EWS)**
- Creates a simple **dynamic knowledge graph**
- Produces a baseline **forecast** and a Streamlit dashboard

## Why this idea is strong
Most teams will stop at classification or dashboards. This project adds:
1. **Dynamic graph construction**
2. **Population-level early warning**
3. **Forecasting**
4. **Explainable signals from multiple sources**
5. **API-first design** for low-storage development

## Overall structure
```
nrt_zip_project/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── data_endpoints.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_sources/
│   │   ├── __init__.py
│   │   ├── cdc_api.py
│   │   ├── census_api.py
│   │   └── reddit_api.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── text_signals.py
│   │   └── ews.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── forecast.py
│   └── graph/
│       ├── __init__.py
│       └── build_graph.py
├── scripts/
│   ├── run_pipeline.py
│   └── app.py
└── data/
    └── cache/
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env
```

## Environment variables
Add these to `.env` when available:
```env
CDC_APP_TOKEN=
CENSUS_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=dmarg-research/0.1 by you
```

## Run the pipeline
```bash
# Recommended (ensures `import src...` works when running scripts directly)
PYTHONPATH=. python scripts/run_pipeline.py --state KS --use-reddit false --use-trends false --horizon 12 --snapshots 6

# Multi-state run
PYTHONPATH=. python scripts/run_pipeline.py --states KS,MO,OK --use-reddit false --use-trends true --horizon 12 --snapshots 6

# Enable policy simulation outputs
PYTHONPATH=. python scripts/run_pipeline.py --state KS --simulate-policy true --intervention naloxone_distribution
```

## Launch dashboard
```bash
streamlit run scripts/app.py
```

## Recommended build path
### Phase 1: minimum winning demo
- CDC overdose API
- Census API
- Early Warning Score
- Streamlit dashboard

### Phase 2: stronger novelty
- Optional Reddit connector
- Dynamic graph edges over time
- Forecasting with intervention simulation

## Suggested demo storyline
1. Pull recent overdose data from CDC
2. Pull county/state context from Census
3. Pull optional behavioral signals
4. Compute EWS
5. Show graph nodes and edges
6. Forecast near-term risk
7. Show one intervention slider in Streamlit

## Notes
- This template is intentionally lightweight and storage-friendly.
- It pulls only the rows you request via APIs.
- For the AI challenge, stay at **population level** and avoid identifying individuals.
