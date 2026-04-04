# NSF NRT AI Challenge Starter Kit


This starter kit is built for the UMKC NSF NRT AI challenge on substance abuse risk detection. It is designed to work with **API-based data access** so you do not need to download large datasets locally.

## Detailed guide
- Full explanation of outputs + what the app is doing: [docs/RESULTS_AND_APP_EXPLAINED.md](docs/RESULTS_AND_APP_EXPLAINED.md)

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
├── apps/
│   ├── __init__.py
│   ├── pipeline.py
│   └── dashboard.py
├── config/
│   └── data_endpoints.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_sources/
│   │   ├── __init__.py
│   │   ├── cdc_api.py
│   │   ├── census_api.py
│   │   ├── nida_api.py
│   │   ├── reddit_api.py
│   │   └── trends_api.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── ews.py
│   │   ├── fusion.py
│   │   ├── semantic_signals.py
│   │   ├── sentiment.py
│   │   └── text_signals.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── build_graph.py
│   │   ├── graph_analytics.py
│   │   └── temporal_graph.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── risk_narrator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── anomaly.py
│   │   ├── arima_forecast.py
│   │   ├── ensemble.py
│   │   ├── forecast.py
│   │   └── policy_sim.py
│   ├── utils/
│   │   ├── ts.py
│   │   └── __init__.py
│   └── video/
│       ├── __init__.py
│       └── video_signals.py
├── scripts/
│   ├── run_pipeline.py
│   └── app.py
└── data/
    └── cache/
```

## File guide (what each file does)

### Apps (recommended entrypoints)
- apps/pipeline.py: CLI pipeline that fetches data (CDC/Census/NIDA + optional Reddit/Trends/video), computes EWS, runs fusion/forecast/anomalies, builds graphs, and writes outputs to data/cache/.
- apps/dashboard.py: Streamlit dashboard that reads outputs from data/cache/ and visualizes risk, temporal snapshots, graphs, policy sims, and optional video window signals.

### Scripts (legacy wrappers)
- scripts/run_pipeline.py: Backward-compatible wrapper that delegates to apps/pipeline.py.
- scripts/app.py: Backward-compatible wrapper that delegates to apps/dashboard.py.

### Core library (used by apps/)
- src/data_sources/*: API clients (CDC, Census, NIDA/SAMHSA; optional Reddit and Google Trends).
- src/features/ews.py: Early Warning Score (EWS) calculation.
- src/features/fusion.py: Multimodal signal normalization + fusion into a single score/alert.
- src/features/sentiment.py and src/features/text_signals.py: Lexicon-style text scoring (privacy-safe exports by default).
- src/features/semantic_signals.py: Embedding-style semantic similarity signals (sentence-transformers if available, TF-IDF fallback).
- src/models/*: Forecasting, anomaly detection, and policy simulation.
- src/graph/*: Knowledge graph build + temporal snapshots and edge-change summaries.
- src/llm/risk_narrator.py: Deterministic narrative generator (no external LLM required).
- src/video/video_signals.py: Optional video-to-windowed behavioral signals skeleton (OpenCV-based; non-identifying aggregates only).

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
# Recommended (module run; no PYTHONPATH needed)
python -m apps.pipeline --state KS --use-reddit false --use-trends false --horizon 12 --snapshots 6

# Multi-state run
python -m apps.pipeline --states KS,MO,OK --use-reddit false --use-trends true --horizon 12 --snapshots 6

# Enable policy simulation outputs
python -m apps.pipeline --state KS --simulate-policy true --intervention naloxone_distribution

# Optional: enable Reddit + semantic comparison (privacy-safe by default)
python -m apps.pipeline --state KS --use-reddit true --use-trends false

# Optional: enable video-derived behavioral signals (requires opencv-python)
# Comma-separated local file paths:
python -m apps.pipeline --state KS --use-video true --video-paths "/path/a.mp4,/path/b.mov"

# Legacy wrappers (still supported)
PYTHONPATH=. python scripts/run_pipeline.py --state KS --use-reddit false --use-trends false
```

## Launch dashboard
```bash
streamlit run apps/dashboard.py

# Legacy wrapper (still supported)
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

### Privacy / ethics defaults
- Reddit exports are **privacy-safe by default**: raw title/selftext and post IDs are not written to disk.
- If you must export raw text for debugging, explicitly opt-in with `--save-reddit-raw true` (not recommended for deliverables).

### New outputs
- `data/cache/reddit_method_compare_{STATE}.csv` compares lexicon vs semantic (embedding-style) aggregate scores.
- `data/cache/video_windows_{STATE}.csv` contains only window-level, non-identifying behavioral metrics.
- `data/cache/narrative_{STATE}.md` is a Markdown version of the narrative brief (same content as the JSON).
