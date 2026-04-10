# DMARG — Results & App Explanation (Full Guide)

This document explains, in plain language, what the project is doing, what outputs it produces, and how to interpret what you see in the Streamlit dashboard.

It is written to match the current codebase structure:
- Pipeline entrypoint: `apps/pipeline.py`
- Dashboard entrypoint: `apps/dashboard.py`
- Outputs directory: `data/cache/`

---

## 1) What the system is (high-level)

**DMARG (Dynamic Multimodal Addiction Risk Graph)** is a *population-level* risk-signal pipeline. It is not an individual-level prediction tool.

At a high level, DMARG:
1. Pulls public/official signals (CDC overdose-related series)
2. Pulls demographic context (Census)
3. Pulls additional indicators (NIDA/SAMHSA style statistics)
4. Optionally pulls behavioral proxy signals (Reddit text, Google Trends, and/or non-identifying video-derived window metrics)
5. Converts these into normalized numeric signals
6. Produces:
   - **EWS** (Early Warning Score)
   - **Fusion score** (multimodal combined score + alert)
   - **Forecast** (near-term trajectory)
   - **Anomalies** (spike detection)
   - **Knowledge graph** (entities + relationships)
   - **Temporal graph** summary (snapshots + edge changes)
   - **Policy simulation** (optional)
   - **Narrative brief** (JSON + Markdown)

---

## 2) What happens when you run the pipeline

When you run:

```bash
python -m apps.pipeline --state KS --use-reddit false --use-trends false --horizon 12 --snapshots 6
```

the pipeline:

### Step A — Fetch / build base time series
- **CDC county overdose dataset** is pulled for the chosen state.
- A monthly time series is inferred (or synthetically filled if the raw CDC result doesn’t include month granularity).
- Outputs are stored as CSVs.

### Step B — Pull context & additional indicators
- **Census context** provides demographic/environmental context.
- **NIDA/SAMHSA client** supplies state-level substance-related statistics (real if available, synthetic fallback if not).

### Step C — Optional behavioral proxy signals
Depending on flags:
- `--use-reddit true`:
  - Pull a small sample of posts matching substance-abuse related queries.
  - Compute lexicon-based “distress” / “substance” features.
  - Compute semantic similarity features (SBERT if installed, TF‑IDF fallback).
  - By default, **raw text is NOT written** to disk.
- `--use-trends true`:
  - Pull Google Trends interest summaries (or fallback).
- `--use-video true --video-paths ...`:
  - Extract non-identifying window-level metrics from local videos (OpenCV required).
  - Only aggregated signals are used in risk.

### Step D — Convert raw signals into risk inputs
- Signals are normalized to comparable scales.
- Signals are grouped into “domains” (CDC, socioeconomics, social text, trends, video, etc.).

### Step E — Produce decision-support outputs
- **EWS**: a simple, interpretable early warning score.
- **Fusion**: combined score + alert level + confidence.
- **Forecast**: next `--horizon` points for the overdose proxy series.
- **Anomalies**: spike detection on the recent historical series.
- **Graphs**:
  - A snapshot knowledge graph
  - Temporal snapshots over the last `--snapshots` points, plus edge-change summaries
- **Narrative**: a short brief that summarizes the outputs in a human-readable format.

---

## 3) Where outputs are written

All outputs are written under:

- `data/cache/`

This is intentionally “flat” so it’s easy for a dashboard to read files.

---

## 4) Output files (what each file means)

Below is a practical “output dictionary”. Each file is named by state (e.g., `KS`, `CA`).

### 4.1 Core risk outputs

#### `risk_{STATE}.csv`
**What it is:** the main single-row table of computed risk inputs and the computed EWS.

**Why it exists:** provides a compact, dashboard-friendly snapshot of the state’s risk situation.

**Typical columns you’ll see (example meanings):**
- `state`: state abbreviation
- `overdose_rate_per_100k`: proxy value derived from CDC time series (named as a rate, but may be a proxy depending on CDC fields)
- `poverty_rate_pct`: Census-derived context
- `opioid_misuse_pct`: NIDA/SAMHSA indicator
- `trend_velocity`: short-term slope / change measure in the overdose time series
- `social_composite`: lexicon-based composite social risk signal (if Reddit enabled; else 0)
- `semantic_social_composite`: semantic similarity composite social risk signal (if Reddit enabled; else 0)
- `trends_interest`: Google Trends interest proxy (if enabled; else 0)
- `video_*`: non-identifying aggregate metrics from video (if enabled; else 0)
- `ews`: the overall Early Warning Score (0–1)
- `ews_alert_level` (if present): qualitative label such as Low / Medium / High
- `ews_{domain}` columns: domain-level components that sum/aggregate into the EWS

**How to interpret:**
- Use `ews` as the quick “warning dial”.
- Use `ews_*` columns to understand *why* EWS is high/low.
- Use `trend_velocity` to see whether risk is moving up/down.

#### `fusion_{STATE}.csv`
**What it is:** the multimodal fusion output (score + alert + confidence + per-domain scores).

**Why it exists:** EWS is interpretable; fusion is designed to combine multiple domains more systematically.

**Typical columns:**
- `fusion_score`: combined score (0–1)
- `alert_level`: label derived from score thresholds
- `confidence`: heuristic based on signal availability / stability
- `domain_*`: per-domain contributions

**How to interpret:**
- If `fusion_score` is high and `confidence` is also high, it’s a stronger warning.
- If `fusion_score` is high but `confidence` is low, the system is warning but acknowledging missing/noisy signals.

---

### 4.2 Time series & forecasting outputs

#### `overdose_ts_{STATE}.csv`
**What it is:** the monthly time series used for forecasting and anomalies.

**Why it exists:** provides transparent access to the underlying series.

**Typical columns:**
- `date`: monthly timestamp
- `value`: overdose proxy value

**How to interpret:**
- Trend direction and volatility here strongly influence forecast and anomalies.

#### `forecast_{STATE}.csv`
**What it is:** near-term forecast table for the series.

**Typical columns:**
- `step`: forecast step number (1..horizon)
- `forecast`: expected value
- `lower_ci` / `upper_ci` (if present): uncertainty bounds

**How to interpret:**
- A rising forecast suggests increasing risk pressure.
- Wide CI bands indicate high uncertainty.

---

### 4.3 Anomaly (spike) detection

#### `anomalies_{STATE}.csv`
**What it is:** recent history with anomaly scores.

**Typical columns:**
- `date` (or an index)
- `value`: observed series value
- `anomaly_score`: 0–1 spike score (higher means “more unusual”)
- Sometimes additional columns depending on detector internals

**How to interpret:**
- Look for points where `anomaly_score` is high — these are candidates for sudden changes.
- Anomaly != “bad”; it means “unusual” and should prompt investigation.

---

### 4.4 Knowledge graph exports

#### `graph_nodes_{STATE}.csv`
**What it is:** node list export of the state’s knowledge graph.

**Typical columns:**
- `node`: node identifier used by edges
- `node_type`: category (e.g., `location`, `event`, `risk_score`, `context`)
- `label`: human-readable label for display
- `risk_score`: optional scalar used for sizing/importance in the dashboard
- `color`: hex color used by the dashboard visualization

**How to interpret:**
- Nodes represent concepts and measured signals.

#### `graph_edges_{STATE}.csv`
**What it is:** edge list export of the state’s knowledge graph.

**Typical columns:**
- `source`, `target`
- `relation`: semantic relationship label
- `weight`: strength / importance

**How to interpret:**
- Edges represent relationships (e.g., “has_signal”, “correlates_with”, “influences”).

---

### 4.5 Temporal graph outputs (dynamic behavior)

#### `temporal_summary_{STATE}.csv`
**What it is:** per-snapshot summary metrics for a sequence of graph snapshots.

**Typical columns:**
- `timestamp`
- `mean_ews` / `max_ews` (depending on snapshot computation)
- Possibly node/edge counts

**How to interpret:**
- Think of this as “risk evolution” across the last `--snapshots` points.

#### `temporal_edges_{STATE}.csv`
**What it is:** time-indexed edge weights across snapshots.

**Typical columns:**
- `timestamp`, `source`, `target`, `relation`, `weight`

**How to interpret:**
- The dashboard uses this to rank edges by:
  - absolute change (first → last)
  - volatility (standard deviation)
- This highlights which relationships are strengthening/weakening over time.

---

### 4.6 Policy simulation outputs (optional)

#### `policy_{STATE}_{INTERVENTION}.csv`
**What it is:** baseline vs scenario trajectories for one intervention.

**Typical columns:**
- `month`
- `value_baseline` (or baseline series)
- `value_scenario` (scenario mean)

**How to interpret:**
- This is a *toy decision-support simulation*, not a causal guarantee.
- Use it to compare “directional impact” under simplified assumptions.

#### `policy_compare_{STATE}.csv`
**What it is:** summary comparison across multiple interventions.

**How to interpret:**
- Rank interventions by expected reduction / improvement metrics.

---

### 4.7 Optional Reddit/semantic outputs

#### `reddit_{STATE}.csv`
**What it is:** scored Reddit-derived features (privacy-safe by default).

**Privacy default behavior:**
- By default, raw `title`, `selftext`, combined `text`, and post `id` are not exported.

**How to interpret:**
- Look for columns like `distress_score`, `substance_score`, `composite_risk`, and semantic columns.

#### `reddit_method_compare_{STATE}.csv`
**What it is:** aggregated comparison between lexicon vs semantic methods.

**How to interpret:**
- If semantic is higher than lexicon, it suggests the language is “semantically close” to seed phrases even if explicit keywords are missing.

---

### 4.8 Optional Google Trends outputs

#### `trends_{STATE}.csv`
**What it is:** trends summary categories (opioid/stimulant/treatment) with mean interest.

**How to interpret:**
- Rising interest in treatment terms can mean help-seeking behavior; rising interest in drug names can signal exposure/concern.

---

### 4.9 Optional video outputs

#### `video_windows_{STATE}.csv`
**What it is:** window-level non-identifying behavioral metrics extracted from video(s).

**What it is NOT:** it is not face recognition, identity tracking, or individual-level classification.

**Typical columns:**
- `window_index` (or `start_sec`/`end_sec`): time window
- `activity_mean`: coarse motion magnitude proxy
- `anomaly_score`: unusual motion/lighting proxy
- `low_light_frac`: fraction of low-brightness frames in window
- `scene_change_rate`: crude scene-change frequency proxy

**How to interpret:**
- High activity_mean could indicate more motion.
- High low_light_frac indicates poor lighting.
- High anomaly_score indicates “something changed” relative to baseline windows.

#### `video_status_{STATE}.json`
**What it is:** lightweight metadata about the last attempted video-processing step for that state.

**Why it exists:** lets the dashboard distinguish:
- video disabled (default)
- video enabled but missing `--video-paths`
- video enabled but failed (e.g., OpenCV missing)
- video enabled and ran but produced 0 windows

---

### 4.10 Narrative outputs

#### `narrative_{STATE}.json`
**What it is:** machine-readable narrative brief.

**How to interpret:**
- `summary`: paragraph-level brief
- `bullets`: key points
- `caveats`: important disclaimers

#### `narrative_{STATE}.md`
**What it is:** the same narrative brief in Markdown for easy sharing.

---

## 5) What is happening in the Streamlit app (dashboard)

Run:

```bash
streamlit run apps/dashboard.py
```

### What the app does
- Reads output files from `data/cache/` for the selected state.
- Creates interactive charts and tables for quick inspection.
- Does not recompute the pipeline; it visualizes existing outputs.

### Sidebar controls
- **State abbreviation**: selects which `*_STATE.*` files to load.
- **Show national outputs** (if available): loads `graph_nodes_national.csv` and `graph_edges_national.csv` when present.

### Top metrics row
- **EWS**: quick warning score.
- **Trend Velocity**: short-term slope/velocity of the overdose series.
- **Fusion Score**: multimodal score (if available).

### Tabs (what each one shows)

#### Overview
- Bar chart of `ews_*` domain components.
- Displays `fusion_{STATE}.csv` if present.

#### Temporal
- Plots snapshot summary (`temporal_summary_{STATE}.csv`).
- Edge trajectories:
  - ranks edges by change/volatility
  - lets you pick an edge and plot weight over time

#### Forecast
- Displays `forecast_{STATE}.csv` as a line plot (with CI bands if present).

#### Anomalies
- Plots observed `value` and `anomaly_score` from `anomalies_{STATE}.csv`.

#### Knowledge Graph
- Shows an interactive graph visualization (nodes + edges) with relation/weight filters.
- Node/edge tables remain available in an expander for debugging/export.

#### Policy Simulation
- Shows `policy_compare_{STATE}.csv` if present (requires pipeline run with policy simulation enabled).

#### Video
- If `video_windows_{STATE}.csv` exists, plots the chosen window metrics.
- If not present, it explains how to enable video mode.
- If `video_status_{STATE}.json` exists, the dashboard shows more specific guidance (missing OpenCV, missing paths, etc.).

#### Narrative
- Loads and renders `narrative_{STATE}.json` into a human-friendly view.

#### Raw Data
- Shows sample CDC rows and the full risk row.

---

## 6) “What does it mean?” — interpreting results as a story

A typical interpretation workflow:
1. **Check EWS + alert** in `risk_{STATE}.csv`.
2. Look at **domain contributions** (`ews_*`) to see what is driving the score.
3. Check `trend_velocity` and the `overdose_ts_{STATE}.csv` plot trend.
4. Look at `anomalies_{STATE}.csv` for spikes.
5. Look at `forecast_{STATE}.csv` to see the near-term projected direction.
6. Use the knowledge graph outputs to explain which concepts/signals are connected.
7. If enabled:
   - Compare lexicon vs semantic in `reddit_method_compare_{STATE}.csv`.
   - Inspect `video_windows_{STATE}.csv` trends for non-identifying behavioral shifts.

---

## 7) Privacy / ethics notes (important)

This project is designed to align with “population-level, non-identifying” constraints:
- Reddit exports are privacy-safe by default (no raw text saved).
- Video signals are aggregated, window-level metrics only.
- The system should be used as **early warning + decision support**, not as a tool to identify or target individuals.

---

## 8) Troubleshooting (common issues)

### “Dashboard says run pipeline first”
- You don’t have `data/cache/risk_{STATE}.csv` yet.
- Run the pipeline for that state.

### “No video outputs”
- You didn’t run with `--use-video true` and `--video-paths ...`, or OpenCV isn’t installed.
- If present, check `data/cache/video_status_{STATE}.json` for the exact error (e.g., missing `opencv-python`).

### “No Reddit outputs”
- You didn’t run with `--use-reddit true`, or credentials aren’t set.

### “Some outputs are zero”
- Optional sources default to 0 when disabled.
- This is expected and keeps the pipeline robust.
