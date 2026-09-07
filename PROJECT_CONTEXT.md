# PROJECT_CONTEXT.md — SkyGuard AI (SIH26073)
> SINGLE SOURCE OF TRUTH. Paste this whole file (or link the GitHub raw URL) into ANY new AI session
> (Claude, Antigravity, Gemini, new account) as the FIRST message before asking for any code.
> Rule: "Read this file completely before writing or changing anything. Do not assume architecture
> not written here. If something is ambiguous, ask, don't guess."
Last updated: 2026-09-07 | Owner: 2-person team | Status: PRE-BUILD (Day 0)

---

## 1. PROBLEM STATEMENT (VERIFIED, OFFICIAL — sih2026.vuce.in/ps/SIH26073)
- **ID**: SIH26073, Ministry of Earth Sciences (MoES) / IMD, Theme: Disaster Management, Category: Software.
- **Task**: Real-time AI/ML anomaly detection for AWS using ONLY Temperature (°C), Pressure (hPa), Relative Humidity (%).
- **Must distinguish**: genuine meteorological events vs sensor/data anomalies, minimizing false alarms, scalable design.
- **Required outputs**: real-time anomaly alerts, severity + confidence scores, root-cause classification,
  visualization dashboard, sensor health status, corrected-value estimation (OPTIONAL).
- **Suggested (not mandatory) tech**: Explainable AI SHAP/LIME ("Preferable"), Edge AI for ESP32 (low-power option).
- **Evaluation is explicitly done on ANOMALY-INJECTED DATA** — official confirmation that synthetic anomaly
  injection is the sanctioned ground-truth methodology, not a workaround.
- **Expected inputs explicitly permit**: historical AWS datasets, simulated anomalies, OR streaming data. Real
  hardware is NOT required.
- **Final deliverable**: "Fully executable code with example usage and a document explaining various use cases."
  No mandatory hardware demo, no mandatory ESP32 flash, no mandatory live station feed.
- **Known ambiguity (resolve explicitly, don't hide it)**: the worked example resolves the case by comparing
  "neighboring stations" (spatial check), even though the parameter list says "only" T/P/H. Our resolution:
  spatial/neighboring-station comparison is a DEMO-ONLY / OPTIONAL enrichment layer, never a dependency of the
  core detector. The core system must work with single-station T/P/H only (see Section 5).

## 2. PROJECT IDENTITY
- **Name**: SkyGuard AI — Intelligent Real-Time Anomaly Detection & Sensor Health System for AWS.
- **Team**: 2 people, beginner ML/deployment knowledge, ~3 productive build days (7–10 Sep 2026).
- **Objective**: NOT the most sophisticated model. The most credible, explainable, reliably-executable,
  demoable prototype that scores well against the published rubric (see Section 3).

## 3. EVALUATION RUBRIC (OFFICIAL WEIGHTS — DO NOT REORDER PRIORITIES)
| Criteria | Weight | Our build priority |
|---|---|---|
| Innovation & Novelty | 25% | HIGH — hybrid physics+ML+context reasoning, not "we used AI" |
| Detection Accuracy | 20% | HIGH — measured on injected test set, precision/recall/F1 reported |
| Real-Time Capability | 15% | MEDIUM — simple streaming simulator, not enterprise infra |
| Explainability | 10% | HIGH — SHAP + rule-based reasons, built in from day 1, not bolted on |
| Scalability | 10% | LOW-effort, HIGH-talk — stateless per-station design, explain in doc |
| Practical Deployability | 10% | MEDIUM — Streamlit/HF Spaces deployed link |
| Visualization/UI | 5% | MEDIUM — Streamlit dashboard, no more |
| Energy Efficiency | 5% | LOW-effort — mention lightweight model (no GPU, no deep learning) |

## 4. FINAL ARCHITECTURE (FROZEN — DO NOT REDESIGN MID-BUILD)
```
DATA INPUT (CSV replay / simulator, 1 row every 2-5s)
   -> VALIDATION (range/physical sanity checks - rule layer)
   -> PREPROCESSING (missing-value handling, resampling, timestamp align)
   -> FEATURE ENGINEERING (rate-of-change, rolling z-score, rolling mean/std,
      dewpoint-humidity consistency, pressure-altitude sanity, lag features)
   -> TEMPORAL ANALYSIS (EWMA + rolling MAD; persistence counter)
   -> MULTIVARIATE ANALYSIS (Mahalanobis distance across T/P/H; correlation check)
   -> ML MODEL (Isolation Forest on engineered feature vector)
   -> ANOMALY FUSION (weighted combination: rule flags + IF score + Mahalanobis + rate-of-change)
   -> ROOT-CAUSE CLASSIFICATION (rule-based decision tree over which sub-signals fired)
   -> CONFIDENCE (0-100, weighted sum of sub-signal agreement, see ML_APPROACH)
   -> SEVERITY (LOW/MEDIUM/HIGH/CRITICAL from confidence x deviation magnitude)
   -> SENSOR HEALTH (rolling per-station score from anomaly rate/severity/persistence/missingness)
   -> EXPLANATION (SHAP TreeExplainer on Isolation Forest + templated natural-language reason)
   -> OPTIONAL CORRECTION (rolling median / short-horizon regression, clearly labeled "estimated")
   -> DASHBOARD / ALERT (Streamlit, live chart + alert feed + health panel)
```
**Why this and not deep learning**: 3 variables, no large labeled dataset, 3-day timeline, and the rubric
rewards explainability + real-time + energy efficiency — all favor a lightweight, interpretable hybrid over
LSTM/autoencoder. See ML_APPROACH section of research for full comparison table.

## 5. REAL WEATHER vs SENSOR FAULT — CORE DECISION RULE
Recommendation: **Neighboring-station data = OPTIONAL/DEMO-ONLY feature (Option C)**, never a required input,
because the PS explicitly restricts inputs to T/P/H for one station. Instead, distinguish using signals
derivable from a single station's T/P/H history:
1. **Persistence** — genuine weather events evolve over multiple consecutive readings; sensor glitches are
   often single-point or perfectly flat/frozen.
2. **Multivariate physical coherence** — real heatwaves raise T while relative humidity drops in a physically
   consistent way (Clausius–Clapeyron relationship) and pressure changes gradually; sensor faults usually break
   this coupling (e.g., T spikes 15°C in one reading while P and H stay static).
3. **Rate-of-change bounds** — real T/P/H changes are bounded by known meteorological rates per interval;
   faults often exceed physically possible rates instantaneously.
4. **Model agreement** — if only 1 of 3 sub-detectors fires, weight toward "possible real event"; if all fire
   simultaneously with a physically incoherent pattern, weight toward "sensor fault."
This yields root-cause label `POSSIBLE_REAL_WEATHER_EVENT` as a distinct, defensible output — this IS the
"self-aware" grand-challenge angle the PS's Grand Challenge line is asking for.

## 6. ANOMALY CLASSES (FROZEN LIST)
TEMPERATURE_SPIKE, TEMPERATURE_DROP, TEMPERATURE_DRIFT, PRESSURE_SPIKE, PRESSURE_DRIFT, HUMIDITY_SPIKE,
HUMIDITY_FROZEN (generalized to SENSOR_FROZEN/STUCK_AT for any variable), SENSOR_BIAS, MISSING_DATA /
COMMUNICATION_FAILURE, MULTIVARIATE_INCONSISTENCY, POSSIBLE_REAL_WEATHER_EVENT.

## 7. TECH STACK (FREE, VERIFIED SEPT 2026)
Python 3.13.5, pandas, numpy, scikit-learn (IsolationForest), shap, plotly/matplotlib, Streamlit (UI + local
real-time loop via `st.rerun`/session_state), SQLite (local storage), GitHub (source of truth), VS Code,
Streamlit Community Cloud (deployment, free), Claude (free tier, ~15-40 msgs/5hr window, use sparingly, batch
prompts), Google Antigravity (free tier now only ~20 req/day — use for narrow, well-scoped tasks only, not
open-ended exploration), Google Stitch (free, 350 standard + 200 experimental generations/month — used ONLY for
generating the dashboard's visual design/HTML-Tailwind, then hand-translated into Streamlit by us/Claude).

## 8. KEY DESIGN DECISIONS (append new ones with date, never delete old ones — strike through if reversed)
- [2026-09-07] Chose rules+statistics+IsolationForest+SHAP hybrid over any deep learning model. REASON: dataset
  size, timeline, explainability weight, energy-efficiency weight.
- [2026-09-07] Chose Streamlit over React/Next.js for UI. REASON: beginner team, 1 language (Python), fastest
  free deployment path, sufficient for 5% UI weight.
- [2026-09-07] Chose CSV-replay + Python generator/queue for "real-time" simulation instead of Kafka/MQTT/Redis.
  REASON: massive infra overkill for a 3-day beginner build; a `time.sleep()` based replayer with a queue
  achieves the same demo effect at zero infra risk.
- [2026-09-07] Spatial/neighbor-station comparison = optional demo enrichment only, core detector never depends
  on it (see Section 5).
  - [2026-09-07] Chose Python 3.13.5 instead of Python 3.11.
  REASON: Python 3.13.5 is already installed locally; avoid unnecessary
  environment changes. Package compatibility will be verified during setup.

## 9. HOW EACH AI SHOULD USE THIS FILE
- **Claude**: Read fully before generating any code. Never propose a different pipeline than Section 4 unless
  asked. Ask before large refactors. Update Section 8/DECISIONS and TODO_STATUS.md after each milestone.
- **Antigravity**: Treat this as the spec contract. Inspect existing repo files before writing. Do the smallest
  diff that satisfies the requested PROMPT (see PROMPTS.md). Never touch files unrelated to the current task.
- **New AI account/session**: paste this file + TODO_STATUS.md + ML_APPROACH section, say "continue from here,"
  and it has full continuity — no project should ever restart from zero.

## 10. FILES IN THIS REPO (consolidated set — we deliberately merged the requested 17 files into 5 to reduce
maintenance overhead, since a 2-person 3-day team cannot keep 17 docs in sync)
- `PROJECT_CONTEXT.md` — this file (requirements + architecture + tech stack + dataset + anomaly types + decisions)
- `TODO_STATUS.md` — TODO / current status / known issues / changelog (single running log)
- `PROMPTS.md` — Stitch prompt + Antigravity prompt sequence + Claude master prompt + coding-session protocol
- `IMPLEMENTATION_SCHEDULE.md` — hour-by-hour Sep 7–10 plan + staged coding order + test plan
- `DEMO_AND_JUDGES.md` — demo script + presentation slides + 50 judge Q&A + risk/fallback plan
