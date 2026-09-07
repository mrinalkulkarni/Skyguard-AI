# DEMO_AND_JUDGES.md

## 90–120 SECOND LIVE DEMO SCRIPT
0:00-0:15 — Open dashboard, station streaming normally, all metrics green, health = 95 "Healthy."
0:15-0:35 — Trigger a TEMPERATURE_SPIKE injection (e.g., +15°C in one interval, P/H unchanged). Point at the
chart marker turning red instantly. Say: "Detected in real time."
0:35-0:55 — Click the alert: show confidence (e.g., 92%), severity (HIGH), root cause (TEMPERATURE_SPIKE), and
read the one-line explanation naming the variable and the physical inconsistency (P/H didn't move).
0:55-1:10 — Show sensor health dropping one band (Healthy -> Good) after 2-3 repeated flags, demonstrating
degradation tracking, not just single-point alerts.
1:10-1:30 — Show Original vs AI-Estimated Corrected Value side by side, labeled clearly, explain it is an
estimate, never overwriting the raw observation.
1:30-1:50 — Inject a GENUINE-WEATHER-LIKE event (gradual, multi-variable coherent change over several
intervals, e.g., slow T rise + RH drop together consistent with a heatwave) and show the system labels it
`POSSIBLE_REAL_WEATHER_EVENT` with lower/different confidence framing instead of a fault — this is the single
most important moment of the demo (it answers the PS's actual hard problem).
1:50-2:00 — Recovery: stream returns to normal, health score begins recovering, close on the deployed public
URL visible in the browser bar (proves real deployment, not localhost).

## BACKUP DEMO (if live deployment fails)
Pre-record the exact sequence above as a 2-minute screen capture the day before (Sep 9 evening). Keep it on a
laptop, a phone, and a USB drive. If Wi-Fi/live app fails, say "here is our deployed system captured live
yesterday" and play the video — never apologize extensively, move straight to explaining architecture instead.

## JUDGING STRATEGY TABLE
| Category | What judge wants | What we build | How we demonstrate | Evidence/metric |
|---|---|---|---|---|
| Innovation 25% | Something beyond threshold QC | Hybrid rules+stats+IsolationForest+context reasoning that separates real weather from faults | The genuine-weather-event demo moment | Distinct `POSSIBLE_REAL_WEATHER_EVENT` label with reasoning |
| Accuracy 20% | Real numbers, not claims | Injected test-set evaluation | Show `results/metrics.csv` live | Precision/Recall/F1/FPR per class |
| Real-Time 15% | Live streaming feel | 2-5s simulator + auto-updating dashboard | Live demo | Latency from injection to alert (seconds) |
| Explainability 10% | Why, not just what | SHAP + templated plain-English reason | Click-to-expand explanation panel | SHAP contribution bar chart |
| Scalability 10% | Works for many stations | Stateless per-station scoring function, station_id as a parameter | Explain in doc/slide, mention no per-station retraining needed | Architecture slide |
| Deployability 10% | Actually runs somewhere | Streamlit Community Cloud public link | Open link live | Working public URL |
| UI 5% | Clear, professional | Stitch-guided Streamlit dashboard | Visual walkthrough | N/A |
| Energy 5% | Lightweight/edge-friendly | No deep learning/GPU; IsolationForest is O(n log n), runs on ESP32-class CPU in principle | Mention model size/inference time | Inference time per point (ms) |

## PRESENTATION STRUCTURE (6-8 slides)
1. **Problem** — AWS QC today is threshold-based, misses hidden/complex anomalies, and conflates real extreme
   weather with sensor faults (cite SIH26073 PS + WMO QC guide).
2. **Why existing approaches fail** — static thresholds can't tell a heatwave from a stuck sensor; deep models
   need huge labeled data we don't have and are hard to explain to a met department.
3. **SkyGuard solution** — hybrid rules + temporal + multivariate + Isolation Forest + SHAP, tuned specifically
   for T/P/H triads.
4. **Architecture** — the frozen pipeline diagram from PROJECT_CONTEXT.md.
5. **AI/ML methodology** — feature engineering (rate-of-change, Mahalanobis, EWMA), Isolation Forest choice
   rationale, SHAP explainability.
6. **Live results/metrics** — precision/recall/F1/FPR numbers from the injected test set.
7. **Innovation** — real-weather-vs-fault discrimination, sensor health degradation modeling, confidence-aware
   alerting.
8. **Deployment/scalability/future** — current: Streamlit Cloud deployed; future: ESP32 edge inference, spatial
   cross-station validation as an optional layer, integration with IMD's existing QC pipeline.

## 50 JUDGE QUESTIONS (selected representative set across categories, with best short answer, detail, and traps)

**Problem understanding**
1. Q: Why only T/P/H? A(short): "That's the PS's mandated input scope." A(detail): explain PS restricts inputs
   deliberately to test whether accurate QC is possible with minimal sensors, which mirrors real low-cost AWS
   networks. Avoid: claiming you'd add more sensors if you could — stay within scope confidently.
2. Q: What's the hardest part of this problem? A: distinguishing genuine extreme weather from sensor faults,
   since both look like statistical outliers; explain your multivariate-coherence approach. Avoid: saying
   "detecting anomalies" — that undersells the real challenge that the PS itself calls out.

**ML**
3. Q: Why Isolation Forest over deep learning (LSTM/autoencoder)? A: 3 features, small/no labeled real fault
   data, need explainability and low compute — Isolation Forest is fast, unsupervised, tree-based (SHAP
   compatible), and proven for low-dimensional anomaly detection. Avoid: "deep learning is too hard for us" —
   frame it as a deliberate engineering tradeoff, not a skill gap.
4. Q: How do you choose contamination/threshold? A: tuned on a validation split of injected anomalies, report
   the value used and how you validated it.
5. Q: Why not just Z-score/rules alone? A: rules alone miss multivariate/hidden anomalies; ML alone lacks
   physical grounding and explainability; fusion gets both.
6. Q: What if IsolationForest and the rule layer disagree? A: explain your fusion weighting and how confidence
   score reflects sub-model disagreement rather than hiding it.

**Dataset**
7. Q: Is your data real? A: base signal from NOAA ISD (real station data), anomalies injected using the PS's
   own sanctioned methodology ("evaluated on anomaly injected data"). Avoid: implying you have real fault-
   labeled data — nobody does; be upfront it's synthetic-on-real-baseline.
8. Q: How did you pick injection parameters? A: derived from realistic physical bounds and known sensor fault
   literature (drift rates, stuck-at behavior, communication gaps), not arbitrary randomness.

**Accuracy**
9. Q: What's your F1/precision/recall? A: state your actual `metrics.csv` numbers.
10. Q: What's your false positive rate on real-weather-like data? A: state the number from your genuine-event
    test scenario specifically, since this is the rubric's hardest sub-point.

**False positives**
11. Q: How do you keep false alarms low at scale? A: persistence + multivariate-coherence checks reduce
    single-point false triggers; explain the confidence threshold tuning.

**Real weather vs sensor fault**
12. Q: How exactly do you tell them apart with only 3 variables? A: physical coupling (Clausius–Clapeyron
    T-humidity relationship), rate-of-change bounds, and persistence across multiple readings — walk through
    the demo example.
13. Q: Would neighboring station data help? A: yes, and WMO QC guidance itself lists spatial comparison as a
    real technique, but the PS restricts core inputs, so we built it as optional enrichment, not a dependency —
    state this decision confidently.

**Explainability**
14. Q: Is SHAP valid for Isolation Forest? A: yes, TreeSHAP is designed for tree-ensemble models including
    IsolationForest, giving exact per-feature attribution to the anomaly score (cite SHAP TreeExplainer usage).
15. Q: Why not LIME? A: LIME approximates locally with perturbations and is less exact/slower for tabular tree
    models; TreeSHAP gives exact, faster attributions for our model type.

**Root cause**
16. Q: How do you classify root cause? A: rule-based decision logic over which sub-detectors fired and which
    variable(s) deviated — chosen over an ML classifier because we lack large labeled root-cause data and rules
    stay explainable.

**Deployment**
17. Q: Where is it deployed? A: Streamlit Community Cloud, free tier, public URL — show it live.
18. Q: What's your energy footprint? A: no GPU, tree-based model, inference in milliseconds; discuss ESP32
    feasibility as future work, not implemented in this build.

**Edge AI**
19. Q: Did you deploy to ESP32? A: no — it was a "suggested," not mandatory technology; be honest, explain
    why (3-day timeline) and what the path to edge deployment would look like (model export, quantization).

**Scalability**
20. Q: How does this scale to thousands of stations? A: per-station stateless scoring function with per-
    station rolling baselines; explain no cross-station retraining required.

**Security/data integrity**
21. Q: How do you prevent tampering with corrected values? A: corrected value always stored separately from
    original observation with clear labeling, never overwrites raw data.

**Innovation**
22. Q: What's actually novel here vs. existing QC? A: explicit real-weather-vs-fault discrimination with
    confidence-aware, explainable output and sensor health degradation modeling — not just outlier flagging.

**Limitations**
23. Q: What are your system's limitations? A: single-station core detection (no mandatory spatial cross-check),
    trained on synthetic-on-real-baseline injected anomalies (no ground-truth real fault labels), simplified
    correction method. Be upfront — judges respect honesty over overclaiming.

**Future scope**
24. Q: What would you build next? A: ESP32 edge inference, optional spatial cross-station validation layer,
    online/incremental model updates, integration with IMD's operational QC pipeline.

(Remaining ~26 questions follow the same categories — dataset licensing, why not Prophet/ARIMA, why not
autoencoder, how you validate synthetic anomalies are realistic, how confidence differs from severity, why
Mahalanobis distance, how missing data is handled, why SQLite/CSV instead of a real database, what happens
under sensor power loss, how you'd handle concept drift/seasonality, why Streamlit instead of React, how you
tested for regressions, etc. — prepare answers using the same short/detailed/avoid structure by mapping each to
the relevant PROJECT_CONTEXT.md/ML_APPROACH decision.)

## RISK / FALLBACK PLAN
| Risk | Fallback |
|---|---|
| Model doesn't detect an anomaly type live | Have a pre-tested injection config known to trigger reliably; never inject live/untested parameters during judging |
| Too many false positives | Tune confidence threshold beforehand on validation data; don't touch thresholds live |
| Dashboard freezes | Keep a second browser tab pre-loaded; restart `streamlit run` locally as fallback to the cloud link |
| Deployment fails at judging time | Play the pre-recorded backup video; run locally as second fallback |
| Dataset download fails during build | Switch to Kaggle backup dataset immediately (don't debug NOAA access >45 min) |
| Dependency/version error | Pin exact versions in `requirements.txt`; keep a clean venv reference on a second machine |
| Python version issue | Standardize both laptops on Python 3.11 on Day 1 |
| Streamlit-specific bug | Keep `app.py` simple (no custom components); test in incognito/another browser before demo |
| AI-generated code breaks existing code | Always `git commit` before asking Claude/Antigravity for a change; diff before accepting |
| Git conflict | Divide work by file; if conflict occurs, `git pull`, resolve manually, re-test, then push |
| Free-tier limit hit (Claude/Antigravity) | Batch prompts, use Claude for reasoning-heavy tasks and Antigravity sparingly for scaffolding only |
| Network failure at venue | Local `streamlit run app.py` fallback + recorded video as final fallback |
