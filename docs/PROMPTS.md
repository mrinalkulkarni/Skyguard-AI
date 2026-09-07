# PROMPTS.md — Stitch / Antigravity / Claude prompts + coding protocol

## A. STITCH PROMPT (paste into stitch.withgoogle.com)
```
Design a professional meteorological monitoring dashboard called "SkyGuard AI" for a national weather agency
control room, not a generic student project. Style: scientific, high-trust, dark-mode-friendly command-center
aesthetic (like an aviation/disaster-management ops screen), navy/slate background, high-contrast alert colors
(green=healthy, amber=warning, red=critical), clean sans-serif typography, generous spacing, no playful icons.

Screen 1 — Main Dashboard:
- Top bar: station name/ID, live clock, connection status indicator (green dot = streaming).
- Three large live gauges/cards: Temperature (°C), Pressure (hPa), Humidity (%), each with current value,
  a small sparkline of last 30 readings, and a colored border if currently anomalous.
- Center: one large real-time multi-line chart (last N minutes) with anomaly points highlighted as red markers.
- Right sidebar "Sensor Health" panel: 0-100 health score per sensor with a horizontal bar and status label
  (Healthy/Good/Degraded/Poor/Critical).
- Bottom panel "Anomaly Alert Feed": scrollable list, each row shows timestamp, root-cause label, severity badge
  (LOW/MEDIUM/HIGH/CRITICAL), confidence %, and a one-line plain-English explanation.
- Clicking an alert expands: original value vs AI-estimated corrected value, and a short SHAP-style feature
  contribution bar chart (which variable contributed most to the anomaly score).

Screen 2 (optional) — Anomaly History/Stats: timeline heatmap of anomalies by type over the last 24 hours, and
counts by root-cause category as a horizontal bar chart.

Export target: clean HTML/CSS with Tailwind classes so it can be manually adapted into a Streamlit layout
(component structure, spacing, and color tokens matter more than exact interactivity).
```
**How to use the output**: Stitch will not produce working Streamlit code. Use it only for: color palette,
layout hierarchy, spacing, and component grouping. Manually re-create the same visual structure using Streamlit
`st.columns`, `st.metric`, `st.plotly_chart`, `st.dataframe`, and custom CSS injected via `st.markdown` with
`unsafe_allow_html=True`. Do not try to import Stitch's Tailwind HTML directly into Streamlit.

---

## B. ANTIGRAVITY PROMPT SEQUENCE
Preface every prompt below with this block (Antigravity free tier is rate-limited to ~20 req/day — batch your
asks, do not go back-and-forth conversationally):
```
Read PROJECT_CONTEXT.md and TODO_STATUS.md completely before doing anything. Inspect the existing repository
structure and current file contents before writing new code. Do NOT overwrite or restructure any working file
unless this prompt explicitly asks for it. Implement ONLY the feature below. After implementing, run/describe a
test proving it works, then update TODO_STATUS.md (mark task done, note remaining issues) and report exactly
what changed and what remains.
```

**PROMPT 1 — Initialize project**: Create folder structure (see IMPLEMENTATION_SCHEDULE.md §Folder Structure),
`requirements.txt` (pandas, numpy, scikit-learn, shap, plotly, streamlit), `.gitignore` (venv, __pycache__,
.streamlit/secrets.toml), `README.md` skeleton, empty module files with docstring headers only.

**PROMPT 2 — Build simulator**: Implement `simulator.py`: load a CSV of historical T/P/H, replay rows one at a
time with configurable delay (2-5s), inject anomalies from a config list (type, start index, duration,
magnitude) using the mathematical injection formulas in ML_APPROACH research (spike/drift/frozen/noise/bias/
missing). Output both clean and injected streams with ground-truth labels for evaluation.

**PROMPT 3 — Build preprocessing**: Implement `preprocessing.py`: handle missing values (forward-fill flag,
not silent), compute rolling mean/std/MAD (windows 5/15/30), rate-of-change, dewpoint estimate from T+RH
(Magnus formula), lag features.

**PROMPT 4 — Build anomaly detection core**: Implement `detector.py`: rule-based physical bounds check,
rolling z-score/MAD check, Mahalanobis distance across 3 variables, IsolationForest on the engineered feature
matrix (contamination tuned on validation split), and a fusion function combining all signals into one
anomaly score 0-1.

**PROMPT 5 — Build root-cause classification**: Implement `rootcause.py`: decision-tree/rule logic mapping
which sub-signals fired + which variable(s) deviated into one of the frozen anomaly classes (Section 6 of
PROJECT_CONTEXT.md).

**PROMPT 6 — Build confidence/severity**: Implement `scoring.py`: confidence = weighted combination of
(rule violation count, IF anomaly score, z-score magnitude, Mahalanobis distance, persistence count, sub-model
agreement), scaled to 0-100; severity thresholds from confidence x deviation magnitude, mapped to
LOW/MEDIUM/HIGH/CRITICAL.

**PROMPT 7 — Build sensor health**: Implement `health.py`: per-station-per-sensor rolling score (0-100) from
recent anomaly frequency, severity-weighted penalty, missing-data rate, drift magnitude, with exponential decay
so old anomalies matter less; map to Healthy/Good/Degraded/Poor/Critical bands.

**PROMPT 8 — Build explainability**: Implement `explain.py`: SHAP TreeExplainer on the fitted IsolationForest,
extract per-feature SHAP contributions for a flagged point, and generate a natural-language template sentence
(pattern in PROJECT_CONTEXT.md/ML_APPROACH). Include a rule-based fallback explanation if SHAP fails/unavailable.

**PROMPT 9 — Build dashboard**: Implement `app.py` (Streamlit) following the Stitch layout: live metrics,
chart, alert feed, sensor health panel, correction display. Use `st.session_state` + a background-thread-free
polling loop (`st.rerun` with `time.sleep`) reading from the simulator's queue/SQLite table.

**PROMPT 10 — Integrate everything**: Wire simulator -> preprocessing -> detector -> rootcause -> scoring ->
health -> explain -> app.py into one working pipeline (`pipeline.py` orchestrator). Confirm end-to-end run on
a 500-row sample.

**PROMPT 11 — Testing**: Implement `test_pipeline.py` running the 13 scenarios from IMPLEMENTATION_SCHEDULE.md
§Testing; compute precision/recall/F1/FPR against injected ground truth; save results to `results/metrics.csv`.

**PROMPT 12 — Deployment**: Prepare for Streamlit Community Cloud: confirm `requirements.txt` pinned versions,
no local file-path assumptions, push to GitHub, connect repo, verify public URL loads.

**PROMPT 13 — Final bug fixing**: Given the error log / failing test provided by the user, find root cause with
minimal diff, fix only that, re-run the full test suite, confirm no regression, update KNOWN_ISSUES section.

---

## C. CLAUDE MASTER PROMPT (paste as first message in a new Claude conversation)
```
You are acting as a Senior AI/ML Engineer + Python Developer + Data Scientist + Full-Stack Engineer +
Hackathon Mentor + Code Reviewer for a 2-person beginner student team building "SkyGuard AI" for Smart India
Hackathon problem SIH26073. We have ~3 productive days left. Reliability beats sophistication.

Ground rules you must follow for the entire session:
1. I will paste PROJECT_CONTEXT.md and TODO_STATUS.md content below. Read it fully before responding.
   Never propose a different architecture than what is frozen there unless I explicitly ask you to reconsider,
   and if you do, flag it clearly as a proposed change, not a silent substitution.
2. We are beginners. Explain every command before asking us to run it. Tell us exactly which file to open,
   exactly where to paste code, and what output to expect. If we report an error, diagnose using the exact
   error text before proposing a fix.
3. Give code in small, testable chunks — one function or one file at a time, not the whole project at once.
   After each chunk, tell us how to test it and what a correct result looks like.
4. Never delete or silently rewrite working code. If a change is risky, tell us to git commit first.
5. Only use free, open-source Python libraries already listed in TECH_STACK (pandas, numpy, scikit-learn, shap,
   plotly, streamlit). Do not introduce new dependencies without explaining why and confirming with us.
6. Never invent APIs, parameters, or library behavior you are not certain of. If unsure, say so and suggest how
   we can verify (e.g., "check the scikit-learn IsolationForest docs for this parameter").
7. Prioritize the official rubric: Innovation 25%, Accuracy 20%, Real-Time 15%, Explainability 10%,
   Scalability 10%, Deployability 10%, UI 5%, Energy 5%. When in doubt about what to spend time on, optimize
   for this order.
8. At the end of each milestone, tell me exactly what to update in TODO_STATUS.md, and remind me to commit to
   Git with a clear message.
9. Stop and wait for my test result before moving to the next component when a component could fail silently.

Here is PROJECT_CONTEXT.md:
<PASTE FULL FILE>

Here is TODO_STATUS.md:
<PASTE FULL FILE>

My first task: <state the specific next task from TODO_STATUS.md>. Begin with STEP 1 of the coding-session
protocol below.
```

### CLAUDE CODING SESSION PROTOCOL (use for every task, paste once, refer back to it)
```
STEP 1 — Read PROJECT_CONTEXT.md and TODO_STATUS.md fully (or the pasted content).
STEP 2 — Ask to see or infer current repo/file state before writing code; do not assume files exist.
STEP 3 — State in one sentence what is currently working and what is not.
STEP 4 — Restate the exact task you are about to do, in your own words, and confirm it matches TODO_STATUS.md.
STEP 5 — Explain your implementation plan in plain language BEFORE writing code (2-5 bullet points).
STEP 6 — Make the smallest change that satisfies the task — one file/function at a time.
STEP 7 — Give an exact test (input + expected output + the command to run it).
STEP 8 — Explicitly check: could this change break any previously-working feature? State yes/no and why.
STEP 9 — Tell the user exactly what to add/change in TODO_STATUS.md (and DECISIONS section if architecture
          changed).
STEP 10 — Tell the user the exact next action (test it and report back / run this command / move to next task).
```

---

## D. DEMO-CRITICAL EXPLANATION TEMPLATE (for explain.py)
```
"{VARIABLE} was flagged because it {changed_direction} {delta} {unit} within one interval, which is
{n_std}σ outside the recent rolling pattern (window={window}). {other_vars} did {consistency_phrase},
{increasing/decreasing} the likelihood of a {root_cause_label} rather than a genuine weather event.
Model agreement: {n_of_3} of 3 sub-detectors flagged this point. Confidence: {confidence}%."
```
