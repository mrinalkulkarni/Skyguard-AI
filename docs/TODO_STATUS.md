# TODO_STATUS.md — running log (append-only, newest on top)

## CURRENT STATUS
Phase: Day 0 (research complete). Nothing coded yet. Next action: environment setup + Prompt 1 (Antigravity)
or Section "Coding Order" in IMPLEMENTATION_SCHEDULE.md.

## TODO (P0 first)
- [x] P0: Repo scaffold + venv + requirements.txt
- [ ] P0: `simulator.py` — CSV replay / anomaly injector
- [ ] P0: `preprocessing.py` — cleaning, resampling, feature engineering
- [ ] P0: `detector.py` — rules + IsolationForest + fusion
- [ ] P0: `rootcause.py` — classification
- [ ] P0: `scoring.py` — confidence + severity + sensor health
- [ ] P0: `explain.py` — SHAP + templated text
- [ ] P0: `app.py` — Streamlit dashboard
- [ ] P1: correction/imputation module
- [ ] P1: evaluation script (precision/recall/F1/FPR on injected test set)
- [ ] P2: HF Spaces / Streamlit Cloud deployment
- [ ] P2: ESP32/edge-efficiency talking points slide (no hardware required)
- [ ] P3 (DO NOT BUILD): real MQTT broker, Kafka, microservices, mobile app, user auth system, multi-tenant DB

## KNOWN ISSUES
(none yet — update as encountered, include: symptom, cause, fix, file changed)

## CHANGELOG
- 2026-09-07 — Research complete, architecture frozen, PROJECT_CONTEXT.md created.
- 2026-09-08 — Scaffold created, all core packages installed under Python 3.13.15, requirements.txt frozen.
