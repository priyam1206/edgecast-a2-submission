# ChatGPT project log

## 2026-11-06 — Full faculty demo deployment preparation
- Confirmed the existing EDGECAST project is a real simulator rather than a precomputed dashboard.
- Verified `python -m pytest -q`: 17/17 tests passed before deployment changes.
- Preserved the existing Nguyen et al. comparator and stabilized cache-aware EDGECAST controller.
- Copied the real preprocessed workload traces into versioned deployment assets:
  - `data/preprocessed/BurstGPT_1_60s.csv`
  - `data/preprocessed/FineServe_lite_60s.csv`
- Updated `app.py` to load the real deployment traces from `data/preprocessed` instead of ignored `.cache` paths.
- Added `render.yaml` for a persistent public Streamlit web service on Render.
- Added `.python-version` = 3.11.9 for deployment reproducibility.
- Deployment policy: the website executes the actual simulator/controller at run time. Do not replace metrics with hard-coded or precomputed result cards.

- Added a `Live Run Verification & Export` panel to the Streamlit demo. It exposes the current configuration and live-generated summaries and lets faculty download the current slot trace CSV and run manifest JSON; metrics remain computed by the real server-side simulation.

## 2026-11-07 — Final faculty form package
- Render service is live at https://edgecast-demo.onrender.com and tracks the private development repo main branch.
- Created public faculty-only submission repository: https://github.com/priyam1206/edgecast-a2-submission
- Created one-click Colab notebook: https://colab.research.google.com/github/priyam1206/edgecast-a2-submission/blob/main/Team_29_A2_Slot_EDGECAST_Colab.ipynb
- Colab clones the public submission repo, installs requirements, runs the 17-test suite, then executes real FineServe/BurstGPT benchmark scripts against the shipped processed traces.
- Smoke-tested the submission ZIP independently: 17/17 tests passed and a real FineServe benchmark window completed successfully.
- Submission code ZIP is approximately 0.62 MB, under the form's 10 MB limit.
- The public submission repo contains only faculty-safe code/data/artifacts; the private development repo remains private.
