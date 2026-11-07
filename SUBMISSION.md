# EDGECAST — Team 29 A2 Submission

Faculty submission package for the EDGECAST research project.

## Live demo
https://edgecast-demo.onrender.com

## Google Colab
Open the notebook:
https://colab.research.google.com/github/priyam1206/edgecast-a2-submission/blob/main/Team_29_A2_Slot_EDGECAST_Colab.ipynb

## Reproducibility
- Actual simulator/controller source is included.
- Real processed FineServe and BurstGPT traces are included.
- Run tests with: python -m pytest -q
- FineServe benchmark: python scripts/benchmark_fineserve.py --trace data/preprocessed/FineServe_lite_60s.csv --windows 30
- BurstGPT benchmark: python scripts/benchmark_burstgpt.py --trace data/preprocessed/BurstGPT_1_60s.csv --windows 30

The reduced Stage-2 comparator preserves the small-demo objective/feasible allocation behavior, while exact enumeration is used rather than reproducing the base paper's Benders implementation line-for-line.
