# EDGECAST — Team 29 A2 Submission

This public repository is the reproducibility package for the EDGECAST project submission.

## Contents
- Actual EDGECAST simulator/controller source
- Nguyen et al. comparator
- Stabilized cache-aware EDGECAST controller
- Real processed FineServe and BurstGPT traces used by the demo
- Automated tests
- Colab notebook: `Team_29_A2_Slot_EDGECAST.ipynb`

## Run locally

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Live demo
https://edgecast-demo.onrender.com

The demo and notebook execute the simulation at runtime; result metrics are not hard-coded.
