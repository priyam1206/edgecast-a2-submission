EDGECAST faculty submission package

This package contains the actual simulation/controller source used by the live demo.
It includes the Nguyen et al. comparator, the stabilized cache-aware EDGECAST policy,
tests, and the processed BurstGPT/FineServe traces used by the interactive demo.

Quick verification:
  python -m pip install -r requirements.txt
  python -m pytest -q

The Colab notebook downloads and executes this package directly.
