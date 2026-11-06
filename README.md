# EdgeCast

**Cache-aware warm model reallocation for Generative AI services in heterogeneous edge networks.**

EdgeCast extends the two-stage GAI edge-service provisioning framework of Nguyen et al. (IEEE TNSM, 2026) by adding **persistent inactive-checkpoint state, constrained background staging, and cache-aware warm/cold reallocation** to the online stage.

The current research question is whether these additions improve the paper's dynamic model-reallocation behavior when model deployment incurs non-negligible service interruption and multiple model transfers share a constrained deployment path.

## Base-paper comparator

The dashboard no longer uses a simple queue-threshold heuristic as the primary paper comparator.

For the model-aggregated demonstrator:

- Stage 1 gives both methods the **same reduced optimum initial active-model allocation**.
- The assigned-paper side uses the paper's Stage-2 queue-service logic in aggregate form.
- Its reallocation decision optimizes the paper's system-level objective **Tp - Tc - sigma** over feasible active-model allocations.
- Base-paper reallocations use cold deployment cost and do not retain inactive model checkpoints.
- For the small interactive problem, the P3 optimum is obtained by exact enumeration rather than by reproducing the paper's Benders solver implementation. The reduced objective and feasible allocation set are preserved, but this is not a line-for-line copy of the authors' solver code.

## EdgeCast difference

EdgeCast evaluates the **same active-allocation objective**, but deployment cost is cache-aware and the online controller now includes an explicit anti-thrashing stability gate:

- target checkpoint already cached -> warm cutover cost;
- target checkpoint absent -> cold deployment cost.

Background staging is a separate action. If the current cache state does not justify a reallocation, EdgeCast may reserve the constrained shared backhaul to stage a checkpoint when doing so would make the same **Tp - Tc - sigma** objective positive.

Staging does not change the active model, so an edge can stage another model while continuing to serve its current model.

## Workloads

The app supports three workload classes:

- **Synthetic stress tests** for controlled mechanism demonstrations:
  - gradual video ramp;
  - sudden spike.
- **BurstGPT** for real Azure temporal workload evaluation:
  - request-count mode;
  - token-weighted work mode.
  - The current public BurstGPT repository exposes BurstGPT_1.csv with two model labels: ChatGPT and GPT-4.
- **FineServe** for real multi-class replay evaluation:
  - Dense 10-30B;
  - Dense 30-100B;
  - MoE >100B in the default Lite preset;
  - optional Dense <10B in the much larger Full preset.
  - FineServe metadata covers 57 models overall, but the public replay CSVs expose architecture/scale classes rather than individual model IDs.

Prepare BurstGPT once:

~~~bash
python -m pip install -r requirements.txt
python scripts/prepare_burstgpt.py
python -m streamlit run app.py
~~~

Prepare the default FineServe Lite trace (~116 MB of replay CSVs before preprocessing):

~~~bash
python scripts/prepare_fineserve.py --preset lite
~~~

The optional full four-class preset is substantially larger because Dense_lt10B is a large Git LFS object:

~~~bash
python scripts/prepare_fineserve.py --preset full
~~~

## Fullscreen demonstration

Configure the experiment in the retractable sidebar and press:

**Run + log fullscreen demo**

The app then:

1. freezes the current configuration;
2. runs both methods on the same data and initial allocation;
3. logs the experiment;
4. opens a clean side-by-side synchronized animation.

The fullscreen view uses one shared play/pause control, timeline, speed selector, and slot index.

## Experiment logging

Each explicit demo run is stored under:

~~~text
.cache/edgecast/runs/<RUN_ID>/
~~~

with:

~~~text
manifest.json
slot_trace.csv
events.json
~~~

These files record configuration, raw summaries, slot-level queues/arrivals/decisions, node state, staging actions, and reallocations.

A single dashboard run is **not** treated as a paper result.

## Real-trace benchmark

Run many BurstGPT windows with the current paper comparator and stabilized EdgeCast policy:

~~~bash
python scripts/benchmark_burstgpt.py --windows 30
~~~

Token-weighted mode:

~~~bash
python scripts/benchmark_burstgpt.py --windows 30 --mode token_work
~~~

The benchmark reports raw metrics for both policies and deltas. Positive latency/backlog deltas mean EdgeCast was lower for that metric; negative values mean it was higher.


FineServe is the stronger multi-class validation path. It compares three policies in every window: the assigned-paper comparator, the earlier cache-aware controller with no persistence gate, and stabilized EdgeCast:

~~~bash
python scripts/benchmark_fineserve.py --windows 30
~~~

Use the reallocation-persistence argument with value 0 to reproduce the earlier warm-switch-thrashing controller.

## Current methodological boundary

The demonstrator is model-aggregated. It does not contain the individual user-edge communication variables required for a literal request-by-request implementation of every P1/P2 equation in the published paper.

Therefore:

- the Stage-2 queue service is the aggregate equivalent of P2b;
- the reduced P3 allocation objective is optimized exactly for the demo state;
- exact enumeration is used instead of Benders for the small interactive problem.

Claims in the paper must preserve this distinction.

## Status

Current implementation includes:

- heterogeneous edge capacities;
- reduced common Stage-1 initial allocation;
- assigned-paper Tp - Tc - sigma Stage-2 comparator;
- cache-aware EdgeCast P3 decision;
- persistent checkpoint cache state;
- background staging;
- constrained concurrent backhaul transfers;
- warm and cold cutovers;
- minimum-persistence anti-thrashing control for active reallocations;
- synthetic, BurstGPT, and FineServe real-trace workloads;
- synchronized fullscreen visualization;
- reproducible run logging;
- automated tests;
- multi-window BurstGPT benchmark.

Final performance claims are intentionally deferred until the evaluation protocol is frozen and the new benchmark is rerun across enough windows and parameter settings.

## References

- C. T. Nguyen, D. T. Hoang, D. N. Nguyen, and H.-A. Pham, “Generative AI Service Provision in Heterogeneous Edge Networks: A Dynamic Two-Stage Optimization Approach,” *IEEE Transactions on Network and Service Management*, vol. 23, 2026. DOI: 10.1109/TNSM.2026.3730014.
- Y. Wang et al., “BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems,” *KDD*, 2025. DOI: 10.1145/3711896.3737413.


## Public faculty demo deployment

The repository includes a Render deployment definition in `render.yaml`.

The public demo executes the actual Python simulator and both controllers on the server. It does not display hard-coded benchmark outputs. The versioned deployment bundle includes the processed real-trace assets used by the interactive BurstGPT and FineServe modes under:

~~~text
data/preprocessed/BurstGPT_1_60s.csv
data/preprocessed/FineServe_lite_60s.csv
~~~

For each interactive run, the **Live Run Verification & Export** panel exposes the runtime configuration, both live-generated summaries, and downloadable JSON/CSV run artifacts.

Render start command:

~~~bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
~~~
