from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import inf
from typing import Dict, Iterable, Sequence

from .predictor import ShortHorizonPredictor


def _model_capacities(nodes, models) -> Dict[str, int]:
    return {
        m: sum(n.capacity for n in nodes if n.available and n.model == m)
        for m in models
    }


def _choose_donor(nodes, target_model, queue_sizes, config, slot, prefer_cached=False):
    counts = {
        m: sum(1 for n in nodes if n.model == m and n.available)
        for m in config.models
    }
    candidates = []
    for idx, node in enumerate(nodes):
        if not node.available or node.model == target_model:
            continue
        if slot - node.last_switch_slot < config.cooldown:
            continue
        source = node.model
        if counts[source] <= 1:
            continue
        pressure = queue_sizes[source] / max(1, node.capacity)
        cache_rank = 0 if (prefer_cached and target_model in node.cached_models) else 1
        candidates.append((cache_rank, pressure, queue_sizes[source], -node.capacity, idx))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][-1]


def optimize_initial_allocation(
    demand: Sequence[Dict[str, int]],
    models: Sequence[str],
    node_capacities: Sequence[int],
    *,
    model_sizes: Dict[str, float] | None = None,
    node_storage: Sequence[float] | None = None,
) -> tuple[str, ...]:
    """Reduced Stage-1 optimizer for the demonstrator.

    The IEEE paper's P1 jointly models request assignment, communication,
    computation, and initial model allocation. The demo workload is already
    aggregated by model and does not contain user-edge communication
    parameters, so the P1 objective collapses here to minimizing the expected
    aggregate workload clearance time under the one-active-model-per-edge and
    storage-feasibility constraints.

    This function is shared by every compared policy so no method receives a
    different initial allocation.
    """
    models = tuple(models)
    capacities = tuple(int(x) for x in node_capacities)
    if not models or not capacities:
        raise ValueError("models and node_capacities must be non-empty")

    work = {
        m: float(sum(max(0, int(slot.get(m, 0))) for slot in demand))
        for m in models
    }
    sizes = model_sizes or {m: 1.0 for m in models}
    storage = tuple(node_storage or (max(sizes.values()) for _ in capacities))
    if len(storage) != len(capacities):
        raise ValueError("node_storage must match node_capacities length")

    feasible_by_node = [
        tuple(m for m in models if float(sizes.get(m, 1.0)) <= float(storage[e]))
        for e in range(len(capacities))
    ]
    if any(not x for x in feasible_by_node):
        raise ValueError("At least one edge cannot host any configured model")

    best = None
    best_score = inf
    best_changes = inf

    # Deterministic reference used only for tie-breaking.
    reference = tuple(models[i % len(models)] for i in range(len(capacities)))

    for allocation in product(*feasible_by_node):
        by_model = {m: 0 for m in models}
        for e, m in enumerate(allocation):
            by_model[m] += capacities[e]

        score = 0.0
        feasible = True
        for m in models:
            if work[m] <= 0:
                continue
            if by_model[m] <= 0:
                feasible = False
                break
            score = max(score, work[m] / by_model[m])

        if not feasible:
            continue
        changes = sum(a != b for a, b in zip(allocation, reference))
        if score < best_score - 1e-12 or (
            abs(score - best_score) <= 1e-12 and changes < best_changes
        ):
            best = tuple(allocation)
            best_score = score
            best_changes = changes

    if best is None:
        raise ValueError("No feasible initial allocation exists for the configured storage")
    return best


@dataclass
class NguyenPaperPolicy:
    """Equation-faithful Stage-2 reallocation for the demo abstraction.

    The assigned paper's P3a/P3b maximizes

        Tp - Tc - sigma

    subject to one active model per edge, storage feasibility, model-specific
    aggregate inference capacity, and the maximum deployment delay sigma.

    For the small interactive demo we enumerate feasible z allocations and
    therefore obtain the exact optimum of this reduced P3 objective rather than
    reproducing the paper's Benders implementation. The workload simulator is
    model-aggregated, so the paper's P2b request-assignment step collapses to
    serving each model queue up to the total capacity of currently active
    compatible edges. This policy intentionally has no cache, prefetch, warm
    switch, threshold, or cooldown extension.
    """

    epsilon: float = 1e-9
    max_exact_nodes: int = 8
    name: str = "Nguyen Stage-2 model (P2b + P3 objective)"

    def __post_init__(self):
        self.last_decision = {}
        self._feasible_allocations = None
        self._signature = None

    def _candidate_allocations(self, nodes, config) -> Iterable[tuple[str, ...]]:
        if len(nodes) > self.max_exact_nodes:
            raise ValueError(
                f"Exact demo P3 enumeration is limited to {self.max_exact_nodes} edges; "
                "reduce the Edge nodes control for the faithful comparison."
            )

        sizes = config.model_sizes_map()
        storage = config.storages()
        signature = (
            tuple(config.models),
            len(nodes),
            tuple(float(x) for x in storage),
            tuple((m, float(sizes[m])) for m in config.models),
        )
        if signature != self._signature:
            feasible_by_node = [
                tuple(
                    m
                    for m in config.models
                    if float(sizes[m]) <= float(storage[e])
                )
                for e in range(len(nodes))
            ]
            if any(not x for x in feasible_by_node):
                raise ValueError("At least one edge has no storage-feasible model")
            self._feasible_allocations = tuple(product(*feasible_by_node))
            self._signature = signature
        return self._feasible_allocations

    @staticmethod
    def _clearance_time(allocation, nodes, queue_sizes, models) -> float:
        capacities = {m: 0.0 for m in models}
        for e, model in enumerate(allocation):
            capacities[model] += float(nodes[e].capacity)

        t = 0.0
        for m in models:
            q = float(queue_sizes[m])
            if q <= 0:
                continue
            cap = capacities[m]
            if cap <= 0:
                return inf
            t = max(t, q / cap)
        return t

    def decide(self, slot, nodes, queue_sizes, arrival_history, config):
        self.last_decision = {
            "slot": slot,
            "action": "hold",
            "method": "P2b aggregate-equivalent + exact P3 objective",
            "reason": "Current allocation is optimal for the paper's Tp - Tc - sigma objective.",
        }

        # The paper models nodes as unavailable while a deployment is in
        # progress. Do not start a second P3 transition during that interval.
        if any(n.switch_remaining > 0 for n in nodes):
            self.last_decision["reason"] = "Model deployment is in progress; affected edge capacity is unavailable."
            return None

        if not any(queue_sizes[m] > 0 for m in config.models):
            self.last_decision["reason"] = "All model queues are empty."
            return None

        current = tuple(n.model for n in nodes)
        tp = self._clearance_time(current, nodes, queue_sizes, config.models)
        if tp == inf:
            # A newly arriving model may have no current active copy. P3 must
            # then find a feasible allocation with finite Tc.
            tp_display = None
        else:
            tp_display = float(tp)

        best_allocation = current
        best_tc = tp
        best_sigma = 0.0
        best_benefit = 0.0
        best_changes = 0

        for candidate in self._candidate_allocations(nodes, config):
            tc = self._clearance_time(candidate, nodes, queue_sizes, config.models)
            if tc == inf:
                continue

            changed = [
                e for e, (old, new) in enumerate(zip(current, candidate))
                if old != new
            ]
            sigma = max(
                (
                    float(config.cold_switch_slots(candidate[e], e))
                    for e in changed
                ),
                default=0.0,
            )

            if tp == inf:
                benefit = inf if tc < inf else -inf
            else:
                benefit = tp - tc - sigma

            # P3 contains the current allocation as a feasible zero-benefit
            # candidate. Tie-break toward fewer changes, then lower Tc.
            if (
                benefit > best_benefit + self.epsilon
                or (
                    abs(benefit - best_benefit) <= self.epsilon
                    and len(changed) < best_changes
                )
                or (
                    abs(benefit - best_benefit) <= self.epsilon
                    and len(changed) == best_changes
                    and tc < best_tc - self.epsilon
                )
            ):
                best_allocation = tuple(candidate)
                best_tc = tc
                best_sigma = sigma
                best_benefit = benefit
                best_changes = len(changed)

        if best_allocation == current or (
            best_benefit != inf and best_benefit <= self.epsilon
        ):
            self.last_decision.update(
                {
                    "Tp": tp_display,
                    "Tc": None if best_tc == inf else float(best_tc),
                    "sigma": float(best_sigma),
                    "net_benefit": 0.0,
                    "target_allocation": list(current),
                }
            )
            return None

        changes = [
            {"node": e, "target": best_allocation[e]}
            for e in range(len(nodes))
            if current[e] != best_allocation[e]
        ]
        benefit_value = None if best_benefit == inf else float(best_benefit)
        self.last_decision = {
            "slot": slot,
            "action": "multi_switch",
            "method": "P2b aggregate-equivalent + exact P3 objective",
            "Tp": tp_display,
            "Tc": float(best_tc),
            "sigma": float(best_sigma),
            "net_benefit": benefit_value,
            "changes": len(changes),
            "target_allocation": list(best_allocation),
            "reason": "Positive Tp - Tc - sigma benefit under the paper's reallocation objective.",
        }
        return {
            "type": "multi_switch",
            "changes": changes,
            "force_cold": True,
            "retain_cache": False,
        }


@dataclass
class ReactivePolicy:
    """Legacy cold heuristic retained as an auxiliary baseline."""

    backlog_trigger: int = 18
    imbalance_trigger: int = 12
    benefit_margin: float = 2.0
    name: str = "Cold reactive heuristic"

    def __post_init__(self):
        self.last_decision = {}

    def decide(self, slot, nodes, queue_sizes, arrival_history, config):
        self.last_decision = {
            "slot": slot,
            "action": "hold",
            "reason": "Current queue imbalance is below the reactive switching gate.",
        }
        if any(n.switch_remaining > 0 for n in nodes):
            self.last_decision["reason"] = "A model deployment is already in progress."
            return None
        target = max(config.models, key=lambda m: queue_sizes[m])
        target_q = queue_sizes[target]
        source = min(config.models, key=lambda m: queue_sizes[m])
        if target_q < self.backlog_trigger or target_q - queue_sizes[source] < self.imbalance_trigger:
            return None
        donor_idx = _choose_donor(nodes, target, queue_sizes, config, slot)
        if donor_idx is None:
            self.last_decision["reason"] = "No safe donor edge is available."
            return None
        donor = nodes[donor_idx]
        net_benefit = target_q / max(1, donor.capacity) - config.switch_time
        if net_benefit <= self.benefit_margin:
            self.last_decision["reason"] = "Current-state benefit does not exceed cold deployment cost."
            return None
        self.last_decision = {
            "slot": slot,
            "action": "switch",
            "target": target,
            "queue": target_q,
            "net_benefit": float(net_benefit),
            "reason": "Observed queue pressure justifies a cold model reallocation.",
        }
        return {
            "type": "switch",
            "node": donor_idx,
            "target": target,
            "force_cold": True,
            "retain_cache": False,
        }


@dataclass
class CacheAwarePolicy:
    """EdgeCast: paper-style P3 reallocation with cache-aware deployment cost.

    The active-model allocation decision uses the same system-level objective
    as the assigned paper,

        Tp - Tc - sigma,

    but sigma depends on checkpoint availability at each edge:
      * cached target -> warm cutover cost
      * uncached target -> cold deployment cost

    Background staging is a separate action. When the current cache state does
    not justify a P3 reallocation, EdgeCast may use the constrained shared
    backhaul to stage one checkpoint if making that checkpoint local would turn
    the same P3 objective positive. Staging never changes the active model, so
    the "last active copy" restriction belongs to the later allocation
    decision, not to prefetch eligibility.
    """

    epsilon: float = 1e-9
    max_exact_nodes: int = 8
    min_reallocation_interval: int = 6
    name: str = "EdgeCast stable cache-aware P3 reallocation"

    def __post_init__(self):
        self.last_decision = {}
        self._feasible_allocations = None
        self._signature = None
        self._last_reallocation_slot = -10_000
        self.name = (
            "EdgeCast stabilized cache-aware P3 reallocation"
            if self.min_reallocation_interval > 0
            else "EdgeCast cache-aware P3 reallocation (no persistence)"
        )

    def _candidate_allocations(self, nodes, config) -> Iterable[tuple[str, ...]]:
        if len(nodes) > self.max_exact_nodes:
            raise ValueError(
                f"Exact demo P3 enumeration is limited to {self.max_exact_nodes} edges; "
                "reduce the Edge nodes control."
            )

        sizes = config.model_sizes_map()
        storage = config.storages()
        signature = (
            tuple(config.models),
            len(nodes),
            tuple(float(x) for x in storage),
            tuple((m, float(sizes[m])) for m in config.models),
        )
        if signature != self._signature:
            feasible_by_node = [
                tuple(
                    m
                    for m in config.models
                    if float(sizes[m]) <= float(storage[e])
                )
                for e in range(len(nodes))
            ]
            if any(not x for x in feasible_by_node):
                raise ValueError("At least one edge has no storage-feasible model")
            self._feasible_allocations = tuple(product(*feasible_by_node))
            self._signature = signature
        return self._feasible_allocations

    @staticmethod
    def _clearance_time(allocation, nodes, queue_sizes, models) -> float:
        return NguyenPaperPolicy._clearance_time(
            allocation,
            nodes,
            queue_sizes,
            models,
        )

    def _switch_cost(self, nodes, config, node_idx: int, target: str, extra_cache=None) -> float:
        cached = target in nodes[node_idx].cached_models
        if extra_cache == (node_idx, target):
            cached = True
        return float(
            config.warm_switch_time
            if cached
            else config.cold_switch_slots(target, node_idx)
        )

    def _best_p3(
        self,
        nodes,
        queue_sizes,
        config,
        *,
        extra_cache=None,
    ):
        current = tuple(n.model for n in nodes)
        tp = self._clearance_time(current, nodes, queue_sizes, config.models)

        best_allocation = current
        best_tc = tp
        best_sigma = 0.0
        best_benefit = 0.0
        best_changes = []

        for candidate in self._candidate_allocations(nodes, config):
            tc = self._clearance_time(candidate, nodes, queue_sizes, config.models)
            if tc == inf:
                continue

            changes = [
                e
                for e, (old, new) in enumerate(zip(current, candidate))
                if old != new
            ]
            sigma = max(
                (
                    self._switch_cost(
                        nodes,
                        config,
                        e,
                        candidate[e],
                        extra_cache=extra_cache,
                    )
                    for e in changes
                ),
                default=0.0,
            )

            if tp == inf:
                benefit = inf if tc < inf else -inf
            else:
                benefit = tp - tc - sigma

            if (
                benefit > best_benefit + self.epsilon
                or (
                    abs(benefit - best_benefit) <= self.epsilon
                    and len(changes) < len(best_changes)
                )
                or (
                    abs(benefit - best_benefit) <= self.epsilon
                    and len(changes) == len(best_changes)
                    and tc < best_tc - self.epsilon
                )
            ):
                best_allocation = tuple(candidate)
                best_tc = tc
                best_sigma = sigma
                best_benefit = benefit
                best_changes = changes

        return {
            "current": current,
            "allocation": best_allocation,
            "Tp": tp,
            "Tc": best_tc,
            "sigma": best_sigma,
            "benefit": best_benefit,
            "changes": best_changes,
        }

    def _best_stage(self, slot, nodes, queue_sizes, config):
        if sum(1 for n in nodes if n.prefetching) >= config.max_parallel_prefetch:
            return None

        best = None
        for e, node in enumerate(nodes):
            if not node.available or node.prefetching:
                continue
            if slot - node.last_switch_slot < config.cooldown:
                continue

            for target in config.models:
                if target == node.model or target in node.cached_models:
                    continue
                if float(config.model_sizes_map()[target]) > float(config.storages()[e]):
                    continue

                hypothetical = self._best_p3(
                    nodes,
                    queue_sizes,
                    config,
                    extra_cache=(e, target),
                )
                benefit = hypothetical["benefit"]
                if benefit == inf:
                    score = inf
                else:
                    score = float(benefit)

                # Stage only if the newly cached checkpoint participates in the
                # resulting best allocation and makes the paper-style objective
                # strictly positive.
                if score <= self.epsilon:
                    continue
                if hypothetical["allocation"][e] != target:
                    continue

                candidate = {
                    "node": e,
                    "target": target,
                    "hypothetical": hypothetical,
                    "score": score,
                }
                if best is None or candidate["score"] > best["score"]:
                    best = candidate
        return best

    def decide(self, slot, nodes, queue_sizes, arrival_history, config):
        self.last_decision = {
            "slot": slot,
            "action": "hold",
            "method": "EdgeCast cache-aware Tp - Tc - sigma",
            "reason": "No positive cache-aware P3 reallocation or useful staging action exists at this slot.",
        }

        if any(n.switch_remaining > 0 for n in nodes):
            self.last_decision["reason"] = "A model cutover is in progress; affected edge capacity is unavailable."
            return None

        if not any(queue_sizes[m] > 0 for m in config.models):
            self.last_decision["reason"] = "All model queues are empty."
            return None

        actual = self._best_p3(nodes, queue_sizes, config)
        tp_display = None if actual["Tp"] == inf else float(actual["Tp"])
        tc_display = None if actual["Tc"] == inf else float(actual["Tc"])
        benefit_display = None if actual["benefit"] == inf else float(actual["benefit"])

        self.last_decision.update(
            {
                "Tp": tp_display,
                "Tc": tc_display,
                "sigma": float(actual["sigma"]),
                "net_benefit": benefit_display,
                "target_allocation": list(actual["allocation"]),
            }
        )

        if (
            actual["allocation"] != actual["current"]
            and (
                actual["benefit"] == inf
                or actual["benefit"] > self.epsilon
            )
        ):
            elapsed = slot - self._last_reallocation_slot
            if self.min_reallocation_interval > 0 and elapsed < self.min_reallocation_interval:
                remaining = self.min_reallocation_interval - elapsed
                self.last_decision.update(
                    {
                        "action": "hold",
                        "stability_gate": "minimum-persistence",
                        "remaining_persistence_slots": int(remaining),
                        "reason": (
                            "A positive cache-aware P3 candidate exists, but the "
                            f"{self.min_reallocation_interval}-slot minimum allocation "
                            "persistence gate suppresses an immediate reversal."
                        ),
                    }
                )
                return None

            changes = [
                {"node": e, "target": actual["allocation"][e]}
                for e in actual["changes"]
            ]
            self.last_decision.update(
                {
                    "action": "multi_switch",
                    "changes": len(changes),
                    "stability_gate": "passed",
                    "reason": (
                        "Positive cache-aware Tp - Tc - sigma benefit justifies reallocation "
                        "after the minimum-persistence stability gate."
                    ),
                }
            )
            self._last_reallocation_slot = slot
            return {
                "type": "multi_switch",
                "changes": changes,
                "force_cold": False,
                "retain_cache": True,
            }

        stage = self._best_stage(slot, nodes, queue_sizes, config)
        if stage is None:
            active_prefetches = [n for n in nodes if n.prefetching]
            if active_prefetches:
                p = active_prefetches[0]
                self.last_decision.update(
                    {
                        "backhaul_busy": True,
                        "reason": (
                            f"Shared backhaul is staging {p.prefetch_target} to {p.name}; "
                            "active inference continues during transfer."
                        ),
                    }
                )
            return None

        e = int(stage["node"])
        target = str(stage["target"])
        hypothetical = stage["hypothetical"]
        hypothetical_benefit = (
            None
            if hypothetical["benefit"] == inf
            else float(hypothetical["benefit"])
        )
        self.last_decision.update(
            {
                "action": "prefetch",
                "edge": nodes[e].name,
                "source": nodes[e].model,
                "target": target,
                "cache_hit": False,
                "backhaul_slots": int(config.prefetch_time),
                "hypothetical_Tc": (
                    None
                    if hypothetical["Tc"] == inf
                    else float(hypothetical["Tc"])
                ),
                "hypothetical_sigma": float(hypothetical["sigma"]),
                "hypothetical_net_benefit": hypothetical_benefit,
                "reason": (
                    "Background staging is selected because caching this checkpoint "
                    "would make the same Tp - Tc - sigma allocation objective positive."
                ),
            }
        )
        return {"type": "prefetch", "node": e, "target": target}


@dataclass
class PredictivePolicy:
    """Legacy prototype kept for reproducibility of earlier experiments."""

    horizon: int = 3
    window: int = 5
    confidence_threshold: float = 0.70
    min_positive_trend: float = 0.60
    projected_backlog_trigger: float = 16.0
    observed_backlog_floor: int = 6
    benefit_margin: float = 2.0
    forecast_safety_margin: float = 1.5
    name: str = "Legacy predictive prototype"

    def __post_init__(self):
        self.predictor = ShortHorizonPredictor(self.window, self.horizon)
        self.last_decision = {}

    def decide(self, slot, nodes, queue_sizes, arrival_history, config):
        self.last_decision = {"slot": slot, "action": "hold", "reason": "Forecast gate is not yet satisfied."}
        if any(n.switch_remaining > 0 for n in nodes):
            self.last_decision["reason"] = "A model deployment is already in progress."
            return None

        fc = self.predictor.forecast(arrival_history)
        capacities = _model_capacities(nodes, config.models)
        projected = {}
        for m in config.models:
            incoming = fc.predicted_arrivals[m] * self.horizon
            service = capacities[m] * self.horizon
            projected[m] = max(0.0, queue_sizes[m] + incoming - service)

        target = max(config.models, key=lambda m: projected[m])
        confidence = float(fc.confidence[target])
        trend = float(fc.trend[target])
        self.last_decision.update(
            {
                "target": target,
                "confidence": confidence,
                "trend": trend,
                "projected_backlog": float(projected[target]),
                "queue": int(queue_sizes[target]),
            }
        )
        if (
            queue_sizes[target] < self.observed_backlog_floor
            or projected[target] < self.projected_backlog_trigger
            or trend < self.min_positive_trend
            or confidence < self.confidence_threshold
        ):
            return None

        donor_idx = _choose_donor(nodes, target, queue_sizes, config, slot)
        if donor_idx is None:
            return None
        donor = nodes[donor_idx]
        net_benefit = (
            projected[target] / max(1, donor.capacity)
            - config.switch_time
            - (1.0 - confidence) * self.forecast_safety_margin
        )
        self.last_decision["net_benefit"] = float(net_benefit)
        if net_benefit <= self.benefit_margin:
            return None

        self.last_decision.update({"action": "switch", "reason": "Legacy forecast gate accepted the switch."})
        return {"type": "switch", "node": donor_idx, "target": target}
