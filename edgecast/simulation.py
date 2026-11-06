from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass
class EdgeNode:
    name: str
    model: str
    capacity: int = 5
    switch_remaining: int = 0
    target_model: str | None = None
    switch_mode: str | None = None
    last_switch_slot: int = -10_000
    cached_models: set[str] = field(default_factory=set)
    prefetch_target: str | None = None
    prefetch_remaining: int = 0
    retain_cache_on_switch: bool = True

    def __post_init__(self) -> None:
        self.cached_models.add(self.model)

    @property
    def available(self) -> bool:
        return self.switch_remaining == 0

    @property
    def prefetching(self) -> bool:
        return self.prefetch_remaining > 0 and self.prefetch_target is not None

    def begin_switch(
        self,
        new_model: str,
        switch_time: int,
        slot: int,
        mode: str,
        *,
        retain_cache: bool = True,
    ) -> None:
        if new_model == self.model or not self.available:
            return
        self.target_model = new_model
        self.switch_remaining = max(1, int(switch_time))
        self.switch_mode = mode
        self.last_switch_slot = slot
        self.retain_cache_on_switch = bool(retain_cache)

    def tick_switch(self) -> None:
        if self.switch_remaining <= 0:
            return
        self.switch_remaining -= 1
        if self.switch_remaining == 0 and self.target_model is not None:
            self.model = self.target_model
            if self.retain_cache_on_switch:
                self.cached_models.add(self.model)
            else:
                # Faithful base-paper path: there is no persistent inactive
                # checkpoint cache state. Only the newly deployed active model
                # remains locally represented.
                self.cached_models = {self.model}
            self.target_model = None
            self.switch_mode = None
            self.retain_cache_on_switch = True

    def begin_prefetch(self, model: str, slots: int) -> bool:
        if self.prefetching or model == self.model or model in self.cached_models:
            return False
        self.prefetch_target = model
        self.prefetch_remaining = max(1, int(slots))
        return True

    def tick_prefetch(self, cache_slots: int) -> None:
        if not self.prefetching:
            return
        self.prefetch_remaining -= 1
        if self.prefetch_remaining == 0 and self.prefetch_target is not None:
            target = self.prefetch_target
            while len(self.cached_models) >= max(1, cache_slots):
                inactive = sorted(m for m in self.cached_models if m != self.model)
                if not inactive:
                    break
                self.cached_models.remove(inactive[0])
            self.cached_models.add(target)
            self.prefetch_target = None


@dataclass
class SimulationConfig:
    models: Tuple[str, ...] = ("text", "image", "video")
    initial_allocation: Tuple[str, ...] = ("text", "text", "image", "image", "video")
    node_capacity: int = 5
    node_capacities: Tuple[int, ...] | None = None
    switch_time: int = 3
    warm_switch_time: int = 1
    prefetch_time: int = 3
    cache_slots: int = 2
    max_parallel_prefetch: int = 1
    cooldown: int = 6
    max_drain_slots: int = 1000
    model_sizes: Dict[str, float] | None = None
    node_storage: Tuple[float, ...] | None = None
    deployment_slots: Dict[str, Tuple[int, ...]] | None = None

    def capacities(self) -> Tuple[int, ...]:
        if self.node_capacities is None:
            return tuple(self.node_capacity for _ in self.initial_allocation)
        if len(self.node_capacities) != len(self.initial_allocation):
            raise ValueError("node_capacities must match initial_allocation length")
        return self.node_capacities

    def model_sizes_map(self) -> Dict[str, float]:
        if self.model_sizes is None:
            return {m: 1.0 for m in self.models}
        return {m: float(self.model_sizes.get(m, 1.0)) for m in self.models}

    def storages(self) -> Tuple[float, ...]:
        sizes = self.model_sizes_map()
        default_storage = max(sizes.values()) if sizes else 1.0
        if self.node_storage is None:
            return tuple(default_storage for _ in self.initial_allocation)
        if len(self.node_storage) != len(self.initial_allocation):
            raise ValueError("node_storage must match initial_allocation length")
        return tuple(float(x) for x in self.node_storage)

    def cold_switch_slots(self, model: str, node_index: int) -> int:
        if self.deployment_slots is not None and model in self.deployment_slots:
            values = self.deployment_slots[model]
            if len(values) != len(self.initial_allocation):
                raise ValueError("deployment_slots entries must match initial_allocation length")
            return max(1, int(values[node_index]))
        return max(1, int(self.switch_time))


@dataclass
class SimulationResult:
    policy_name: str
    slots: List[int]
    queues: Dict[str, List[int]]
    allocations: List[List[str]]
    arrivals: Dict[str, List[int]]
    decisions: List[dict]
    node_states: List[List[dict]]
    total_arrivals: int
    completed: int
    completion_slot: int
    avg_latency: float
    p95_latency: float
    avg_backlog: float
    max_backlog: int
    switches: int
    warm_switches: int
    cold_switches: int
    prefetches: int
    switch_downtime_node_slots: int
    prefetch_node_slots: int

    def summary(self) -> Dict[str, float]:
        return {
            "policy": self.policy_name,
            "total_requests": self.total_arrivals,
            "completion_slot": self.completion_slot,
            "avg_latency": round(self.avg_latency, 3),
            "p95_latency": round(self.p95_latency, 3),
            "avg_backlog": round(self.avg_backlog, 3),
            "max_backlog": self.max_backlog,
            "switches": self.switches,
            "warm_switches": self.warm_switches,
            "cold_switches": self.cold_switches,
            "background_prefetches": self.prefetches,
            "switch_downtime_node_slots": self.switch_downtime_node_slots,
        }


def _weighted_percentile(histogram: Dict[int, int], p: float) -> float:
    total = sum(histogram.values())
    if total <= 0:
        return 0.0
    target = max(1, int(total * p + 0.999999))
    acc = 0
    for value in sorted(histogram):
        acc += histogram[value]
        if acc >= target:
            return float(value)
    return float(max(histogram))


def _queue_size(cohorts) -> int:
    return sum(count for _, count in cohorts)


def _snapshot_nodes(nodes: List[EdgeNode]) -> List[dict]:
    return [
        {
            "name": n.name,
            "active": n.model,
            "target": n.target_model,
            "switch_remaining": n.switch_remaining,
            "switch_mode": n.switch_mode,
            "cache": sorted(n.cached_models),
            "prefetch": n.prefetch_target,
            "prefetch_remaining": n.prefetch_remaining,
            "capacity": n.capacity,
        }
        for n in nodes
    ]


def _start_switch(
    *,
    node: EdgeNode,
    node_idx: int,
    target_model: str,
    config: SimulationConfig,
    slot: int,
    force_cold: bool,
    retain_cache: bool,
):
    if not node.available or node.model == target_model:
        return None

    source_model = node.model
    warm = (not force_cold) and target_model in node.cached_models
    switch_slots = (
        config.warm_switch_time
        if warm
        else config.cold_switch_slots(target_model, node_idx)
    )
    mode = "warm" if warm else "cold"
    node.begin_switch(
        target_model,
        switch_slots,
        slot,
        mode,
        retain_cache=retain_cache,
    )
    return {
        "node": node.name,
        "source": source_model,
        "target": target_model,
        "mode": mode,
        "switch_slots": int(switch_slots),
    }


def run_simulation(
    demand: Sequence[Dict[str, int]],
    policy,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    config = config or SimulationConfig()
    if set(config.initial_allocation) - set(config.models):
        raise ValueError("initial_allocation contains a model not listed in config.models")

    nodes = [
        EdgeNode(f"edge-{i+1}", model, int(capacity))
        for i, (model, capacity) in enumerate(
            zip(config.initial_allocation, config.capacities())
        )
    ]
    queues = {m: deque() for m in config.models}
    arrival_history = {m: [] for m in config.models}
    queue_history = {m: [] for m in config.models}
    allocations: List[List[str]] = []
    decisions: List[dict] = []
    node_states: List[List[dict]] = []
    slots: List[int] = []
    latency_histogram: Dict[int, int] = {}
    latency_sum = 0
    switches = warm_switches = cold_switches = prefetches = 0
    downtime = prefetch_node_slots = completed = total_arrivals = 0
    t = 0
    demand_len = len(demand)

    while True:
        # Nodes undergoing model deployment are unavailable, consistent with
        # the assigned paper's Stage-2 service-interruption model.
        for node in nodes:
            if node.switch_remaining > 0:
                downtime += 1
                node.tick_switch()
            if node.prefetching:
                prefetch_node_slots += 1
                node.tick_prefetch(config.cache_slots)

        arrivals = demand[t] if t < demand_len else {m: 0 for m in config.models}
        for m in config.models:
            count = max(0, int(arrivals.get(m, 0)))
            arrival_history[m].append(count)
            total_arrivals += count
            if count:
                queues[m].append([t, count])

        # Under the model-aggregated demonstrator, solving the paper's P2b for
        # a model queue is equivalent to using as much currently available
        # compatible edge capacity as possible. Q_m is positive and therefore
        # maximizing Q_m b_m maximizes b_m subject to the active-node capacity.
        for m in config.models:
            capacity = sum(
                node.capacity
                for node in nodes
                if node.available and node.model == m
            )
            remaining = capacity
            while remaining > 0 and queues[m]:
                arrival_slot, count = queues[m][0]
                served = min(remaining, count)
                latency = t - arrival_slot + 1
                latency_histogram[latency] = latency_histogram.get(latency, 0) + served
                latency_sum += latency * served
                completed += served
                remaining -= served
                count -= served
                if count == 0:
                    queues[m].popleft()
                else:
                    queues[m][0][1] = count

        queue_sizes = {m: _queue_size(queues[m]) for m in config.models}
        action = policy.decide(
            slot=t,
            nodes=nodes,
            queue_sizes=queue_sizes,
            arrival_history=arrival_history,
            config=config,
        )

        if action is not None:
            if isinstance(action, tuple):
                action = {"type": "switch", "node": action[0], "target": action[1]}

            kind = action.get("type", "switch")

            if kind == "prefetch":
                node_idx = int(action["node"])
                target_model = str(action["target"])
                node = nodes[node_idx]
                active_prefetches = sum(1 for n in nodes if n.prefetching)
                if (
                    active_prefetches < config.max_parallel_prefetch
                    and node.begin_prefetch(target_model, config.prefetch_time)
                ):
                    prefetches += 1
                    if getattr(policy, "last_decision", None):
                        policy.last_decision.update(
                            {
                                "edge": node.name,
                                "source": node.model,
                                "mode": "background-prefetch",
                                "backhaul_slots": config.prefetch_time,
                            }
                        )

            elif kind == "switch":
                node_idx = int(action["node"])
                target_model = str(action["target"])
                detail = _start_switch(
                    node=nodes[node_idx],
                    node_idx=node_idx,
                    target_model=target_model,
                    config=config,
                    slot=t,
                    force_cold=bool(action.get("force_cold", False)),
                    retain_cache=bool(action.get("retain_cache", True)),
                )
                if detail is not None:
                    switches += 1
                    warm_switches += int(detail["mode"] == "warm")
                    cold_switches += int(detail["mode"] == "cold")
                    if getattr(policy, "last_decision", None):
                        policy.last_decision.update(detail)

            elif kind == "multi_switch":
                details = []
                force_cold = bool(action.get("force_cold", False))
                retain_cache = bool(action.get("retain_cache", True))
                for change in action.get("changes", []):
                    node_idx = int(change["node"])
                    target_model = str(change["target"])
                    detail = _start_switch(
                        node=nodes[node_idx],
                        node_idx=node_idx,
                        target_model=target_model,
                        config=config,
                        slot=t,
                        force_cold=force_cold,
                        retain_cache=retain_cache,
                    )
                    if detail is not None:
                        details.append(detail)
                        switches += 1
                        warm_switches += int(detail["mode"] == "warm")
                        cold_switches += int(detail["mode"] == "cold")

                if details and getattr(policy, "last_decision", None):
                    policy.last_decision.update(
                        {
                            "mode": "cold" if all(d["mode"] == "cold" for d in details) else "mixed",
                            "switch_slots": max(d["switch_slots"] for d in details),
                            "change_details": details,
                            "edges": ", ".join(d["node"] for d in details),
                        }
                    )

        decisions.append(dict(getattr(policy, "last_decision", {}) or {}))
        slots.append(t)
        allocations.append(
            [
                f"{n.model}->{n.target_model}" if n.switch_remaining > 0 else n.model
                for n in nodes
            ]
        )
        node_states.append(_snapshot_nodes(nodes))
        for m in config.models:
            queue_history[m].append(_queue_size(queues[m]))

        arrivals_finished = t >= demand_len - 1
        queues_empty = all(_queue_size(q) == 0 for q in queues.values())
        switching_done = all(
            n.switch_remaining == 0 and not n.prefetching for n in nodes
        )
        if arrivals_finished and queues_empty and switching_done:
            break
        if t >= demand_len + config.max_drain_slots:
            break
        t += 1

    total_backlog_samples = [
        sum(queue_history[m][i] for m in config.models)
        for i in range(len(slots))
    ]
    return SimulationResult(
        policy_name=policy.name,
        slots=slots,
        queues=queue_history,
        allocations=allocations,
        arrivals=arrival_history,
        decisions=decisions,
        node_states=node_states,
        total_arrivals=total_arrivals,
        completed=completed,
        completion_slot=slots[-1] if slots else 0,
        avg_latency=(latency_sum / completed) if completed else 0.0,
        p95_latency=_weighted_percentile(latency_histogram, 0.95),
        avg_backlog=(
            sum(total_backlog_samples) / len(total_backlog_samples)
            if total_backlog_samples
            else 0.0
        ),
        max_backlog=max(total_backlog_samples) if total_backlog_samples else 0,
        switches=switches,
        warm_switches=warm_switches,
        cold_switches=cold_switches,
        prefetches=prefetches,
        switch_downtime_node_slots=downtime,
        prefetch_node_slots=prefetch_node_slots,
    )
