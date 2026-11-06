from edgecast.policies import (
    CacheAwarePolicy,
    NguyenPaperPolicy,
    ReactivePolicy,
    optimize_initial_allocation,
)
from edgecast.scenarios import ramp_video_burst, sudden_spike
from edgecast.simulation import EdgeNode, SimulationConfig, run_simulation


def test_reactive_simulation_completes_all_requests():
    demand = ramp_video_burst()
    result = run_simulation(demand, ReactivePolicy(), SimulationConfig())
    assert result.completed == result.total_arrivals
    assert result.total_arrivals > 0


def test_nguyen_paper_policy_completes_all_requests():
    demand = ramp_video_burst(seed=7)
    config = SimulationConfig(switch_time=4, max_drain_slots=1000)
    result = run_simulation(demand, NguyenPaperPolicy(), config)
    assert result.completed == result.total_arrivals
    assert result.total_arrivals > 0
    assert all(d.get("action") != "prefetch" for d in result.decisions)


def test_nguyen_paper_policy_uses_p3_objective_and_cold_only_switches():
    demand = ramp_video_burst(seed=7)
    config = SimulationConfig(
        switch_time=4,
        initial_allocation=("text", "text", "image", "image", "video"),
    )
    result = run_simulation(demand, NguyenPaperPolicy(), config)
    events = [d for d in result.decisions if d.get("action") == "multi_switch"]
    if events:
        assert all(d.get("sigma", 0) >= 0 for d in events)
        assert all(d.get("Tc") is not None for d in events)
        assert result.warm_switches == 0
        assert result.cold_switches == result.switches


def test_reduced_stage1_optimizer_keeps_demanded_models_active():
    demand = [
        {"text": 10, "image": 2, "video": 1},
        {"text": 12, "image": 2, "video": 1},
    ]
    allocation = optimize_initial_allocation(
        demand,
        ("text", "image", "video"),
        (5, 5, 4, 3, 2, 2),
    )
    assert set(allocation) == {"text", "image", "video"}
    assert len(allocation) == 6


def test_cache_aware_simulation_completes_all_requests():
    demand = ramp_video_burst(seed=7)
    result = run_simulation(demand, CacheAwarePolicy(), SimulationConfig())
    assert result.completed == result.total_arrivals


def test_cache_aware_switch_uses_same_system_level_p3_objective():
    config = SimulationConfig(
        models=("text", "video"),
        initial_allocation=("text", "video", "video"),
        node_capacities=(4, 6, 10),
        switch_time=4,
        warm_switch_time=1,
        cooldown=0,
    )
    nodes = [
        EdgeNode("edge-1", "text", 4),
        EdgeNode("edge-2", "video", 6),
        EdgeNode("edge-3", "video", 10),
    ]
    nodes[0].cached_models.add("video")

    policy = CacheAwarePolicy()
    action = policy.decide(
        slot=10,
        nodes=nodes,
        queue_sizes={"text": 0, "video": 10},
        arrival_history={"text": [], "video": []},
        config=config,
    )

    # Current video clearance time is 10/16 = 0.625. Switching edge-1 would
    # make it 10/20 = 0.5, but even the warm sigma=1 gives -0.875 benefit.
    assert action is None
    assert policy.last_decision["action"] == "hold"
    assert policy.last_decision["net_benefit"] == 0.0


def test_cache_aware_can_stage_on_last_active_copy_without_cutting_it_over():
    config = SimulationConfig(
        models=("a", "b"),
        initial_allocation=("a", "b"),
        node_capacities=(10, 10),
        switch_time=4,
        warm_switch_time=1,
        prefetch_time=2,
        cache_slots=2,
        max_parallel_prefetch=1,
        cooldown=0,
    )
    nodes = [
        EdgeNode("edge-1", "a", 10),
        EdgeNode("edge-2", "b", 10),
    ]

    policy = CacheAwarePolicy()
    action = policy.decide(
        slot=5,
        nodes=nodes,
        queue_sizes={"a": 0, "b": 60},
        arrival_history={"a": [], "b": []},
        config=config,
    )

    # Cold P3: 6 - 3 - 4 = -1 (do not switch).
    # If b is staged on edge-1: 6 - 3 - 1 = +2, so staging is useful.
    # edge-1 is the last active copy of a, but staging does not remove it.
    assert action == {"type": "prefetch", "node": 0, "target": "b"}
    assert policy.last_decision["action"] == "prefetch"
    assert policy.last_decision["hypothetical_net_benefit"] == 2.0


def test_cache_aware_uses_warm_sigma_after_checkpoint_is_cached():
    config = SimulationConfig(
        models=("a", "b"),
        initial_allocation=("a", "b"),
        node_capacities=(10, 10),
        switch_time=4,
        warm_switch_time=1,
        cooldown=0,
    )
    nodes = [
        EdgeNode("edge-1", "a", 10),
        EdgeNode("edge-2", "b", 10),
    ]
    nodes[0].cached_models.add("b")

    policy = CacheAwarePolicy()
    action = policy.decide(
        slot=6,
        nodes=nodes,
        queue_sizes={"a": 0, "b": 60},
        arrival_history={"a": [], "b": []},
        config=config,
    )

    assert action is not None
    assert action["type"] == "multi_switch"
    assert action["force_cold"] is False
    assert policy.last_decision["sigma"] == 1.0
    assert policy.last_decision["net_benefit"] == 2.0


def test_shared_backhaul_limit_is_respected():
    demand = [{"a": 0, "b": 16} for _ in range(20)]
    config = SimulationConfig(
        models=("a", "b"),
        initial_allocation=("a", "b", "a"),
        node_capacities=(10, 10, 10),
        switch_time=4,
        warm_switch_time=1,
        prefetch_time=5,
        max_parallel_prefetch=1,
        cache_slots=2,
        cooldown=0,
    )
    result = run_simulation(demand, CacheAwarePolicy(), config)
    for snapshot in result.node_states:
        assert sum(1 for node in snapshot if node["prefetch"]) <= 1


def test_sudden_spike_still_completes():
    demand = sudden_spike()
    result = run_simulation(demand, CacheAwarePolicy(), SimulationConfig())
    assert result.completed == result.total_arrivals


def test_minimum_reallocation_persistence_suppresses_immediate_reversal():
    config = SimulationConfig(
        models=("a", "b"),
        initial_allocation=("a", "b", "b"),
        node_capacities=(10, 10, 10),
        switch_time=4,
        warm_switch_time=1,
        cooldown=0,
    )
    nodes = [
        EdgeNode("edge-1", "a", 10),
        EdgeNode("edge-2", "b", 10),
        EdgeNode("edge-3", "b", 10),
    ]
    nodes[0].cached_models.add("b")

    policy = CacheAwarePolicy(min_reallocation_interval=6)
    first = policy.decide(
        slot=10,
        nodes=nodes,
        queue_sizes={"a": 0, "b": 100},
        arrival_history={"a": [], "b": []},
        config=config,
    )
    assert first is not None and first["type"] == "multi_switch"

    # Rebuild an available state and create a different positive candidate only
    # two slots later. The stability gate should hold the active allocation.
    nodes = [
        EdgeNode("edge-1", "b", 10),
        EdgeNode("edge-2", "a", 10),
        EdgeNode("edge-3", "a", 10),
    ]
    nodes[0].cached_models.add("a")
    nodes[1].cached_models.add("b")
    nodes[2].cached_models.add("b")
    second = policy.decide(
        slot=12,
        nodes=nodes,
        queue_sizes={"a": 100, "b": 0},
        arrival_history={"a": [], "b": []},
        config=config,
    )
    assert second is None
    assert policy.last_decision["stability_gate"] == "minimum-persistence"
    assert policy.last_decision["remaining_persistence_slots"] == 4


def test_zero_reallocation_persistence_preserves_legacy_behavior():
    config = SimulationConfig(
        models=("a", "b"),
        initial_allocation=("a", "b"),
        node_capacities=(10, 10),
        switch_time=4,
        warm_switch_time=1,
        cooldown=0,
    )
    nodes = [EdgeNode("edge-1", "a", 10), EdgeNode("edge-2", "b", 10)]
    nodes[0].cached_models.add("b")
    policy = CacheAwarePolicy(min_reallocation_interval=0)
    action = policy.decide(
        slot=1,
        nodes=nodes,
        queue_sizes={"a": 0, "b": 60},
        arrival_history={"a": [], "b": []},
        config=config,
    )
    assert action is not None
    assert action["type"] == "multi_switch"
