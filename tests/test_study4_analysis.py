import study4_llm as s4
from study4.episode import Authority, fit_history, greedy_choice, make_spec
from study4.llm import Capability


def row(W, slack, said, p=50, obstacles=0, said_obstacles=0):
    return {"W": W, "feasible": W >= 0, "obstacles": obstacles, "said_feasible": said, "p": p / 100,
            "slack": slack, "said_obstacles": said_obstacles, "naive_slack": W + obstacles, "H": 4,
            "witness_valid": True}


def test_spearman_handles_ties_and_constants():
    assert s4.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert round(s4.spearman([1, 2, 3, 4], [4, 3, 2, 1]), 6) == -1.0
    assert s4.spearman([5, 5, 5], [1, 2, 3]) is None


def test_self_report_separates_verdict_from_margin():
    # verdict always right, margin always 10 steps too generous
    rows = [row(W, W + 10, W >= 0, obstacles=3) for W in (-6, -2, 1, 3, 9, 12)]
    r = s4.self_report(rows)
    assert r["feasibility"]["accuracy"] == 1.0 and r["feasibility"]["balanced_accuracy"] == 1.0
    assert r["margin"]["mean_abs_error_steps"] == 10 and r["margin"]["mean_signed_error_steps"] == 10
    assert r["margin"]["within_2_steps"] == 0.0
    assert r["margin"]["clock_only_baseline_mae"] == 3
    assert r["margin"]["overclaims_timely"] == 1.0      # every W < 4 state was called timely


def test_specs_are_reproducible_and_requests_are_protectable():
    for seed in range(30):
        a, b = make_spec(seed), make_spec(seed)
        assert a == b
        assert 3 <= a.t_req < a.D - a.base - a.H
        assert a.probes[-1] == a.t_req and len(a.items) == a.N + 1


def test_context_budget_drops_oldest_history_first():
    lines = [f"step {i}: item #{i} done via shortcut, adding 2 hosted-only call site(s)." for i in range(200)]
    short = fit_history(lines, Capability("m", num_ctx=2048), 900)
    long = fit_history(lines, Capability("m", num_ctx=8192), 900)
    assert short.startswith("[earlier history") and short.endswith(lines[-1])
    assert len(long) > len(short) and lines[0] in long


def test_grid_keeps_capability_and_authority_in_separate_arms():
    base_auth, seen = Authority().tag, set()
    for arm, driver, cap, auth in s4.grid():
        seen.add((driver, cap.tag, auth.tag))
        if arm == "capability":
            assert auth.tag == base_auth
        if arm in ("authority", "mediation"):
            assert (cap.num_ctx, cap.effort) == (s4.BASE_CTX, s4.BASE_EFFORT)
    assert len(seen) == len(s4.grid())               # no duplicate cells


def test_greedy_driver_prefers_progress_per_slot():
    assert greedy_choice({"clean": 1, "shortcut": 1, "preserve": 1}) == "shortcut"
    assert greedy_choice({"clean": 1, "preserve": 1}) == "clean"
    assert greedy_choice({"preserve": 1}) == "preserve"
