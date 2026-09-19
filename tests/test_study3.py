import numpy as np

import study2_window_collapse as s2
import study3_concurrent as s3
from common.env import Action, Kind, State


def orders_for(trial, k):
    return np.random.default_rng([s3.SEED, trial.idx, 3, k]).permuted(
        np.tile(np.arange(k), (trial.D, 1)), axis=1)


def test_one_agent_under_global_reproduces_study2():
    for i in range(25):
        trial = s2.make_trial(i)
        for H in (1, 4, 8):
            ref = s2.run_trajectory(trial, "timely_full", H)
            T_end, completion, eta, _, _ = s3.run_concurrent(trial, "global", 1, 0, H, orders_for(trial, 1))
            assert T_end == ref.T_end and completion == ref.completion
            assert np.array_equal(eta, ref.eta)


def test_single_agent_stale_view_is_exact():
    """Read-your-writes makes a lone agent's stale view equal to the truth."""
    for i in range(15):
        trial = s2.make_trial(i)
        a = s3.run_concurrent(trial, "local", 1, 4, 4, orders_for(trial, 1))
        b = s3.run_concurrent(trial, "global", 1, 0, 4, orders_for(trial, 1))
        assert np.array_equal(a[2], b[2])


def test_reserved_never_violates_and_baseline_never_breaks_feasibility():
    for i in range(30):
        trial = s2.make_trial(i)
        for cond, lag in (("local", 4), ("global", 0), ("reserved", 0)):
            T_end, _, eta, n, _ = s3.run_concurrent(trial, cond, 3, lag, 4, orders_for(trial, 3))
            W = trial.D - np.arange(trial.D + 1) - eta
            assert (W[:, :min(T_end, trial.D - s3.BASE_H) + 1] >= 0).all()
            assert n["unauth_commits"] == 0
            if cond == "reserved":
                assert (W[:, :min(T_end, trial.D - s3.BASE_H - 4) + 1] >= 4).all()


def test_pending_work_counts_as_current_state():
    s = State(t=10, deadline=30, task_needed=5, entangle=[0] * 8)     # W_handover = 14
    a = Action(Kind.ENTANGLE, component=0, amount=13)
    assert s3.admissible(s, a, 0)                  # lands at t=11 with W = 0
    assert not s3.admissible(s, a, 0, horizon=12)  # an in-flight CLEAN still needs a tick


def test_repair_is_always_admitted():
    s = State(t=10, deadline=20, task_needed=5, entangle=[9, 0, 0, 0, 0, 0, 0, 0])  # W = -5
    assert s3.admissible(s, Action(Kind.PRESERVE, component=0, amount=2), 4)
    assert not s3.admissible(s, Action(Kind.PRESERVE, component=1, amount=2), 4)   # no effect
    assert not s3.admissible(s, s3.CLEAN, 4)
