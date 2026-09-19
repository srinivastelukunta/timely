import numpy as np

import study2_window_collapse as s2


def test_trials_are_reproducible_and_paired():
    a, b = s2.make_trial(7), s2.make_trial(7)
    assert (a.N, a.D, a.rho, a.entangle0, a.units, a.u) == (b.N, b.D, b.rho, b.entangle0, b.units, b.u)


def test_feas_is_timely_with_H_zero():
    for i in range(20):
        trial = s2.make_trial(i)
        f, t = s2.run_trajectory(trial, "feas", 0), s2.run_trajectory(trial, "timely_full", 0)
        assert f.T_end == t.T_end and np.array_equal(f.eta, t.eta)


def test_timely_invariant_holds_while_protectable():
    for i in range(40):
        trial = s2.make_trial(i)
        for H in s2.H_VALUES:
            tr = s2.run_trajectory(trial, "timely_inc", H)
            assert tr.inc_unsound == 0
            last = min(tr.T_end, trial.D - s2.BASE_H - H)
            W = trial.D - np.arange(last + 1) - tr.eta[s2.HANDOVER, :last + 1]
            assert W.min() >= H


def test_noop_agent_scores_zero_on_task():
    tr = s2.run_trajectory(s2.make_trial(3), "none", 0, agent="noop")
    assert tr.completion == 0.0


def test_small_run_passes_its_own_sanity_checks():
    r = s2.run(40)
    assert r["sanity_checks"]["inc_admits_what_full_rejects"] == 0
    assert set(r["emit"]) >= {"study2.feas_success_H4", "study2.timely_success_H4", "study2.completion_delta"}
