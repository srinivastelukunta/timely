import random

from common.env import CLEAN, WAIT, Action, Kind, State, checkpoint, restore, transition
from common.filters import (FeasFilter, PermFilter, TimelyFullFilter,
                            TimelyIncFilter)
from common.witness import Revision, window

REVS = [
    Revision("handover", 4, frozenset({0, 1, 2})),
    Revision("run_local", 2, frozenset({0, 1})),
    Revision("export_state", 1, frozenset({2})),
]


def state(t=0, deadline=30, entangle=(0, 0, 0, 0)):
    return State(t=t, deadline=deadline, task_needed=50, entangle=list(entangle))


def test_perm_rejects_only_unauthorised():
    f = PermFilter(REVS)
    assert f.admits(state(), Action(Kind.ENTANGLE, component=0, amount=9))
    assert not f.admits(state(), Action(Kind.UNAUTH, amount=2, authorised=False))
    assert (f.evals, f.rejects) == (2, 1)


def test_feas_threshold_is_exactly_zero():
    s = state(deadline=10)                       # W_handover = 10 - 0 - 4 = 6
    assert window(REVS[0], s) == 6
    f = FeasFilter(REVS)
    assert f.admits(s, Action(Kind.ENTANGLE, component=0, amount=5))      # W' = 0
    assert not f.admits(s, Action(Kind.ENTANGLE, component=0, amount=6))  # W' = -1


def test_timely_threshold_is_H():
    s = state(deadline=10)
    f = TimelyFullFilter(REVS, H=3)
    assert f.admits(s, Action(Kind.ENTANGLE, component=0, amount=2))      # W' = 3
    assert not f.admits(s, Action(Kind.ENTANGLE, component=0, amount=3))  # W' = 2
    assert not f.admits(s, Action(Kind.UNAUTH, amount=0, authorised=False))


def random_action(rng):
    k = rng.random()
    if k < 0.5:
        return Action(Kind.ENTANGLE, component=rng.randrange(4), amount=rng.randint(1, 3),
                      analysable=rng.random() > 0.2)
    if k < 0.7:
        return CLEAN
    if k < 0.9:
        return Action(Kind.PRESERVE, component=rng.randrange(4), amount=rng.randint(1, 3))
    return Action(Kind.UNAUTH, amount=2, authorised=False)


def test_incremental_matches_full_on_random_walks():
    """inc must never admit what full rejects. Here they must agree exactly."""
    for seed in range(200):
        rng = random.Random(seed)
        H = rng.choice([0, 1, 2, 4, 8])
        s = state(deadline=rng.randint(25, 60),
                  entangle=[rng.randint(0, 2) for _ in range(4)])
        full, inc = TimelyFullFilter(REVS, H), TimelyIncFilter(REVS, H, s)
        cp = checkpoint(s)
        while s.t < s.deadline - 2:
            a = random_action(rng)
            ok_full, ok_inc = full.admits(s, a), inc.admits(s, a)
            assert ok_full == ok_inc, (seed, s, a)
            if rng.random() < 0.05:              # state moves outside the filter
                s = restore(s, cp)
                continue
            chosen = a if ok_inc else WAIT
            post = transition(s, chosen)
            inc.notify_commit(s, chosen, post)
            s = post


def test_incremental_skips_unaffected_witnesses():
    s = state(deadline=60)
    full, inc = TimelyFullFilter(REVS, 2), TimelyIncFilter(REVS, 2, s)
    start = inc.revalidations
    free = Action(Kind.ENTANGLE, component=3, amount=1)    # no witness depends on 3
    one = Action(Kind.ENTANGLE, component=2, amount=1)     # handover, export_state
    for a in (free, one, CLEAN):
        assert full.admits(s, a) and inc.admits(s, a)
    assert full.revalidations == 9
    assert inc.revalidations - start == 2
