from common.env import (CLEAN, WAIT, Action, Kind, State, checkpoint, restore,
                        ticks, transition)


def fresh(**kw):
    base = dict(t=0, deadline=20, task_needed=10, entangle=[0, 0, 0])
    base.update(kw)
    return State(**base)


def test_clean_costs_two_ticks_and_is_pending_in_between():
    mid, end = ticks(fresh(), CLEAN)
    assert (mid.t, mid.pending, mid.internal) == (1, 1, 0)
    assert (end.t, end.pending, end.internal) == (2, 0, 1)
    assert end.entangle == [0, 0, 0]


def test_entangle_and_preserve():
    s = transition(fresh(), Action(Kind.ENTANGLE, component=1, amount=3))
    assert (s.t, s.internal, s.entangle) == (1, 1, [0, 3, 0])
    s = transition(s, Action(Kind.PRESERVE, component=1, amount=2))
    assert (s.internal, s.entangle) == (1, [0, 1, 0])
    s = transition(s, Action(Kind.PRESERVE, component=1, amount=2))
    assert s.entangle == [0, 0, 0]          # floors at zero


def test_transition_does_not_mutate_its_input():
    s0 = fresh()
    transition(s0, Action(Kind.ENTANGLE, component=0, amount=2))
    assert (s0.t, s0.internal, s0.entangle, s0.version) == (0, 0, [0, 0, 0], 0)


def test_restore_reverses_internal_and_pending_only():
    s = fresh()
    cp = checkpoint(s)
    s = transition(s, Action(Kind.ENTANGLE, component=0, amount=2))
    s = transition(s, Action(Kind.UNAUTH, amount=2, authorised=False))
    s = ticks(s, CLEAN)[0]                   # stop mid-CLEAN, work is pending
    assert s.pending == 1 and s.ext_entangle == 2 and len(s.external) == 1

    r = restore(s, cp)
    assert (r.internal, r.entangle, r.pending) == (0, [0, 0, 0], 0)
    # the third category survives, and so does the clock
    assert r.external == s.external
    assert r.ext_entangle == 2
    assert r.t == s.t
    assert r.version > s.version


def test_deadline_commitment_is_recorded_and_survives_restore():
    s = fresh(t=18, deadline=19)
    cp = checkpoint(s)
    s = transition(s, WAIT)
    assert ("deploy", 19) in s.external
    assert ("deploy", 19) in restore(s, cp).external


def test_write_sets():
    assert CLEAN.write_set == frozenset()
    assert Action(Kind.ENTANGLE, component=2, amount=1).write_set == frozenset({2})
    assert Action(Kind.ENTANGLE, component=2, amount=1, analysable=False).write_set is None
