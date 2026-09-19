"""
Admissibility filters. Every filter is a pure rejector: it can refuse a
proposed action, it cannot compel one, and it cannot stop the clock.

  none          admits everything
  perm          authorisation and safety only
  feas          perm + every critical revision feasible in Tr(s, a)
  timely_full   perm + W(Tr(s, a), r) >= H, every witness recomputed
  timely_inc    same guarantee through the incremental witness cache

`feas` is `timely_full` with H = 0. The two differ in the threshold and
in nothing else, which is what makes the comparison fair.
"""
from __future__ import annotations

from time import perf_counter_ns

from .env import Action, State, transition
from .witness import Revision, WitnessCache, recovery_duration


class Filter:
    name = "none"

    def __init__(self, revisions: list[Revision], H: int = 0):
        self.revisions = revisions
        self.H = H
        self.evals = 0
        self.rejects = 0
        self.revalidations = 0
        self.wall_ns = 0

    def admits(self, state: State, action: Action) -> bool:
        t0 = perf_counter_ns()
        ok = self._check(state, action)
        self.wall_ns += perf_counter_ns() - t0
        self.evals += 1
        self.rejects += not ok
        return ok

    def _check(self, state: State, action: Action) -> bool:
        return True

    def notify_commit(self, pre: State, action: Action, post: State) -> None:
        pass


class PermFilter(Filter):
    name = "perm"

    def _check(self, state, action):
        return action.authorised


class ThresholdFilter(PermFilter):
    """Full recomputation of every witness against the post-state."""
    threshold = 0

    def _check(self, state, action):
        if not action.authorised:
            return False
        post = transition(state, action)
        budget = post.deadline - post.t - self.threshold
        for r in self.revisions:
            self.revalidations += 1
            if r.critical and recovery_duration(r, post) > budget:
                return False
        return True


class FeasFilter(ThresholdFilter):
    name = "feas"


class TimelyFullFilter(ThresholdFilter):
    name = "timely_full"

    def __init__(self, revisions, H=0):
        super().__init__(revisions, H)
        self.threshold = H


class TimelyIncFilter(PermFilter):
    name = "timely_inc"

    def __init__(self, revisions, H, state: State):
        super().__init__(revisions, H)
        self.cache = WitnessCache(revisions, state)
        self.revalidations = self.cache.revalidations   # building the cache counts
        self._last: tuple[Action, list[int], int] | None = None

    def _check(self, state, action):
        if not action.authorised:
            return False
        post = transition(state, action)
        eta = self.cache.evaluate(state, post, action)
        self.revalidations = self.cache.revalidations
        self._last = (action, eta, state.version)
        budget = post.deadline - post.t - self.H
        for e, r in zip(eta, self.revisions):
            if r.critical and e > budget:
                return False
        return True

    def notify_commit(self, pre, action, post):
        last = self._last
        if last is not None and last[0] is action and last[2] == pre.version:
            self.cache.commit(last[1], post)  # commit against the checked version
        else:                                # unchecked (WAIT) or the state moved
            self.cache.commit_unchecked(pre, action, post)
            self.revalidations = self.cache.revalidations
        self._last = None


def make_filter(cond: str, revisions: list[Revision], H: int, state: State) -> Filter:
    if cond == "none":
        return Filter(revisions, H)
    if cond == "perm":
        return PermFilter(revisions, H)
    if cond == "feas":
        return FeasFilter(revisions, H)
    if cond == "timely_full":
        return TimelyFullFilter(revisions, H)
    if cond == "timely_inc":
        return TimelyIncFilter(revisions, H, state)
    raise ValueError(cond)
