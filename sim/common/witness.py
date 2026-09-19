"""
Recovery witnesses, windows, and the incremental witness cache.

A witness for revision r records the dependency set D_r. Its recovery
duration is  eta_r(s) = base_r + sum of entanglement over D_r  and its
window is    W(s, r) = deadline - t - eta_r(s).

W here is the closed-form window. It assumes a hold
policy that neither raises nor lowers eta. Because PRESERVE can lower
eta, the closed form is a lower bound on Definition 3, never an
overestimate.
"""
from __future__ import annotations

from dataclasses import dataclass

from .env import EXT, Action, State


@dataclass(frozen=True)
class Revision:
    name: str
    base: int                  # irreducible recovery duration
    deps: frozenset[int]       # D_r, component indices (EXT is always implied)
    critical: bool = True


def recovery_duration(rev: Revision, state: State) -> int:
    return rev.base + state.ext_entangle + sum(state.entangle[c] for c in rev.deps)


def window(rev: Revision, state: State) -> int:
    return state.deadline - state.t - recovery_duration(rev, state)


class WitnessCache:
    """
    Incremental maintenance, steps 1 to 4 and 6 of the maintenance algorithm.
    (Step 5, reservation, only matters under concurrency: Study 3.)

    Cached values are recovery durations. Time is an analysable uniform
    effect, so a cached window is refreshed by arithmetic, not by
    revalidating the witness.
    """

    def __init__(self, revisions: list[Revision], state: State):
        self.revisions = revisions
        self.index: dict[int, list[int]] = {}
        for i, r in enumerate(revisions):
            for c in (*r.deps, EXT):
                self.index.setdefault(c, []).append(i)
        self._all = tuple(range(len(revisions)))
        self._affected: dict[frozenset[int], tuple[int, ...]] = {}
        self.revalidations = 0
        self._refresh(state)

    def _refresh(self, state: State) -> None:
        self.eta = [recovery_duration(r, state) for r in self.revisions]
        self.version = state.version
        self.revalidations += len(self.revisions)

    def affected(self, action: Action) -> tuple[int, ...]:
        ws = action.write_set
        if ws is None:                       # unanalysable: revalidate everything
            return self._all
        hit = self._affected.get(ws)
        if hit is None:
            found: set[int] = set()
            for c in ws:
                found.update(self.index.get(c, ()))
            hit = self._affected[ws] = tuple(sorted(found))
        return hit

    def evaluate(self, state: State, post: State, action: Action) -> list[int]:
        """Recovery durations in `post`, revalidating only affected witnesses."""
        if state.version != self.version:    # checked state moved under us
            self._refresh(state)
        eta = list(self.eta)
        for i in self.affected(action):
            eta[i] = recovery_duration(self.revisions[i], post)
            self.revalidations += 1
        return eta

    def commit(self, eta: list[int], post: State) -> None:
        self.eta = eta
        self.version = post.version

    def commit_unchecked(self, pre: State, action: Action, post: State) -> None:
        """An action committed without passing through evaluate (e.g. WAIT)."""
        if pre.version != self.version:      # never stamp a stale cache as current
            self._refresh(post)
            return
        for i in self.affected(action):
            self.eta[i] = recovery_duration(self.revisions[i], post)
            self.revalidations += 1
        self.version = post.version
