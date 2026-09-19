"""
Symbolic delegation workflow shared by Studies 2 and 3.

Discrete time. An external irreversible commitment fires at `deadline`.
The state holds three categories of work distinct, because a window
computed as if checkpoint restore reversed external effects is wrong:

  internal      reversible internal work        (restore rolls it back)
  pending       cancellable pending work        (restore cancels it)
  external      irreversible external commitments (restore cannot touch it)

Entanglement is tracked per component. Entanglement created by internal
work lives in `entangle` and can be bought back with PRESERVE.
Entanglement created by an external commitment lives in `ext_entangle`
and can never be bought back or restored away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

EXT = -1  # pseudo-component: entanglement owed to external commitments


class Kind(Enum):
    CLEAN = "clean"          # task +1, eta unchanged, 2 steps
    ENTANGLE = "entangle"    # task +1, eta += delta, 1 step
    PRESERVE = "preserve"    # task +0, eta -= min(eta, rho), 1 step
    UNAUTH = "unauth"        # task +2, unauthorised external shortcut, 1 step
    WAIT = "wait"            # nothing proposed; the clock still runs


_COST = {Kind.CLEAN: 2}
_GAIN = {Kind.CLEAN: 1, Kind.ENTANGLE: 1, Kind.UNAUTH: 2}


@dataclass(frozen=True)
class Action:
    kind: Kind
    component: int | None = None
    amount: int = 0            # delta for ENTANGLE/UNAUTH, rho for PRESERVE
    analysable: bool = True    # False: the action does not declare its writes
    authorised: bool = True

    @property
    def cost(self) -> int:
        return _COST.get(self.kind, 1)

    @property
    def task_gain(self) -> int:
        return _GAIN.get(self.kind, 0)

    @property
    def write_set(self) -> frozenset[int] | None:
        """Components whose entanglement may change. None means undeclared."""
        if not self.analysable:
            return None
        if self.kind in (Kind.ENTANGLE, Kind.PRESERVE):
            return frozenset((self.component,))
        if self.kind is Kind.UNAUTH:
            return frozenset((EXT,))
        return frozenset()


WAIT = Action(Kind.WAIT)
CLEAN = Action(Kind.CLEAN)


@dataclass
class State:
    t: int
    deadline: int
    task_needed: int
    entangle: list[int]
    internal: int = 0
    pending: int = 0
    external: list[tuple[str, int]] = field(default_factory=list)
    ext_entangle: int = 0
    version: int = 0

    def copy(self) -> "State":
        return State(self.t, self.deadline, self.task_needed, list(self.entangle),
                     self.internal, self.pending, list(self.external),
                     self.ext_entangle, self.version)

    def level(self, component: int) -> int:
        return self.ext_entangle if component == EXT else self.entangle[component]

    @property
    def done(self) -> bool:
        return self.internal >= self.task_needed


@dataclass(frozen=True)
class Checkpoint:
    internal: int
    entangle: tuple[int, ...]


def checkpoint(state: State) -> Checkpoint:
    return Checkpoint(state.internal, tuple(state.entangle))


def restore(state: State, cp: Checkpoint) -> State:
    """
    Roll back reversible internal work and cancel pending work.
    Time, external commitments and the entanglement they caused survive.
    """
    s = state.copy()
    s.internal = cp.internal
    s.entangle = list(cp.entangle)
    s.pending = 0
    s.version += 1
    return s


def advance(s: State) -> None:
    """One clock tick, in place. The deadline commitment fires on arrival."""
    s.t += 1
    if s.t == s.deadline:
        s.external.append(("deploy", s.t))


_advance = advance


def apply_effect(s: State, action: Action) -> None:
    """
    Entanglement effects of `action`, in place, without touching the clock
    or task progress. Study 3 needs this: several agents share one tick.
    """
    k = action.kind
    if k is Kind.ENTANGLE:
        s.entangle[action.component] += action.amount
    elif k is Kind.PRESERVE:
        c = action.component
        s.entangle[c] -= min(s.entangle[c], action.amount)
    elif k is Kind.UNAUTH:
        s.external.append(("unauthorised_publish", s.t))
        s.ext_entangle += action.amount
    s.version += 1


def ticks(state: State, action: Action) -> list[State]:
    """States after each clock tick of `action`. The last one is Tr(s, a)."""
    s = state.copy()
    out: list[State] = []
    k = action.kind
    if k is Kind.CLEAN:
        s.pending += 1                  # in flight, still cancellable
        _advance(s)
        out.append(s.copy())
        s.pending -= 1
        s.internal += 1
        _advance(s)
    elif k is Kind.ENTANGLE:
        s.internal += 1
        s.entangle[action.component] += action.amount
        _advance(s)
    elif k is Kind.PRESERVE:
        c = action.component
        s.entangle[c] -= min(s.entangle[c], action.amount)
        _advance(s)
    elif k is Kind.UNAUTH:
        s.internal += action.task_gain
        s.external.append(("unauthorised_publish", s.t))
        s.ext_entangle += action.amount
        _advance(s)
    else:
        _advance(s)
    s.internal = min(s.internal, s.task_needed)
    s.version += 1
    out.append(s)
    return out


def transition(state: State, action: Action) -> State:
    return ticks(state, action)[-1]
