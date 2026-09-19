"""
Study 3: concurrent window erosion.

Two or three agents draw work from one shared backlog and write to one
shared state. Every recovery witness depends on that state, so the
window is a shared resource that no single agent's write footprint
covers (the invariant-confluence point of Bailis et al.).

The baseline is already correct in EVERY condition:
  * permissions are enforced globally at the commit layer and the same
    restrictions are propagated to every agent's own check,
  * every commit is atomic against the CURRENT state and re-checks
    feasibility there, so no action can make a critical revision
    infeasible, in any condition.
This is deliberate. The result must not reduce to subagent restriction
non-propagation (ROGUE, arXiv:2606.00341), nor to a baseline that lacks
transactional correctness. We assert zero unauthorised commits and ~zero
feasibility violations below, and the comparison is void if they fail.

The conditions differ only in where the TIMELINESS check W >= H is made:
  local     each agent checks its own view: shared state as of `lag`
            ticks ago, plus its own writes since (read-your-writes)
  global    each agent checks the current state at the start of the tick
  reserved  checks are serialised against current state plus a ledger of
            reservations on the shared components, through the
            incremental witness cache, with a version-checked ledger
            (an agent whose read version moved must retry)

A tick has two phases. Checks happen at the start, effects commit at the
end. `local` and `global` therefore share a check-to-commit gap in which
another agent can spend the same window; `reserved` closes it.

Repair clause, identical in all conditions: an action that only removes
entanglement (PRESERVE with a non-zero effect) is always admitted, since
it dominates waiting on every window. Without it a violated invariant
would deadlock the filter. It never fires in Study 2.

Trials are the Study 2 trials (same seed, same generator), so k = 1 is a
paired single-agent reference: `global` with one agent reproduces the
Study 2 `timely_full` trajectory exactly.

Outputs: results/study3.json, figures/fig5_violations_vs_staleness.pdf,
         figures/fig6_concurrency_costs.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study2_window_collapse as s2  # noqa: E402
from common.env import CLEAN, Action, Kind, State, advance, apply_effect  # noqa: E402
from common.witness import WitnessCache, recovery_duration  # noqa: E402

SEED = s2.SEED
ROOT = s2.ROOT
REVISIONS, HANDOVER, BASE_H = s2.REVISIONS, s2.HANDOVER, s2.BASE_H
N_COMP = len(s2.COMPONENTS)

AGENTS = [1, 2, 3]
LAGS = [0, 1, 2, 4]
H_VALUES = [1, 2, 4, 8]
CONDITIONS = ["local", "global", "reserved"]
HEADLINE = {"k": 3, "H": 4, "lag": 2}

CELLS = [(c, k, lag, H) for k in AGENTS for H in H_VALUES
         for c in CONDITIONS for lag in (LAGS if c == "local" else [0])]

METRICS = ["viol_any", "viol_ticks", "protect_ticks", "max_depth", "feas_viol_any",
           "unauth_commits", "completion", "throughput", "T_end", "evals",
           "window_rejects", "commit_rejects", "retries", "commits", "blocked",
           "preserve", "agent_ticks", "erosion", "t_bind",
           "succ_continue", "succ_hold", "adv_continue", "adv_hold"]
M = {name: i for i, name in enumerate(METRICS)}


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------
def post_of(view: State, a: Action, horizon: int = 0) -> State:
    """
    State once `a` AND all pending in-flight work have landed. Pending work
    is part of the current state: slack that an in-flight CLEAN still needs
    is not free for another agent to spend.
    """
    p = view.copy()
    apply_effect(p, a)
    p.t = max(view.t + a.cost, horizon)
    return p


def is_repair(view: State, post: State) -> bool:
    return (post.ext_entangle == view.ext_entangle
            and all(b <= a for a, b in zip(view.entangle, post.entangle))
            and post.entangle != view.entangle)


def admissible(view: State, a: Action, threshold: int, horizon: int = 0) -> bool:
    """Full recomputation against `view`. threshold 0 is the feasibility check."""
    if not a.authorised:
        return False
    post = post_of(view, a, horizon)
    budget = post.deadline - post.t - threshold
    for r in REVISIONS:
        if r.critical and recovery_duration(r, post) > budget:
            return is_repair(view, post)
    return True


class Ledger:
    """Reservations on the shared components named in the critical witnesses."""

    def __init__(self):
        self.add = [0] * N_COMP      # entanglement promised but not yet committed
        self.pres = [0] * N_COMP     # entanglement already claimed by a pending PRESERVE
        self.horizon = 0             # when the last reserved in-flight work lands
        self.version = 0

    def clear(self, horizon: int = 0):
        self.add = [0] * N_COMP
        self.pres = [0] * N_COMP
        self.horizon = horizon

    def reserve(self, a: Action, t: int):
        if a.kind is Kind.ENTANGLE:
            self.add[a.component] += a.amount
        elif a.kind is Kind.PRESERVE:
            self.pres[a.component] += a.amount
        elif a.kind is Kind.CLEAN:
            self.horizon = max(self.horizon, t + a.cost)
        else:
            return
        self.version += 1


def reserved_admissible(cur: State, cache: WitnessCache, ledger: Ledger, a: Action, H: int) -> bool:
    if not a.authorised:
        return False
    post = post_of(cur, a, ledger.horizon)
    eta = cache.evaluate(cur, post, a)                  # incremental maintenance
    assert eta == [recovery_duration(r, post) for r in REVISIONS], "witness cache diverged"
    budget = post.deadline - post.t - H
    for e, r in zip(eta, REVISIONS):
        if r.critical and e + sum(ledger.add[c] for c in r.deps) > budget:
            return a.kind is Kind.PRESERVE and a.amount > 0
    return True


# ----------------------------------------------------------------------
# One concurrent trajectory
# ----------------------------------------------------------------------
def run_concurrent(trial: s2.Trial, cond: str, k: int, lag: int, H: int, orders: np.ndarray):
    cur = trial.initial_state()
    D, N, rho = trial.D, trial.N, trial.rho
    queue = list(range(N))
    done = 0
    busy_until = [0] * k
    hist = [list(cur.entangle)]
    hist_inflight = [0]              # when in-flight work lands, as seen at each tick start
    own: list[dict[int, tuple[int, int]]] = [{} for _ in range(k)]
    eta = np.empty((len(REVISIONS), D + 1), dtype=np.int64)
    eta[:, 0] = [recovery_duration(r, cur) for r in REVISIONS]
    cache = WitnessCache(REVISIONS, cur) if cond == "reserved" else None
    ledger = Ledger()
    n = dict.fromkeys(["evals", "window_rejects", "commit_rejects", "retries", "commits",
                       "blocked", "preserve", "unauth_commits"], 0)
    t_bind = -1

    def productive(unit: int, t: int) -> list[Action]:
        c, d, analysable, unauth = trial.units[unit]
        out = [Action(Kind.UNAUTH, amount=2 * d, authorised=False)] if unauth else []
        out.append(Action(Kind.ENTANGLE, component=c, amount=d, analysable=analysable))
        if t + CLEAN.cost <= D:
            out.append(CLEAN)
        return out

    def local_view(i: int, t: int) -> State:
        t0 = max(0, t - lag)
        ent = list(hist[t0])
        for tau in range(t0, t):
            w = own[i].get(tau)
            if w is not None:
                ent[w[0]] = max(0, ent[w[0]] + w[1])
        return State(t=t, deadline=D, task_needed=N, entangle=ent)

    def select(cands: list[Action], view: State, check, live: bool = True):
        """First admitted candidate in preference order, then PRESERVE."""
        nonlocal t_bind
        for idx, a in enumerate(cands):
            n["evals"] += 1
            if check(a):
                return a, idx
            if live and a.authorised:
                n["window_rejects"] += 1
                if t_bind < 0:
                    t_bind = view.t
        p = s2.preserve_candidate(view, rho)
        if p is not None:
            n["evals"] += 1
            if check(p):
                return p, len(cands)
            if live:
                n["window_rejects"] += 1
        return None, len(cands) + 1

    def ledger_view() -> State:
        v = cur.copy()
        v.entangle = [max(0, e - p) for e, p in zip(cur.entangle, ledger.pres)]
        return v

    def cap(a: Action) -> Action:
        """A PRESERVE can only claim entanglement nobody else has claimed."""
        if a.kind is not Kind.PRESERVE:
            return a
        free = max(0, cur.entangle[a.component] - ledger.pres[a.component])
        return Action(Kind.PRESERVE, component=a.component, amount=min(a.amount, free))

    while cur.t < D and done < N:
        t = cur.t
        free_agents = [int(i) for i in orders[t] if busy_until[i] <= t]
        unit_of = {i: (queue[j] if j < len(queue) else None) for j, i in enumerate(free_agents)}
        chosen: dict[int, tuple[Action | None, int, list[Action]]] = {}

        # ---- phase 1: checks at the start of the tick -------------------
        inflight = max(busy_until)
        if cache is not None:
            ledger.clear(inflight)
        read_version = ledger.version
        for i in free_agents:
            if unit_of[i] is None:
                continue
            cands = productive(unit_of[i], t)
            if cache is None:
                fresh = cond == "global" or lag == 0
                view = cur if fresh else local_view(i, t)
                seen = inflight if fresh else hist_inflight[max(0, t - lag)]
                a, idx = select(cands, view, lambda a, v=view, h=seen: admissible(v, a, H, h))
            else:
                if ledger.version != read_version:       # somebody reserved first: retry
                    n["retries"] += 1
                    empty = Ledger()
                    empty.horizon = inflight
                    select(cands, cur, lambda a: reserved_admissible(cur, cache, empty, a, H),
                           live=False)                   # the wasted optimistic pass
                a, idx = select(cands, ledger_view(),
                                lambda a: reserved_admissible(cur, cache, ledger, cap(a), H))
                if a is not None:
                    a = cap(a)
                    ledger.reserve(a, t)
            chosen[i] = (a, idx, cands)

        # ---- phase 2: atomic commits at the end of the tick -------------
        for i in free_agents:
            if i not in chosen:
                continue
            a, idx, cands = chosen[i]
            inflight = max(busy_until)                   # includes commits earlier this tick
            if a is not None and cache is None and not admissible(cur, a, 0, inflight):
                n["commit_rejects"] += 1                 # feasibility, against current state
                a = None
                for b in cands[idx + 1:]:                # the rejection refreshes the view
                    n["evals"] += 1
                    if admissible(cur, b, H, inflight):
                        a = b
                        break
                if a is None:
                    p = s2.preserve_candidate(cur, rho)
                    if p is not None:
                        n["evals"] += 1
                        a = p if admissible(cur, p, H, inflight) else None
            if a is None:
                n["blocked"] += 1
                continue
            post = cur.copy()
            apply_effect(post, a)
            if cache is not None:
                cache.commit_unchecked(cur, a, post)
            n["commits"] += 1
            if a.kind is Kind.ENTANGLE:
                own[i][t] = (a.component, a.amount)
                queue.remove(unit_of[i])
                done += 1
            elif a.kind is Kind.CLEAN:
                queue.remove(unit_of[i])
                busy_until[i] = t + CLEAN.cost
            elif a.kind is Kind.PRESERVE:
                own[i][t] = (a.component, post.entangle[a.component] - cur.entangle[a.component])
                n["preserve"] += 1
            elif a.kind is Kind.UNAUTH:
                n["unauth_commits"] += 1
            cur = post

        advance(cur)
        done += sum(1 for i in range(k) if busy_until[i] == cur.t and busy_until[i] > 0)
        eta[:, cur.t] = [recovery_duration(r, cur) for r in REVISIONS]
        hist.append(list(cur.entangle))
        hist_inflight.append(max(busy_until))

    T_end = cur.t
    eta[:, T_end + 1:] = eta[:, [T_end]]
    return T_end, min(done, N) / N, eta, n, t_bind


def measure(trial: s2.Trial, cond: str, k: int, lag: int, H: int, orders: np.ndarray) -> list[float]:
    T_end, completion, eta, n, t_bind = run_concurrent(trial, cond, k, lag, H, orders)
    D = trial.D
    W = D - np.arange(D + 1) - eta                       # (revisions, D + 1)
    last = min(T_end, D - BASE_H - H)                    # invariant is satisfiable up to here
    below = (W[:, :last + 1] < H).any(axis=0)
    depth = int(max(0, (H - W[:, :last + 1]).max()))
    feas_last = min(T_end, D - BASE_H)
    tb = t_bind if t_bind > 0 else T_end
    sc = s2.score(SimpleNamespace(T_end=T_end, eta=eta), trial, H)
    out = dict(
        viol_any=below.any(), viol_ticks=below.sum(), protect_ticks=last + 1, max_depth=depth,
        feas_viol_any=(W[:, :feas_last + 1] < 0).any(), unauth_commits=n["unauth_commits"],
        completion=completion, throughput=completion * trial.N / T_end, T_end=T_end,
        evals=n["evals"], window_rejects=n["window_rejects"], commit_rejects=n["commit_rejects"],
        retries=n["retries"], commits=n["commits"], blocked=n["blocked"], preserve=n["preserve"],
        agent_ticks=k * T_end, erosion=(W[HANDOVER, 0] - W[HANDOVER, tb]) / tb, t_bind=tb,
        succ_continue=sc["continue"]["uniform"][HANDOVER], succ_hold=sc["hold"]["uniform"][HANDOVER],
        adv_continue=sc["continue"]["adversarial"], adv_hold=sc["hold"]["adversarial"])
    return [float(out[m]) for m in METRICS]


def run_trial(i: int) -> np.ndarray:
    trial = s2.make_trial(i)
    orders = {k: np.random.default_rng([SEED, i, 3, k]).permuted(
        np.tile(np.arange(k), (trial.D, 1)), axis=1) for k in AGENTS}
    return np.array([measure(trial, c, k, lag, H, orders[k]) for c, k, lag, H in CELLS])


# ----------------------------------------------------------------------
def wilson_rate(x: np.ndarray) -> dict:
    return {"rate": float(x.mean()), "ci95": s2.wilson(int(x.sum()), x.size)}


def summarise(x: np.ndarray) -> dict:
    g = lambda name: x[:, M[name]]
    return {
        "timeliness_violation": {
            "trials_with_any": wilson_rate(g("viol_any")),
            "tick_rate": float(g("viol_ticks").sum() / g("protect_ticks").sum()),
            "mean_max_depth_when_violated": float(g("max_depth")[g("viol_any") > 0].mean())
            if g("viol_any").any() else 0.0,
        },
        "feasibility_violation": {"trials_with_any": float(g("feas_viol_any").mean())},
        "unauthorised_commits": int(g("unauth_commits").sum()),
        "revision_success_uniform": {"continue": float(g("succ_continue").mean()),
                                     "hold": float(g("succ_hold").mean())},
        "revision_success_adversarial": {"continue": float(g("adv_continue").mean()),
                                         "hold": float(g("adv_hold").mean())},
        "task_completion_mean": float(g("completion").mean()),
        "throughput_units_per_tick": float(g("throughput").mean()),
        "delegation_length_mean": float(g("T_end").mean()),
        "window_erosion": {"per_tick_until_binding": float(g("erosion").mean()),
                           "ticks_until_binding": float(g("t_bind").mean())},
        "blocking": {
            "window_rejects_per_eval": float(g("window_rejects").sum() / g("evals").sum()),
            "commit_rejects_per_commit": float(g("commit_rejects").sum() / g("commits").sum()),
            "ledger_retries_per_commit": float(g("retries").sum() / g("commits").sum()),
            "blocked_agent_tick_fraction": float(g("blocked").sum() / g("agent_ticks").sum()),
            "preserve_agent_tick_fraction": float(g("preserve").sum() / g("agent_ticks").sum()),
            "filter_evals_per_commit": float(g("evals").sum() / g("commits").sum()),
        },
    }


def key(c: str, k: int, lag: int, H: int) -> str:
    return f"{c}|k={k}|lag={lag}|H={H}"


def run(n_trials: int, workers: int) -> dict:
    t0 = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            data = np.stack(list(pool.map(run_trial, range(n_trials), chunksize=20)))
    else:
        data = np.stack([run_trial(i) for i in range(n_trials)])
    elapsed = time.perf_counter() - t0                   # data: (trials, cells, metrics)
    col = {cell: data[:, j, :] for j, cell in enumerate(CELLS)}
    cells = {key(*cell): summarise(x) for cell, x in col.items()}

    # ---- assertions (task file) ---------------------------------------
    checks: dict[str, object] = {}
    worst_feas = max(c["feasibility_violation"]["trials_with_any"] for c in cells.values())
    assert worst_feas <= 1e-3, f"baseline too weak: feasibility violated in {worst_feas:.4f} of trials"
    checks["max_feasibility_violation_rate_any_cell"] = worst_feas
    unauth = sum(c["unauthorised_commits"] for c in cells.values())
    assert unauth == 0, "a restriction failed to propagate: comparison invalid"
    checks["unauthorised_commits_total"] = unauth

    viol = lambda c, k, lag, H: cells[key(c, k, lag, H)]["timeliness_violation"]["trials_with_any"]["rate"]
    single = max(viol(c, 1, lag, H) for c, k, lag, H in CELLS if k == 1)
    assert single == 0.0, "a single agent violated timeliness: filter bug, not a concurrency effect"
    checks["single_agent_violation_rate_max"] = single
    reserved_max = max(viol("reserved", k, 0, H) for k in AGENTS for H in H_VALUES)
    assert reserved_max == 0.0, "reservation admitted a violation"
    checks["reserved_violation_rate_max"] = reserved_max
    for k in AGENTS:
        for H in H_VALUES:
            a, b = col["local", k, 0, H], col["global", k, 0, H]
            assert np.array_equal(a, b), "local with zero lag must equal global"
    checks["local_lag0_equals_global"] = True

    # ---- paired costs at the headline cell ----------------------------
    hk, hH, hlag = HEADLINE["k"], HEADLINE["H"], HEADLINE["lag"]
    thr = lambda c, lag=0, k=hk: col[c, k, lag, hH][:, M["throughput"]]

    def paired_loss(base: np.ndarray, other: np.ndarray) -> dict:
        """Relative throughput loss as a ratio of means; CI from the paired differences."""
        d = base - other
        half = 1.959964 * d.std(ddof=1) / np.sqrt(d.size)
        return {"relative": float(d.mean() / base.mean()),
                "ci95": [float((d.mean() - half) / base.mean()), float((d.mean() + half) / base.mean())],
                "units_per_tick": float(d.mean()),
                "mean_of_trial_ratios": float((d / base).mean())}   # skewed by stalled trials

    evals = lambda c, lag=0: cells[key(c, hk, lag, hH)]["blocking"]["filter_evals_per_commit"]

    erosion = {f"k={k}": cells[key("global", k, 0, hH)]["window_erosion"] for k in AGENTS}
    e1 = erosion["k=1"]["per_tick_until_binding"]
    for k in AGENTS:
        erosion[f"k={k}"]["ratio_to_single_agent"] = erosion[f"k={k}"]["per_tick_until_binding"] / e1

    return {
        "seed": SEED,
        "n_trials_per_cell": n_trials,
        "elapsed_s": elapsed,
        "design": {
            "agents": AGENTS, "lags": LAGS, "H_values": H_VALUES, "conditions": CONDITIONS,
            "headline_cell": HEADLINE,
            "trials": "identical to Study 2 (same seed and generator); the backlog is shared",
            "violation": "some tick t <= min(T_end, D - base_handover - H) has W(s_t, r) < H "
                         "for a critical r",
            "baseline": "all conditions: global permission enforcement, restrictions propagated to "
                        "every agent, atomic commit with a feasibility re-check on current state",
        },
        "cells": cells,
        "window_erosion_vs_single_agent": erosion,
        "throughput_loss_reserved_vs_global": paired_loss(thr("global"), thr("reserved")),
        "throughput_loss_reserved_vs_local": paired_loss(thr("local", hlag), thr("reserved")),
        "sanity_checks": checks,
        "emit": {
            "study3.local_violation_rate": viol("local", hk, hlag, hH),
            "study3.global_violation_rate": viol("global", hk, 0, hH),
            "study3.reserved_violation_rate": viol("reserved", hk, 0, hH),
            "study3.blocking_overhead": paired_loss(thr("global"), thr("reserved"))["relative"],
            "study3.check_overhead_evals_reserved_over_global": evals("reserved") / evals("global"),
            "study3.ledger_retries_per_commit":
                cells[key("reserved", hk, 0, hH)]["blocking"]["ledger_retries_per_commit"],
            "study3.local_violation_tick_rate":
                cells[key("local", hk, hlag, hH)]["timeliness_violation"]["tick_rate"],
            "study3.global_violation_tick_rate":
                cells[key("global", hk, 0, hH)]["timeliness_violation"]["tick_rate"],
            "study3.erosion_ratio_k3_over_k1": erosion["k=3"]["ratio_to_single_agent"],
        },
    }

# ----------------------------------------------------------------------
# Figures. Same validated Okabe-Ito subset and conventions as Study 2:
# blue is the mechanism that maintains the window; marker and line style
# repeat the identity so the figures survive greyscale.
# ----------------------------------------------------------------------
INK, MUTED, GRID = s2.INK, s2.MUTED, s2.GRID


def make_figures(results: dict, outdir: Path, ext: str = "pdf") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 8.5})
    cells, H = results["cells"], results["design"]["headline_cell"]["H"]
    lags = results["design"]["lags"]
    outdir.mkdir(exist_ok=True)
    get = lambda c, k, lag: cells[key(c, k, lag, H)]

    def style(ax, xlabel):
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel)

    # Fig 5: violations against observation staleness. lag 0 IS the global check.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    per_agent = [(2, "per-agent check, 2 agents", "#CC79A7", "^", "--"),
                 (3, "per-agent check, 3 agents", "#D55E00", "s", "-")]
    panels = [("trials with a violation (fraction)",
               lambda x: x["timeliness_violation"]["trials_with_any"]),
              ("ticks in violation (fraction)",
               lambda x: {"rate": x["timeliness_violation"]["tick_rate"]})]
    for ax, (ylabel, pick) in zip(axes, panels):
        for k, label, colour, marker, ls in per_agent:
            q = [pick(get("global" if lag == 0 else "local", k, lag)) for lag in lags]
            y = [v["rate"] for v in q]
            err = None
            if "ci95" in q[0]:
                err = [[v["rate"] - v["ci95"][0] for v in q], [v["ci95"][1] - v["rate"] for v in q]]
            ax.errorbar(lags, y, yerr=err, color=colour, marker=marker, ms=4.5, lw=1.4, ls=ls,
                        elinewidth=0.8, label=label, clip_on=False, zorder=3)
        ax.plot(lags, [0.0] * len(lags), color="#0072B2", marker="o", ms=4.5, lw=1.4,
                label="reserved, 2 or 3 agents", clip_on=False, zorder=4)
        ax.set_xticks(lags)
        ax.set_xticklabels(["0\n(global)"] + [str(v) for v in lags[1:]])
        ax.set_ylim(bottom=-0.02 * ax.get_ylim()[1])
        ax.set_ylabel(ylabel)
        style(ax, "staleness of the agent's view (ticks)")
    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda j: labels[j].startswith("reserved"))
    fig.legend([handles[j] for j in order], [labels[j] for j in order], loc="upper center", ncol=3,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02), handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.5)
    fig.savefig(outdir / f"fig5_violations_vs_staleness.{ext}", dpi=200)
    plt.close(fig)

    # Fig 6: what concurrency costs. Three measures, three axes, never combined.
    agents = results["design"]["agents"]
    hlag = results["design"]["headline_cell"]["lag"]
    series = [("local", hlag, f"local (lag {hlag})", "#D55E00", "s", "-"),
              ("global", 0, "global", "#CC79A7", "^", "--"),
              ("reserved", 0, "reserved", "#0072B2", "o", "-")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    ero = results["window_erosion_vs_single_agent"]
    y = [ero[f"k={k}"]["per_tick_until_binding"] for k in agents]
    axes[0].plot(agents, y, color=INK, marker="D", ms=4.5, lw=1.4, clip_on=False, zorder=3)
    for k, yi in zip(agents, y):
        axes[0].annotate(f"{yi:.2f}", (k, yi), xytext=(7, -9), textcoords="offset points",
                         ha="left", fontsize=7.5, color=MUTED)
    axes[0].set_ylabel("window lost per tick, until binding")
    axes[0].set_ylim(0, max(y) * 1.2)
    for ax, ylabel, pick in (
            (axes[1], "throughput (units per tick)", lambda x: x["throughput_units_per_tick"]),
            (axes[2], "filter evaluations per commit", lambda x: x["blocking"]["filter_evals_per_commit"])):
        for c, lag, label, colour, marker, ls in series:
            ax.plot(agents, [pick(get(c, k, lag)) for k in agents], color=colour, marker=marker,
                    ms=4.5, lw=1.4, ls=ls, label=label, clip_on=False, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    for ax in axes:
        ax.set_xticks(agents)
        ax.set_xlim(0.8, 3.2)
        style(ax, "agents")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.62, 1.02), handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.2)
    fig.savefig(outdir / f"fig6_concurrency_costs.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "study3.json")
    args = ap.parse_args()
    results = run(args.trials, args.workers)
    args.out.parent.mkdir(exist_ok=True)
    with open(args.out, "w", newline="\n") as f:
        json.dump(results, f, indent=2)
    make_figures(results, ROOT / "figures")
    print(json.dumps({k: results[k] for k in ("emit", "sanity_checks", "window_erosion_vs_single_agent",
                                              "throughput_loss_reserved_vs_global")}, indent=2))
    print(f"{args.trials} trials per cell in {results['elapsed_s']:.1f}s -> {args.out}")
