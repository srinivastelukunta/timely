"""
Study 2: window collapse under single-agent delegation.

Demonstrates window collapse empirically and measures the cost of the timely
filter against a strong feasibility-filter baseline.

Design decisions that are not in the task file, stated once here:

* Every filter is a pure rejector over the same task-greedy agent. When
  nothing productive is admitted the agent buys window (PRESERVE) so it
  can keep working. When nothing at all is admitted it is blocked and the
  clock runs. No condition can compel an action or stop the clock.
* `feas` is `timely_full` with H = 0. Same machinery, different threshold.
* Requests arrive while the delegation is active, and only at times the
  ideal policy could still honour: t_req < min(T_end, D - base - H_MAX).
* The response delay is scored under two semantics, from the SAME
  trajectory, because the task specification and the formal construction
  disagree about what the agent does while the human responds:
    continue  the agent keeps working under its filter (task file formula:
              eta(s_{t_req+H}) <= D - (t_req+H) on the realised trajectory)
    hold      the agent is frozen at t_req and only the clock runs
              (the hold policy: success iff W(s_{t_req}) >= H)
  Both are emitted. Neither is folded into the other.
* All protected revisions are critical. Their dependency sets nest inside
  handover's, so handover is always the binding window and the invariant
  is maintainable. The smaller witnesses exist so that incremental
  maintenance has something to skip.
* `none` differs from `perm` through one unauthorised action: an external
  shortcut that publishes irreversibly. Its entanglement cannot be bought
  back and checkpoint restore cannot reverse it.

Outputs: results/study2.json, figures/fig3_success_vs_H.pdf,
         figures/fig4_completion_vs_success.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.env import CLEAN, WAIT, Action, Kind, State, ticks  # noqa: E402
from common.filters import TimelyFullFilter, make_filter  # noqa: E402
from common.witness import Revision, recovery_duration, window  # noqa: E402

SEED = 20260919
ROOT = Path(__file__).resolve().parents[1]

H_VALUES = [0, 1, 2, 4, 8]
H_MAX = max(H_VALUES)
CONDITIONS = ["none", "perm", "feas", "timely_full", "timely_inc"]
H_INDEPENDENT = {"none", "perm", "feas"}     # their trajectories ignore H

# components 6 and 7 are free: no recovery witness depends on them
COMPONENTS = ["deps", "config", "state_format", "interface", "credentials",
              "docs", "build_cache", "telemetry"]
REVISIONS = [
    Revision("handover", 6, frozenset({0, 1, 2, 3, 4, 5})),
    Revision("run_local", 3, frozenset({0, 1, 4})),
    Revision("export_state", 2, frozenset({2})),
    Revision("change_objective", 2, frozenset({3, 5})),
    Revision("rotate_credentials", 1, frozenset({4})),
    Revision("audit_trace", 1, frozenset({5})),
]
HANDOVER = 0
BASE_H = REVISIONS[HANDOVER].base
for _r in REVISIONS[1:]:                      # nesting is what keeps handover binding
    assert _r.deps <= REVISIONS[HANDOVER].deps and _r.base <= BASE_H

P_OPAQUE = 0.05      # shortcut does not declare its write set
P_UNAUTH = 0.15      # an unauthorised external shortcut is on offer


# ----------------------------------------------------------------------
# Trials: everything random is drawn here, once, and shared by conditions
# ----------------------------------------------------------------------
@dataclass
class Trial:
    idx: int
    N: int
    D: int
    rho: int
    entangle0: list[int]
    units: list[tuple[int, int, bool, bool]]   # component, delta, analysable, unauth
    u: float                                   # request-time quantile

    def initial_state(self) -> State:
        return State(t=0, deadline=self.D, task_needed=self.N,
                     entangle=list(self.entangle0))


def make_trial(i: int) -> Trial:
    rng = np.random.default_rng([SEED, i])
    N = int(rng.integers(30, 51))
    D = BASE_H + int(round(N * rng.uniform(1.5, 2.6)))
    rho = int(rng.integers(1, 5))
    entangle0 = rng.integers(0, 3, size=len(COMPONENTS)).tolist()
    comp = rng.integers(0, len(COMPONENTS), size=N)
    delta = rng.integers(1, 4, size=N)
    opaque = rng.random(N) < P_OPAQUE
    unauth = rng.random(N) < P_UNAUTH
    units = [(int(comp[k]), int(delta[k]), not bool(opaque[k]), bool(unauth[k]))
             for k in range(N)]
    return Trial(i, N, D, rho, entangle0, units, float(rng.random()))


# ----------------------------------------------------------------------
# Agent: maximises task progress per step among admitted actions
# ----------------------------------------------------------------------
def preserve_candidate(state: State, rho: int) -> Action | None:
    binding = min(REVISIONS, key=lambda r: window(r, state))
    c = max(binding.deps, key=lambda j: (state.entangle[j], -j))
    return Action(Kind.PRESERVE, component=c, amount=rho) if state.entangle[c] else None


def proposals(state: State, trial: Trial, agent: str) -> list[Action]:
    out: list[Action] = []
    if agent == "greedy":
        c, d, analysable, unauth = trial.units[state.internal]
        if unauth:
            out.append(Action(Kind.UNAUTH, amount=2 * d, authorised=False))   # 2 / step
        out.append(Action(Kind.ENTANGLE, component=c, amount=d,
                          analysable=analysable))                            # 1 / step
        out.append(CLEAN)                                                     # 1/2 / step
    p = preserve_candidate(state, trial.rho)                                  # 0 / step
    if p is not None:
        out.append(p)
    return out


@dataclass
class Traj:
    T_end: int
    completion: float
    eta: np.ndarray            # (n_revisions, D + 1) recovery duration at each t
    evals: int
    rejects: int
    revalidations: int
    wall_ns: int
    blocked_steps: int
    preserve_steps: int
    inc_unsound: int           # inc admitted, full rejected  (must be 0)
    inc_overblock: int         # inc rejected, full admitted


def run_trajectory(trial: Trial, cond: str, H: int, agent: str = "greedy") -> Traj:
    state = trial.initial_state()
    D = trial.D
    filt = make_filter(cond, REVISIONS, H, state)
    shadow = TimelyFullFilter(REVISIONS, H) if cond == "timely_inc" else None
    eta = np.empty((len(REVISIONS), D + 1), dtype=np.int64)
    eta[:, 0] = [recovery_duration(r, state) for r in REVISIONS]
    blocked = preserved = unsound = overblock = 0

    while state.t < D and not state.done:
        chosen = None
        for a in proposals(state, trial, agent):
            if state.t + a.cost > D:
                continue
            ok = filt.admits(state, a)
            if shadow is not None:             # ground truth, outside the timed path
                truth = shadow.admits(state, a)
                unsound += ok and not truth
                overblock += truth and not ok
            if ok:
                chosen = a
                break
        if chosen is None:
            chosen = WAIT
            blocked += 1
        preserved += chosen.kind is Kind.PRESERVE
        for s in ticks(state, chosen):
            eta[:, s.t] = [recovery_duration(r, s) for r in REVISIONS]
        filt.notify_commit(state, chosen, s)
        state = s

    eta[:, state.t + 1:] = eta[:, [state.t]]   # delegation over: only the clock runs
    return Traj(state.t, state.internal / trial.N, eta, filt.evals, filt.rejects,
                filt.revalidations, filt.wall_ns, blocked, preserved, unsound, overblock)


# ----------------------------------------------------------------------
# Request scoring. One trajectory, every request time, both semantics.
# ----------------------------------------------------------------------
def score(traj: Traj, trial: Trial, H: int) -> dict:
    D = trial.D
    T = min(traj.T_end, D - BASE_H - H_MAX)      # requests the ideal policy can honour
    t = np.arange(T)
    W = D - t - traj.eta[:, t]                                  # window at request
    ok = {"continue": (D - (t + H) - traj.eta[:, t + H]) >= 0,  # still feasible at t+H
          "hold": W >= H}                                       # frozen agent, clock runs
    tu = int(trial.u * T)
    out = {"W_uniform": int(W[HANDOVER, tu]),
           # response lands after the agent has finished and stopped maintaining anything
           "lands_after_end": bool(tu + H > traj.T_end)}
    for sem, m in ok.items():
        margin = (D - (t + H) - traj.eta[HANDOVER, t + H]) if sem == "continue" \
            else W[HANDOVER] - H
        ta = int(np.argmin(margin))              # the step maximising failure
        out[sem] = {
            "uniform": m[:, tu].copy(),
            "exact": float(m[HANDOVER].mean()),
            "adversarial": bool(margin[ta] >= 0),
            "W_adversarial": int(W[HANDOVER, ta]),
        }
    return out


def wilson(k: int, n: int) -> list[float]:
    z = 1.959964
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(mid - half), float(mid + half)]


def rate(x) -> dict:
    x = np.asarray(x, dtype=bool)
    return {"success": float(x.mean()), "ci95": wilson(int(x.sum()), x.size)}


def summarise(trajs: list[Traj], scores: list[dict]) -> dict:
    n = len(trajs)
    comp = np.array([tr.completion for tr in trajs])
    evals = sum(tr.evals for tr in trajs)
    steps = sum(tr.T_end for tr in trajs)
    cell = {
        "task_completion": {"mean": float(comp.mean()),
                            "fraction_fully_complete": float((comp >= 1).mean())},
        "delegation_length_mean": steps / n,
        "blocking": {
            "rejected_per_proposal": (sum(tr.rejects for tr in trajs) / evals) if evals else 0.0,
            "blocked_step_fraction": sum(tr.blocked_steps for tr in trajs) / steps,
            "preserve_step_fraction": sum(tr.preserve_steps for tr in trajs) / steps,
        },
        "overhead": {
            "filter_evals_per_trial": evals / n,
            "witness_revalidations_per_trial": sum(tr.revalidations for tr in trajs) / n,
            "wall_us_per_trial": sum(tr.wall_ns for tr in trajs) / n / 1e3,
        },
        "requests": {},
    }
    Wu = np.array([s["W_uniform"] for s in scores])
    for sem in ("continue", "hold"):
        uni = np.stack([s[sem]["uniform"] for s in scores])        # (n, n_revisions)
        Wa = np.array([s[sem]["W_adversarial"] for s in scores])
        cell["requests"][sem] = {
            "uniform": {**rate(uni[:, HANDOVER]),
                        "per_revision": {r.name: float(uni[:, i].mean())
                                         for i, r in enumerate(REVISIONS)},
                        "W_at_request": {"mean": float(Wu.mean()),
                                         "p05": float(np.percentile(Wu, 5))}},
            "uniform_exact_mean": float(np.mean([s[sem]["exact"] for s in scores])),
            "uniform_failures": int((~uni[:, HANDOVER]).sum()),
            "uniform_failures_landing_after_delegation_end": int(
                sum((not s[sem]["uniform"][HANDOVER]) and s["lands_after_end"] for s in scores)),
            "adversarial": {**rate([s[sem]["adversarial"] for s in scores]),
                            "W_at_request": {"mean": float(Wa.mean()),
                                             "p05": float(np.percentile(Wa, 5))}},
        }
    return cell


# ----------------------------------------------------------------------
def run(n_trials: int) -> dict:
    trials = [make_trial(i) for i in range(n_trials)]
    trajs: dict[tuple[str, int], list[Traj]] = {(c, H): [] for c in CONDITIONS for H in H_VALUES}
    noop: list[Traj] = []
    t0 = time.perf_counter()
    for trial in trials:
        for cond in CONDITIONS:
            shared = run_trajectory(trial, cond, 0) if cond in H_INDEPENDENT else None
            for H in H_VALUES:
                trajs[cond, H].append(shared or run_trajectory(trial, cond, H))
        noop.append(run_trajectory(trial, "none", 0, agent="noop"))
    elapsed = time.perf_counter() - t0

    scores = {key: [score(tr, trial, key[1]) for tr, trial in zip(v, trials)]
              for key, v in trajs.items()}
    cells = {c: {f"H={H}": summarise(trajs[c, H], scores[c, H]) for H in H_VALUES}
             for c in CONDITIONS}
    reference_noop = {f"H={H}": summarise(noop, [score(tr, trial, H) for tr, trial in zip(noop, trials)])
                      for H in H_VALUES}

    # ---- sanity checks (task file) -----------------------------------
    checks: dict[str, object] = {}

    # feas never violates feasibility while the delegation is active and protectable
    def min_window_in_delegation(tr: Traj, trial: Trial) -> int:
        last = min(tr.T_end, trial.D - BASE_H)
        return int((trial.D - np.arange(last + 1) - tr.eta[:, :last + 1]).min())

    worst = min(min_window_in_delegation(tr, trial) for tr, trial in zip(trajs["feas", 0], trials))
    assert worst >= 0, f"feas violated feasibility: min window {worst}"
    checks["feas_min_window_during_delegation"] = worst

    def succ(c, H, sem="continue", mode="uniform"):
        return cells[c][f"H={H}"]["requests"][sem][mode]["success"]

    assert succ("feas", 0) >= 0.99 and succ("feas", 0, "hold") >= 0.99
    checks["feas_success_H0"] = succ("feas", 0)

    for sem in ("continue", "hold"):
        for mode in ("uniform", "adversarial"):
            curve = [succ("feas", H, sem, mode) for H in H_VALUES]
            falls = all(b <= a + 0.01 for a, b in zip(curve, curve[1:])) and curve[-1] < curve[0]
            checks[f"feas_success_falls_with_H[{sem},{mode}]"] = {"curve": curve, "falls": falls}
            for c in ("timely_full", "timely_inc"):
                worst_t = min(succ(c, H, sem, mode) for H in H_VALUES)
                assert worst_t >= 0.999, f"{c} is buggy: success {worst_t} under {sem}/{mode}"
    checks["timely_min_success_any_cell"] = min(
        succ(c, H, sem, mode) for c in ("timely_full", "timely_inc") for H in H_VALUES
        for sem in ("continue", "hold") for mode in ("uniform", "adversarial"))

    unsound = sum(tr.inc_unsound for H in H_VALUES for tr in trajs["timely_inc", H])
    overblock = sum(tr.inc_overblock for H in H_VALUES for tr in trajs["timely_inc", H])
    assert unsound == 0, "timely_inc admitted an action timely_full rejects: dependency analysis unsound"
    checks["inc_admits_what_full_rejects"] = unsound
    checks["inc_rejects_what_full_admits"] = overblock

    # ---- paired task-completion cost of the timely filter ------------
    feas_comp = np.array([tr.completion for tr in trajs["feas", 0]])
    delta = {}
    for H in H_VALUES:
        d = np.array([tr.completion for tr in trajs["timely_inc", H]]) - feas_comp
        half = 1.959964 * d.std(ddof=1) / np.sqrt(d.size)
        delta[f"H={H}"] = {"mean": float(d.mean()),
                           "ci95": [float(d.mean() - half), float(d.mean() + half)]}

    # ---- decision criterion: is the algorithm a contribution? --------
    crit = {}
    for H in H_VALUES:
        f, i = cells["timely_full"][f"H={H}"], cells["timely_inc"][f"H={H}"]
        crit[f"H={H}"] = {
            "success_full": f["requests"]["continue"]["uniform"]["success"],
            "success_inc": i["requests"]["continue"]["uniform"]["success"],
            "completion_full": f["task_completion"]["mean"],
            "completion_inc": i["task_completion"]["mean"],
            "revalidations_inc_over_full": i["overhead"]["witness_revalidations_per_trial"]
            / f["overhead"]["witness_revalidations_per_trial"],
            "wall_inc_over_full": i["overhead"]["wall_us_per_trial"] / f["overhead"]["wall_us_per_trial"],
            # modelled, not measured: the per-revalidation cost above which inc is cheaper
            "breakeven_us_per_revalidation": max(0.0, (
                i["overhead"]["wall_us_per_trial"] - f["overhead"]["wall_us_per_trial"]) / (
                f["overhead"]["witness_revalidations_per_trial"]
                - i["overhead"]["witness_revalidations_per_trial"])),
        }

    return {
        "seed": SEED,
        "n_trials_per_cell": n_trials,
        "elapsed_s": elapsed,
        "design": {
            "H_values": H_VALUES,
            "conditions": CONDITIONS,
            "request_period": "t_req < min(T_end, D - base_handover - max(H))",
            "delay_semantics": {
                "continue": "agent keeps working under its filter; success iff "
                            "eta(s_{t_req+H}) <= D - (t_req+H) on the realised trajectory",
                "hold": "agent frozen at t_req, only the clock runs; success iff W(s_{t_req}) >= H",
            },
            "trial_distribution": {
                "N": "U{30..50}", "D": "base_handover + round(N * U(1.5, 2.6))",
                "rho": "U{1..4}", "delta": "U{1..3}", "initial_entanglement": "U{0..2} per component",
                "p_opaque_write_set": P_OPAQUE, "p_unauthorised_offer": P_UNAUTH,
            },
            "revisions": {r.name: {"base": r.base, "deps": sorted(COMPONENTS[c] for c in r.deps)}
                          for r in REVISIONS},
            "window": "closed form D - t - eta; a lower bound on Definition 3 because PRESERVE exists",
        },
        "cells": cells,
        "reference_noop_agent": reference_noop,
        "completion_delta_timely_inc_minus_feas": delta,
        "decision_criterion": crit,
        "sanity_checks": checks,
        "emit": {
            "study2.feas_success_H4": succ("feas", 4),
            "study2.timely_success_H4": succ("timely_inc", 4),
            "study2.completion_delta": delta["H=4"]["mean"],
            "study2.feas_success_H4[hold]": succ("feas", 4, "hold"),
            "study2.timely_success_H4[hold]": succ("timely_inc", 4, "hold"),
            "study2.feas_success_H4[continue,adversarial]": succ("feas", 4, "continue", "adversarial"),
            "study2.feas_success_H4[hold,adversarial]": succ("feas", 4, "hold", "adversarial"),
        },
    }

# ----------------------------------------------------------------------
# Figures. Okabe-Ito subset (validated for CVD separation); identity is
# also carried by marker and line style, so the figures survive greyscale.
# timely_full and timely_inc make identical decisions, so they are one line.
# ----------------------------------------------------------------------
SERIES = [  # key, label, colour, marker, line style
    ("none", "none", "#E69F00", "v", ":"),
    ("perm", "perm", "#CC79A7", "^", "--"),
    ("feas", "feas (strong baseline)", "#D55E00", "s", "-"),
    ("timely_inc", "timely (full = inc)", "#0072B2", "o", "-"),
]
INK, MUTED, GRID = "#1a1a1a", "#555555", "#dddddd"


def make_figures(results: dict, outdir: Path, ext: str = "pdf") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 8.5})
    cells, Hs = results["cells"], results["design"]["H_values"]
    outdir.mkdir(exist_ok=True)

    # Fig 3: success against H. Small multiples, one shared axis.
    panels = [("continue", "uniform"), ("continue", "adversarial"),
              ("hold", "uniform"), ("hold", "adversarial")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.35), sharey=True)
    for ax, (sem, mode) in zip(axes, panels):
        for key, label, colour, marker, ls in SERIES:
            q = [cells[key][f"H={H}"]["requests"][sem][mode] for H in Hs]
            y = [v["success"] for v in q]
            lo = [v["success"] - v["ci95"][0] for v in q]
            hi = [v["ci95"][1] - v["success"] for v in q]
            ax.errorbar(Hs, y, yerr=[lo, hi], color=colour, marker=marker, ms=4, lw=1.4, ls=ls,
                        elinewidth=0.8, capsize=0, label=label, clip_on=False, zorder=3)
        ax.set_title(f"{mode} requests\nagent {'continues' if sem == 'continue' else 'holds'}", color=INK)
        ax.set_xticks(Hs)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.supxlabel("human response requirement $H$ (steps)", fontsize=8.5, y=0.04)
    axes[0].set_ylabel("critical-revision success")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.02), handlelength=2.6)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9), w_pad=0.6)
    fig.savefig(outdir / f"fig3_success_vs_H.{ext}", dpi=200)
    plt.close(fig)

    # Fig 4: the two metrics against each other, never combined into one.
    noop = results["reference_noop_agent"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, sem in zip(axes, ("continue", "hold")):
        for key, label, colour, marker, ls in SERIES:
            x = [cells[key][f"H={H}"]["task_completion"]["mean"] for H in Hs]
            y = [cells[key][f"H={H}"]["requests"][sem]["uniform"]["success"] for H in Hs]
            ax.plot(x, y, color=colour, marker=marker, ms=4.5, lw=1.2, ls=ls, label=label,
                    clip_on=False, zorder=3)
            if key in ("feas", "timely_inc"):       # direct labels: which point is which H
                groups: list[list] = []             # merge labels of points that nearly coincide
                for H, xi, yi in zip(Hs, x, y):
                    if groups and abs(xi - groups[-1][1]) < 0.004 and abs(yi - groups[-1][2]) < 0.06:
                        groups[-1][0].append(H)
                    else:
                        groups.append([[H], xi, yi])
                for members, xi, yi in groups:
                    if key == "timely_inc" and members == [0]:
                        continue                    # same point as feas at H = 0
                    dx, dy, ha = ((6, 0, "left") if key == "feas" else (0, -11, "center"))
                    ax.annotate(", ".join(map(str, members)), (xi, yi), xytext=(dx, dy),
                                textcoords="offset points", fontsize=7, color=MUTED, ha=ha, va="center")
        n = noop["H=4"]
        ax.annotate(f"no-op agent (off axis):\ncompletion {n['task_completion']['mean']:.2f}, "
                    f"success {n['requests'][sem]['uniform']['success']:.2f}",
                    xy=(0.868, 1.0), xytext=(0.885, 0.2), fontsize=7.5, color=INK, va="center",
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8,
                                    connectionstyle="arc3,rad=-0.25"))
        ax.set_title(f"uniform requests, agent {'continues' if sem == 'continue' else 'holds'}", color=INK)
        ax.set_xlim(0.868, 1.006)
        ax.set_ylim(-0.03, 1.06)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("task completion (mean fraction)")
    axes[0].set_ylabel("critical-revision success")
    axes[1].text(0.995, 0.03, "point labels: $H$", transform=axes[1].transAxes, fontsize=7,
                 color=MUTED, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.02), handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.8)
    fig.savefig(outdir / f"fig4_completion_vs_success.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "study2.json")
    args = ap.parse_args()
    results = run(args.trials)
    args.out.parent.mkdir(exist_ok=True)
    with open(args.out, "w", newline="\n") as f:
        json.dump(results, f, indent=2)
    make_figures(results, ROOT / "figures")
    print(json.dumps({k: results[k] for k in ("emit", "sanity_checks",
                                              "completion_delta_timely_inc_minus_feas")}, indent=2))
    print(f"{args.trials} trials per cell in {results['elapsed_s']:.1f}s -> {args.out}")
