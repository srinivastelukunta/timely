"""
Study 1: Layered protection under a widening capability gap.

Establishes the negative result numerically and quantifies what
layering actually buys. No plateau is possible; layering buys gap-headroom.
Common-cause coupling lowers that headroom at every layer count. It caps
it only when beta > 1 - theta, which none of the plotted values reach.

Outputs: results/study1.json, figures/fig1_survival.pdf, figures/fig2_headroom.pdf
"""
import json, itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20260919)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Okabe-Ito colourblind-safe palette
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


# ----------------------------------------------------------------------
# Core model
# ----------------------------------------------------------------------
def layer_survival(k, lam, form="exp"):
    """p_i(k): probability layer i still functions at capability gap k."""
    k = np.asarray(k, dtype=float)
    if form == "exp":
        return np.exp(-lam * k)
    if form == "logistic":
        # alternative decay form for the sensitivity analysis
        return 1.0 / (1.0 + np.exp(lam * (k - 1.0 / lam)))
    raise ValueError(form)


def p_surv(k, lams, C_mat, beta=0.0, form="exp"):
    """
    System survival at gap k.

    Hazards compose in series, covering layers in parallel, with a
    beta-factor common-cause term. beta=0 recovers the naive independent
    model used in the v0.1 draft.

    C_mat[i, j] in [0,1] is the coverage of hazard j by layer i.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    p = np.stack([layer_survival(k, l, form) for l in lams])      # (n_layers, n_k)
    n_layers, n_hazards = C_mat.shape

    # common-cause failure intensity: grows as the layers themselves weaken
    q_cc = beta * (1.0 - p.mean(axis=0))                           # (n_k,)

    out = np.ones_like(k)
    for j in range(n_hazards):
        cj = C_mat[:, j][:, None]                                  # (n_layers, 1)
        indep_fail = np.prod(1.0 - cj * p, axis=0)                 # (n_k,)
        hazard_fail = q_cc + (1.0 - q_cc) * indep_fail
        out *= (1.0 - hazard_fail)
    return out


def k_headroom(lams, C_mat, beta=0.0, theta=0.5, form="exp", kmax=400.0):
    """Largest gap k at which system survival still meets threshold theta."""
    lo, hi = 0.0, kmax
    if p_surv(np.array([lo]), lams, C_mat, beta, form)[0] < theta:
        return 0.0
    if p_surv(np.array([hi]), lams, C_mat, beta, form)[0] >= theta:
        return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if p_surv(np.array([mid]), lams, C_mat, beta, form)[0] >= theta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


results = {"seed": 20260919, "model": "P_surv = prod_j [1 - (q_cc + (1-q_cc) prod_i (1 - c_ij p_i(k)))]"}

# ----------------------------------------------------------------------
# 1. The no-plateau result: single layers vs full stacks
# ----------------------------------------------------------------------
LAM = 0.10
ks = np.linspace(0, 120, 1200)

curves = {}
for n in [1, 2, 4, 8, 16]:
    lams = np.full(n, LAM)
    Cm = np.ones((n, 1))                    # n layers, all fully covering one hazard
    curves[n] = p_surv(ks, lams, Cm, beta=0.0)

results["no_plateau"] = {
    "lambda": LAM,
    "note": "identical layers, full coverage of a single hazard, no common cause",
    "survival_at_k": {
        str(n): {str(k): float(np.interp(k, ks, curves[n])) for k in [10, 25, 50, 100]}
        for n in curves
    },
}

# analytic headroom check: k_theta = -ln(1 - (1-theta)^(1/n)) / lambda
theta = 0.5
head = {}
for n in [1, 2, 4, 8, 16, 32, 64]:
    lams = np.full(n, LAM)
    Cm = np.ones((n, 1))
    numeric = k_headroom(lams, Cm, beta=0.0, theta=theta)
    analytic = -np.log(1.0 - (1.0 - theta) ** (1.0 / n)) / LAM
    head[n] = {"numeric": numeric, "analytic": float(analytic)}
results["headroom_no_cc"] = {"theta": theta, "lambda": LAM, "by_n_layers": head}

# doubling test: does doubling the layer count double the headroom?
results["headroom_gain_per_doubling"] = {
    f"{n}->{2*n}": head[2 * n]["numeric"] - head[n]["numeric"]
    for n in [1, 2, 4, 8, 16, 32]
}

# ----------------------------------------------------------------------
# 2. Common-cause saturation
# ----------------------------------------------------------------------
betas = [0.0, 0.01, 0.05, 0.10, 0.25]
cc = {}
for b in betas:
    cc[b] = {}
    for n in [1, 2, 4, 8, 16, 32, 64]:
        lams = np.full(n, LAM)
        Cm = np.ones((n, 1))
        cc[b][n] = k_headroom(lams, Cm, beta=b, theta=theta)
results["headroom_with_common_cause"] = {
    str(b): {str(n): (None if np.isinf(v) else float(v)) for n, v in d.items()}
    for b, d in cc.items()
}
# marginal value of the 64th layer relative to the 8th, per beta
results["saturation_ratio_64_over_8"] = {
    str(b): (float(cc[b][64] / cc[b][8]) if cc[b][8] > 0 else None) for b in betas
}

# ----------------------------------------------------------------------
# 3. Coverage necessity: one weakly covered hazard caps the system
# ----------------------------------------------------------------------
n = 8
lams = np.full(n, LAM)
cov_levels = [0.0, 0.1, 0.25, 0.5, 1.0]
capped = {}
for c_weak in cov_levels:
    Cm = np.ones((n, 2))
    Cm[:, 1] = c_weak                        # hazard 2 only weakly covered
    capped[c_weak] = k_headroom(lams, Cm, beta=0.0, theta=theta)
results["coverage_necessity"] = {
    "n_layers": n,
    "note": "hazard 1 fully covered by all 8 layers; hazard 2 covered at the listed level",
    "headroom_by_weak_coverage": {str(c): float(v) for c, v in capped.items()},
}

# ----------------------------------------------------------------------
# 4. Sensitivity to decay form and heterogeneous lambdas
# ----------------------------------------------------------------------
sens = {}
for form in ["exp", "logistic"]:
    lams = np.full(8, LAM)
    Cm = np.ones((8, 1))
    sens[form] = k_headroom(lams, Cm, beta=0.0, theta=theta, form=form)

het = []
for _ in range(2000):
    lams = RNG.uniform(0.05, 0.20, size=8)
    Cm = (RNG.random((8, 4)) < 0.5).astype(float)
    for j in range(4):                       # guarantee every hazard has a coverer
        if Cm[:, j].sum() == 0:
            Cm[RNG.integers(0, 8), j] = 1.0
    het.append(k_headroom(lams, Cm, beta=0.0, theta=theta))
het = np.array(het)
results["sensitivity"] = {
    "decay_form_headroom": {k: float(v) for k, v in sens.items()},
    "heterogeneous_monte_carlo": {
        "n_draws": 2000,
        "mean_headroom": float(het.mean()),
        "p05": float(np.percentile(het, 5)),
        "p95": float(np.percentile(het, 95)),
        "max_headroom": float(het.max()),
        "fraction_infinite": float(np.mean(np.isinf(het))),
    },
}

# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(5.0, 3.1))
for idx, n in enumerate([1, 2, 4, 8, 16]):
    ax.plot(ks, curves[n], color=C[idx], lw=1.6, label=f"{n} layer" + ("s" if n > 1 else ""))
ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
ax.set_xlabel("capability gap $k$")
ax.set_ylabel("$P_{\\mathrm{surv}}$")
ax.set_ylim(-0.02, 1.02)
ax.set_xlim(0, 120)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("Every finite stack collapses; layering shifts the curve right", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / "fig1_survival.pdf")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.0, 3.1))
ns = [1, 2, 4, 8, 16, 32, 64]
for idx, b in enumerate(betas):
    vals = [cc[b][n] for n in ns]
    ax.plot(ns, vals, marker="o", ms=3.5, color=C[idx], lw=1.5,
            label=f"$\\beta$ = {b:g}")
ax.set_xscale("log", base=2)
ax.set_xticks(ns)
ax.set_xticklabels([str(x) for x in ns])
ax.set_xlabel("number of layers")
ax.set_ylabel("gap headroom $k_{0.5}$")
ax.legend(frameon=False, fontsize=8)
ax.set_title("Common-cause coupling lowers headroom without capping it here", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / "fig2_headroom.pdf")
plt.close(fig)

with open(RESULTS / "study1.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2)[:3000])

# ----------------------------------------------------------------------
# 5. Correlated layers: effective independence is what buys headroom
# ----------------------------------------------------------------------
# A group of layers sharing one failure mode behaves as a single layer.
# Because headroom is logarithmic in the layer count, a stack with many
# layers but few independent modes has the headroom of the smaller stack.
corr = {}
for n_nominal in [8, 16, 32, 64]:
    corr[n_nominal] = {}
    for n_modes in [1, 2, 4, 8, 16, 32, 64]:
        if n_modes > n_nominal:
            continue
        lams = np.full(n_modes, LAM)          # perfectly correlated groups collapse to modes
        Cm = np.ones((n_modes, 1))
        corr[n_nominal][n_modes] = k_headroom(lams, Cm, beta=0.0, theta=theta)

extra = {
    "correlated_layers": {
        "note": ("layers sharing a failure mode collapse to one effective layer; "
                 "headroom depends on the number of independent modes, not the nominal count"),
        "headroom_by_nominal_and_modes": {
            str(nn): {str(nm): float(v) for nm, v in d.items()} for nn, d in corr.items()
        },
        "headroom_lost_64_layers_8_modes": float(corr[64][64] - corr[64][8]),
    },
    "headroom_law": {
        "closed_form": "k_theta(n) = -ln(1 - (1-theta)^(1/n)) / lambda",
        "asymptotic_gain_per_doubling": float(np.log(2) / LAM),
        "interpretation": ("each doubling of the number of INDEPENDENT layers buys a "
                           "constant additive headroom of ln(2)/lambda, so headroom grows "
                           "logarithmically in layer count and never yields an asymptote"),
    },
}
results.update(extra)
with open(RESULTS / "study1.json", "w") as f:
    json.dump(results, f, indent=2)

print("=== headroom law ===")
print(json.dumps(results["headroom_law"], indent=2))
print("=== correlated layers ===")
print(json.dumps(results["correlated_layers"], indent=2))
print("=== saturation (64 vs 8 layers) by beta ===")
print(json.dumps(results["saturation_ratio_64_over_8"], indent=2))
print("=== coverage necessity ===")
print(json.dumps(results["coverage_necessity"], indent=2))
print("=== sensitivity ===")
print(json.dumps(results["sensitivity"], indent=2))
print("=== survival at k ===")
print(json.dumps(results["no_plateau"]["survival_at_k"], indent=2))
