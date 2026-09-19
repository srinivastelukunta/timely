"""Print compact tables from a Study 3 results file."""
import json, sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "study3.json"
r = json.load(open(path))
d = r["design"]
rows = [("local", lag) for lag in d["lags"] if lag] + [("global", 0), ("reserved", 0)]
cell = lambda c, k, lag, H: r["cells"][f"{c}|k={k}|lag={lag}|H={H}"]

def table(title, get, H, fmt="{:9.3f}"):
    print(f"\n{title}  [H={H}]\n{'':16s}" + "".join(f"{'k=' + str(k):>9s}" for k in d["agents"]))
    for c, lag in rows:
        name = f"{c} lag={lag}" if c == "local" else c
        print(f"{name:16s}" + "".join(fmt.format(get(cell(c, k, lag, H))) for k in d["agents"]))

H = d["headline_cell"]["H"]
table("trials with any timeliness violation", lambda x: x["timeliness_violation"]["trials_with_any"]["rate"], H)
table("fraction of protectable ticks in violation", lambda x: x["timeliness_violation"]["tick_rate"], H)
table("mean max depth below H, when violated", lambda x: x["timeliness_violation"]["mean_max_depth_when_violated"], H, "{:9.2f}")
table("trials with any FEASIBILITY violation", lambda x: x["feasibility_violation"]["trials_with_any"], H)
table("revision success, uniform, agent continues", lambda x: x["revision_success_uniform"]["continue"], H)
table("revision success, uniform, agent holds", lambda x: x["revision_success_uniform"]["hold"], H)
table("revision success, adversarial, agent holds", lambda x: x["revision_success_adversarial"]["hold"], H)
table("task completion", lambda x: x["task_completion_mean"], H)
table("throughput (units / tick)", lambda x: x["throughput_units_per_tick"], H)
table("window rejects per filter eval", lambda x: x["blocking"]["window_rejects_per_eval"], H)
table("blocked agent-tick fraction", lambda x: x["blocking"]["blocked_agent_tick_fraction"], H)
table("ledger retries per commit", lambda x: x["blocking"]["ledger_retries_per_commit"], H)
table("filter evals per commit", lambda x: x["blocking"]["filter_evals_per_commit"], H, "{:9.2f}")

k = d["headline_cell"]["k"]
print(f"\ntrials with any violation, k={k}, by H\n{'':16s}" + "".join(f"{'H=' + str(h):>9s}" for h in d["H_values"]))
for c, lag in rows:
    name = f"{c} lag={lag}" if c == "local" else c
    print(f"{name:16s}" + "".join(f"{cell(c, k, lag, h)['timeliness_violation']['trials_with_any']['rate']:9.3f}" for h in d["H_values"]))
print("\nerosion:", json.dumps(r["window_erosion_vs_single_agent"]))
print("throughput loss reserved vs global:", r["throughput_loss_reserved_vs_global"])
print("throughput loss reserved vs local :", r["throughput_loss_reserved_vs_local"])
print("emit:", json.dumps(r["emit"], indent=1))
print("checks:", r["sanity_checks"], "| elapsed", round(r["elapsed_s"], 1), "s")
