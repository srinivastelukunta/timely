"""Print a compact table from a Study 2 results file."""
import json, sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "study2.json"
r = json.load(open(path))
Hs = r["design"]["H_values"]
conds = r["design"]["conditions"] + ["noop(ref)"]
cell = lambda c, H: (r["reference_noop_agent"] if c == "noop(ref)" else r["cells"][c])[f"H={H}"]

def table(title, get, fmt="{:8.3f}"):
    print(f"\n{title}\n{'':14s}" + "".join(f"{'H=' + str(H):>8s}" for H in Hs))
    for c in conds:
        print(f"{c:14s}" + "".join(fmt.format(get(cell(c, H))) for H in Hs))

for sem in ("continue", "hold"):
    for mode in ("uniform", "adversarial"):
        table(f"revision success [{sem}, {mode}]", lambda x: x["requests"][sem][mode]["success"])
table("exact mean over all request times [continue]", lambda x: x["requests"]["continue"]["uniform_exact_mean"])
table("task completion (mean fraction)", lambda x: x["task_completion"]["mean"])
table("fraction of trials fully complete", lambda x: x["task_completion"]["fraction_fully_complete"])
table("W at request, mean [uniform]", lambda x: x["requests"]["continue"]["uniform"]["W_at_request"]["mean"], "{:8.2f}")
table("W at request, p05 [uniform]", lambda x: x["requests"]["continue"]["uniform"]["W_at_request"]["p05"], "{:8.2f}")
table("rejected per proposal", lambda x: x["blocking"]["rejected_per_proposal"])
table("preserve step fraction", lambda x: x["blocking"]["preserve_step_fraction"])
table("blocked step fraction", lambda x: x["blocking"]["blocked_step_fraction"])
table("filter evals / trial", lambda x: x["overhead"]["filter_evals_per_trial"], "{:8.1f}")
table("witness revalidations / trial", lambda x: x["overhead"]["witness_revalidations_per_trial"], "{:8.1f}")
table("filter wall time / trial (us)", lambda x: x["overhead"]["wall_us_per_trial"], "{:8.1f}")
