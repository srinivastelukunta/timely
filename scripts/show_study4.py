"""Print compact tables from results/study4.json."""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "study4.json"
r = json.load(open(path))
tiers = [t for t in r["models"] if r["by_tier_at_base"][t]["behaviour"]["n_episodes"]]
f = lambda v, spec="{:9.2f}": f"{'-':>9s}" if v is None else spec.format(v)


def table(title, cells, rows):
    cols = list(cells)
    print(f"\n{title}\n{'':38s}" + "".join(f"{c.replace(chr(124), chr(47))[-10:]:>10s}" for c in cols))
    for name, pick in rows:
        vals = []
        for c in cols:
            try:
                vals.append(pick(cells[c]))
            except (KeyError, TypeError):
                vals.append(None)
        print(f"{name:38s}" + "".join(" " + f(v) for v in vals))


SELF = [("probes", lambda c: c["self_report"]["n_probes"]),
        ("truth: fraction feasible", lambda c: c["self_report"]["truth_feasible_rate"]),
        ("feasibility accuracy", lambda c: c["self_report"]["feasibility"]["accuracy"]),
        ("  majority-class baseline", lambda c: c["self_report"]["feasibility"]["majority_class_baseline"]),
        ("  balanced accuracy", lambda c: c["self_report"]["feasibility"]["balanced_accuracy"]),
        ("  says feasible | feasible", lambda c: c["self_report"]["feasibility"]["says_feasible_when_feasible"]),
        ("  says infeasible | infeasible", lambda c: c["self_report"]["feasibility"]["says_infeasible_when_infeasible"]),
        ("  Brier score", lambda c: c["self_report"]["feasibility"]["brier"]),
        ("margin: mean abs error (steps)", lambda c: c["self_report"]["margin"]["mean_abs_error_steps"]),
        ("  clock-only baseline MAE", lambda c: c["self_report"]["margin"]["clock_only_baseline_mae"]),
        ("  mean signed error", lambda c: c["self_report"]["margin"]["mean_signed_error_steps"]),
        ("  within 2 steps", lambda c: c["self_report"]["margin"]["within_2_steps"]),
        ("  Spearman with measured W", lambda c: c["self_report"]["margin"]["spearman_with_measured_window"]),
        ("  timely verdict accuracy", lambda c: c["self_report"]["margin"]["timely_judgement_accuracy"]),
        ("  overclaims timely | W < H", lambda c: c["self_report"]["margin"]["overclaims_timely"]),
        ("obstacle count MAE", lambda c: c["self_report"]["obstacle_count_mae"])]
BEHAVE = [("episodes", lambda c: c["behaviour"]["n_episodes"]),
          ("task completion", lambda c: c["behaviour"]["task_completion"]),
          ("shortcut rate when offered", lambda c: c["behaviour"]["shortcut_rate_when_offered"]),
          ("preserve rate", lambda c: c["behaviour"]["preserve_rate"]),
          ("unauthorised attempts when offered", lambda c: c["behaviour"]["unauthorised_attempt_rate_when_offered"]),
          ("window at request (mean)", lambda c: c["behaviour"]["window_at_request_mean"]),
          ("window at request < H", lambda c: c["behaviour"]["window_at_request_below_H"]),
          ("oracle-feasible at handover", lambda c: c["behaviour"]["oracle_feasible_at_handover"]),
          ("handover success", lambda c: c["behaviour"]["handover_success"]),
          ("  given oracle-feasible", lambda c: c["behaviour"]["handover_success_given_oracle_feasible"]),
          ("replacement can continue", lambda c: c["behaviour"]["replacement_can_continue"]),
          ("replacement wasted-step rate", lambda c: c["behaviour"]["replacement_wasted_step_rate"]),
          ("seconds per episode", lambda c: c["behaviour"]["seconds_per_episode"])]

print(f"{r['n_episodes']} episodes")
table("SELF-REPORT, all unmediated probes", {"all": r["all_unmediated"]}, SELF)
table("SELF-REPORT by tier, base config, both drivers", {t: r["by_tier_at_base"][t] for t in tiers}, SELF)
for t in tiers:
    table(f"CAPABILITY ARM [{t}] (scripted histories)", r["capability_arm"][t], SELF[8:14] + SELF[2:3] + BEHAVE[5:6])
for t in tiers:
    table(f"BEHAVIOUR, model drives [{t}]", {k: v for k, v in r["by_driver"][t]["model"].items()}, BEHAVE)
    table(f"BEHAVIOUR, scripted greedy drives [{t}]", {k: v for k, v in r["by_driver"][t]["greedy"].items()}, BEHAVE)
print("\nemit:", json.dumps(r.get("emit", {}), indent=1))
