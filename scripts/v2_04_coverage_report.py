"""E2 -- does broader scenario coverage beat more depth at a fixed budget?

Two arms, 64 training examples each, identical category distribution, priority
distribution matched at design time, same three seeds, same prompt, same LoRA
settings, same decoding, same validation and test splits.

    Arm A  high coverage   32 families x 2 examples
    Arm B  low  coverage   16 families x 4 examples

The reporting rule was fixed before any arm ran: a null result is reported as
"no detectable difference", never as evidence that the arms are equivalent.

    python scripts/v2_04_coverage_report.py

Outputs experiments/v2/reports/coverage_depth/{coverage.json,COVERAGE.md}
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio, metrics as M  # noqa: E402
from forgelm.ledger import REPO_ROOT, Run  # noqa: E402
from forgelm.schema import FIELD_ORDER  # noqa: E402

REPORTS = REPO_ROOT / "reports"
CONFIGS = REPO_ROOT / "experiments" / "v2" / "configs"
OUT = REPO_ROOT / "experiments" / "v2" / "reports" / "coverage_depth"

SEEDS = (1337, 2718, 3141)
ARMS = {"high": "Arm A -- high coverage (32 families x 2)",
        "low": "Arm B -- low coverage (16 families x 4)"}


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def load_arm(arm: str, seed: int):
    path = REPORTS / "predictions" / f"lora_{arm}cov_s{seed}_test.jsonl"
    return dataio.read_jsonl(path) if path.exists() else None


def family_stats(preds):
    fam = defaultdict(list)
    for row in preds:
        fam[row["scenario_family"]].append(row)
    zero = sum(1 for rows in fam.values()
               if not any(r["exact_match"] for r in rows))
    return {"n_families": len(fam), "n_zero_scoring": zero}


def main() -> int:
    run = Run(kind="coverage_report").start()
    try:
        arm_specs = {
            arm: dataio.read_json(CONFIGS / f"arm_{arm}_coverage.json")
            for arm in ARMS
        }

        loaded: dict[str, dict[int, list]] = {arm: {} for arm in ARMS}
        for arm in ARMS:
            for seed in SEEDS:
                preds = load_arm(arm, seed)
                if preds is None:
                    print(f"  (missing: {arm} coverage, seed {seed})")
                    continue
                loaded[arm][seed] = preds

        available = sorted(set(loaded["high"]) & set(loaded["low"]))
        if not available:
            raise SystemExit("no seed has both arms; nothing to compare")
        print(f"seeds with both arms: {available}")

        # ---- per seed -----------------------------------------------------
        per_seed = []
        for seed in available:
            high, low = loaded["high"][seed], loaded["low"][seed]
            hm, lm = M.compute_metrics(high), M.compute_metrics(low)
            cmp = M.compare(low, high, "low_coverage", "high_coverage",
                            metric="exact_match")
            per_seed.append({
                "seed": seed,
                "high": {"exact_match": hm["exact_match"],
                         "schema_valid_rate": hm["schema_valid_rate"],
                         "field_accuracy": hm["field_accuracy"],
                         **family_stats(high)},
                "low": {"exact_match": lm["exact_match"],
                        "schema_valid_rate": lm["schema_valid_rate"],
                        "field_accuracy": lm["field_accuracy"],
                        **family_stats(low)},
                "comparison": cmp,
            })
            d = cmp["paired_diff"]
            print(f"  seed {seed}: high={hm['exact_match']:.4f} "
                  f"low={lm['exact_match']:.4f} "
                  f"diff={d['diff']*100:+.1f}pp "
                  f"CI[{d['lo']*100:+.1f},{d['hi']*100:+.1f}] "
                  f"p={cmp['mcnemar']['p_value']:.4f}")

        # ---- pooled across seeds -------------------------------------------
        shared_ids = None
        for arm in ARMS:
            for seed in available:
                ids = {r["example_id"] for r in loaded[arm][seed]}
                shared_ids = ids if shared_ids is None else (shared_ids & ids)
        shared = sorted(shared_ids)

        def vectors(arm: str, key: str):
            out = []
            for seed in available:
                index = {r["example_id"]: r for r in loaded[arm][seed]}
                out.append([1.0 if index[e][key] else 0.0 for e in shared])
            return out

        pooled = {}
        for label, key in (("exact_match", "exact_match"),
                           ("schema_valid_rate", "schema_valid")):
            pooled[label] = {
                "high": M.hierarchical_bootstrap(vectors("high", key)),
                "low": M.hierarchical_bootstrap(vectors("low", key)),
            }

        # Seed-paired difference: average the per-seed differences, then
        # bootstrap over examples within each seed.
        diffs = [s["comparison"]["paired_diff"]["diff"] for s in per_seed]
        mean_diff = sum(diffs) / len(diffs)
        n_sig = sum(1 for s in per_seed
                    if s["comparison"]["paired_diff"]["excludes_zero"])
        directions = {"high_better": sum(1 for d in diffs if d > 0),
                      "low_better": sum(1 for d in diffs if d < 0),
                      "tied": sum(1 for d in diffs if d == 0)}

        detected = n_sig > len(per_seed) / 2
        verdict = ("higher coverage helps" if detected and mean_diff > 0 else
                   "more depth helps" if detected and mean_diff < 0 else
                   "no detectable difference")

        payload = {
            "experiment": "E2 coverage vs depth at a fixed example budget",
            "arms": {a: {"description": ARMS[a],
                         "n_examples": arm_specs[a]["n_examples"],
                         "n_families": len(arm_specs[a]["families"]),
                         "examples_per_family": arm_specs[a]["examples_per_family"],
                         "stats": arm_specs[a]["stats"]}
                     for a in ARMS},
            "seeds": available,
            "per_seed": per_seed,
            "pooled": pooled,
            "mean_diff_high_minus_low": round(mean_diff, 4),
            "n_seeds_with_significant_difference": n_sig,
            "direction_counts": directions,
            "verdict": verdict,
            "reporting_rule": (
                "Fixed before any arm ran: a null result is reported as 'no "
                "detectable difference', not as evidence that the arms are "
                "equivalent. With 64 training examples per arm, 3 seeds and 86 "
                "test examples this design is underpowered for small effects."),
        }

        # ---- markdown -------------------------------------------------------
        L: list[str] = []
        add = L.append
        add("# E2 -- coverage versus depth at a fixed example budget")
        add("")
        add(f"**Verdict: {verdict}.**")
        add("")
        add("Two arms, **64 training examples each**, differing only in how "
            "those examples are spread over scenario families.")
        add("")
        add("| | Arm A -- high coverage | Arm B -- low coverage |")
        add("|---|---|---|")
        add(f"| Scenario families | {len(arm_specs['high']['families'])} "
            f"| {len(arm_specs['low']['families'])} |")
        add(f"| Examples per family | {arm_specs['high']['examples_per_family']} "
            f"| {arm_specs['low']['examples_per_family']} |")
        add(f"| Total examples | {arm_specs['high']['n_examples']} "
            f"| {arm_specs['low']['n_examples']} |")
        add(f"| Category distribution | "
            f"{arm_specs['high']['stats']['by_category']} | "
            f"{arm_specs['low']['stats']['by_category']} |")
        add(f"| Priority distribution | "
            f"{arm_specs['high']['stats']['by_priority']} | "
            f"{arm_specs['low']['stats']['by_priority']} |")
        add("")
        add("Category distribution is identical by construction. Priority "
            "distribution was matched at design time by choosing Arm B's "
            "families to minimise distance to Arm A's; the residual gap is "
            "reported above rather than smoothed over, because only six family "
            "pairings exist per category and an exact match is not achievable.")
        add("")

        add("## Per seed (frozen v1 test split, n=86)")
        add("")
        add("| Seed | High coverage | Low coverage | Difference (high-low) | "
            "95% CI | McNemar p | Detected? |")
        add("|---|---|---|---|---|---|---|")
        for s in per_seed:
            d = s["comparison"]["paired_diff"]
            add(f"| {s['seed']} | {fmt_pct(s['high']['exact_match'])} "
                f"| {fmt_pct(s['low']['exact_match'])} "
                f"| {d['diff']*100:+.1f} pp "
                f"| [{d['lo']*100:+.1f}, {d['hi']*100:+.1f}] pp "
                f"| {s['comparison']['mcnemar']['p_value']:.4f} "
                f"| {'yes' if d['excludes_zero'] else 'no'} |")
        add("")

        add("## Pooled across seeds")
        add("")
        add("| Metric | Arm | Mean | Seed min | Seed max | Spread |")
        add("|---|---|---|---|---|---|")
        for label, arms in pooled.items():
            for arm, stats in arms.items():
                add(f"| {label} | {arm} | {fmt_pct(stats['mean'])} "
                    f"| {fmt_pct(stats['seed_min'])} "
                    f"| {fmt_pct(stats['seed_max'])} "
                    f"| {stats['seed_spread']*100:.1f} pp |")
        add("")
        add(f"Mean difference (high - low) across seeds: "
            f"**{mean_diff*100:+.1f} pp**. "
            f"{n_sig} of {len(per_seed)} seeds show a difference whose CI "
            f"excludes zero. Direction: high better in "
            f"{directions['high_better']}, low better in "
            f"{directions['low_better']}.")
        add("")

        add("## Reading this honestly")
        add("")
        if verdict == "no detectable difference":
            add("At this budget the experiment **does not detect** a difference "
                "between spreading 64 examples over 32 scenarios and "
                "concentrating them in 16. That is *not* the same as showing "
                "the two are equivalent -- it is a statement about what this "
                "design could resolve.")
            add("")
            add("The design is underpowered on purpose-built grounds, not as an "
                "excuse: 64 training examples per arm is a third of v1's, the "
                "coverage contrast is only 2:1, there are 3 seeds, and the test "
                "split is 86 examples. E1 measured the seed spread on the full "
                "configuration at 14 pp, which is larger than most effects this "
                "comparison could hope to see.")
        else:
            add(f"The experiment detects a difference favouring "
                f"**{'higher coverage' if mean_diff > 0 else 'more depth'}**, "
                f"consistent across {n_sig} of {len(per_seed)} seeds. Given the "
                f"seed spread measured in E1 (14 pp on the full configuration), "
                f"treat the direction as the finding and the magnitude as "
                f"poorly constrained.")
        add("")
        add("Zero-scoring held-out families, per seed:")
        add("")
        add("| Seed | High coverage | Low coverage |")
        add("|---|---|---|")
        for s in per_seed:
            add(f"| {s['seed']} | {s['high']['n_zero_scoring']}/"
                f"{s['high']['n_families']} | {s['low']['n_zero_scoring']}/"
                f"{s['low']['n_families']} |")
        add("")

        dataio.write_json(payload, OUT / "coverage.json")
        (OUT / "COVERAGE.md").write_text("\n".join(L), encoding="utf-8")

        run.metrics = {"verdict": verdict, "mean_diff": mean_diff,
                       "n_significant": n_sig}
        run.finish("success")
        print(f"\nVERDICT: {verdict} (mean diff {mean_diff*100:+.1f} pp, "
              f"{n_sig}/{len(per_seed)} seeds significant)")
        print(f"wrote {(OUT / 'COVERAGE.md').relative_to(REPO_ROOT)}")
        return 0

    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
