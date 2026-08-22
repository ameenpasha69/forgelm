"""E1 -- does the v1 conclusion survive training-seed variance?

v1 reported one training run. Its confidence interval answered "if I drew a
different test set, how would this move?" and said nothing about "if I had
trained with a different seed, how would this move?". This script answers the
second question, using the rule fixed in experiments/v2/PREREGISTRATION.md
BEFORE any of these runs existed:

    survives            all seeds meet v1's primary criterion
    partially survives  a majority but not all
    does not survive    fewer than a majority

Every seed is reported. The best seed is never promoted.

    python scripts/v2_01_multiseed_report.py

Outputs
    experiments/v2/reports/multiseed/multiseed.json
    experiments/v2/reports/multiseed/MULTISEED.md
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio, metrics as M  # noqa: E402
from forgelm.ledger import REPO_ROOT, Run, load_runs, sha256_file  # noqa: E402
from forgelm.schema import FIELD_ORDER  # noqa: E402

REPORTS = REPO_ROOT / "reports"
OUT = REPO_ROOT / "experiments" / "v2" / "reports" / "multiseed"

# seed -> (predictions file, adapter dir, training history file).
# Seed 1337 is v1's own run: identical configuration, identical seed. It is
# reused rather than repeated, which is a disclosed deviation from the
# pre-registration (which said 1337 would be re-run as a reproducibility
# check). The machine has 5.9 GB of RAM and re-running it would have cost ~40
# minutes for a result the v1 evidence already contains.
SEED_SOURCES: dict[int, tuple[str, str, str]] = {
    1337: ("reports/predictions/lora_test.jsonl",
           "artifacts/lora_adapter",
           "reports/training_history.json"),
    2718: ("reports/predictions/lora_seed2718_test.jsonl",
           "artifacts/lora_adapter_seed2718",
           "reports/training_history_seed2718.json"),
    3141: ("reports/predictions/lora_seed3141_test.jsonl",
           "artifacts/lora_adapter_seed3141",
           "reports/training_history_seed3141.json"),
}


def adapter_checksum(adapter_dir: Path) -> str | None:
    weights = adapter_dir / "adapter_model.safetensors"
    return sha256_file(weights) if weights.exists() else None


def training_facts(history_path: Path, seed: int) -> dict:
    if not history_path.exists():
        return {}
    history = dataio.read_json(history_path)
    evals = history.get("eval_losses") or []
    trains = history.get("train_losses") or []
    return {
        "runtime_seconds": history.get("elapsed_seconds"),
        "global_step": history.get("global_step"),
        "best_checkpoint": history.get("best_checkpoint"),
        "best_eval_loss": history.get("best_metric"),
        "selected_epoch": (min(evals, key=lambda x: x[1])[0] if evals else None),
        "eval_losses": [(round(e, 2), round(l, 5)) for e, l in evals],
        "final_train_loss": round(trains[-1][1], 5) if trains else None,
    }


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> int:
    run = Run(kind="multiseed_report").start()
    try:
        baselines = {}
        for name in ("zeroshot", "fewshot"):
            path = REPORTS / "predictions" / f"{name}_test.jsonl"
            baselines[name] = dataio.read_jsonl(path)

        seeds: dict[int, dict] = {}
        for seed, (pred_rel, adapter_rel, hist_rel) in sorted(SEED_SOURCES.items()):
            pred_path = REPO_ROOT / pred_rel
            if not pred_path.exists():
                print(f"  (seed {seed}: no predictions at {pred_rel}; skipping)")
                continue
            preds = dataio.read_jsonl(pred_path)
            metrics = M.compute_metrics(preds)

            # Per-scenario-family accuracy on the held-out families.
            from collections import defaultdict
            fam = defaultdict(list)
            for row in preds:
                fam[row["scenario_family"]].append(row)
            by_family = {
                f: {"n": len(rows),
                    "n_correct": sum(1 for r in rows if r["exact_match"]),
                    "exact_match": round(
                        sum(1 for r in rows if r["exact_match"]) / len(rows), 4)}
                for f, rows in sorted(fam.items())
            }

            seeds[seed] = {
                "seed": seed,
                "predictions": pred_rel,
                "adapter_dir": adapter_rel,
                "adapter_sha256": adapter_checksum(REPO_ROOT / adapter_rel),
                "training": training_facts(REPO_ROOT / hist_rel, seed),
                "metrics": {k: v for k, v in metrics.items()
                            if k not in ("confusion_category",
                                         "confusion_priority")},
                "by_scenario_family": by_family,
                "n_families_zero_correct": sum(
                    1 for s in by_family.values() if s["n_correct"] == 0),
                "n_families": len(by_family),
                "_preds": preds,
            }
            print(f"  seed {seed}: exact_match={metrics['exact_match']:.4f} "
                  f"schema_valid={metrics['schema_valid_rate']:.4f} "
                  f"zero-scoring families="
                  f"{seeds[seed]['n_families_zero_correct']}/{len(by_family)}")

        if len(seeds) < 2:
            raise SystemExit("need at least two seeds to say anything about "
                             "seed variance")

        # ---- per-seed criterion check (v1's pre-declared rule) -------------
        criterion = []
        for seed, data in sorted(seeds.items()):
            preds = data["_preds"]
            rate = data["metrics"]["exact_match"]
            beats_both = all(
                rate > M.compute_metrics(b)["exact_match"]
                for b in baselines.values())
            cmp_fs = M.compare(baselines["fewshot"], preds, "fewshot",
                               f"lora_seed{seed}", metric="exact_match")
            excludes = cmp_fs["paired_diff"]["excludes_zero"]
            met = bool(beats_both and excludes)
            criterion.append({
                "seed": seed,
                "exact_match": rate,
                "beats_both_baselines": beats_both,
                "vs_fewshot": cmp_fs["paired_diff"],
                "mcnemar_p": cmp_fs["mcnemar"]["p_value"],
                "meets_v1_primary_criterion": met,
            })
            print(f"  seed {seed}: meets v1 criterion = {met} "
                  f"(diff {cmp_fs['paired_diff']['diff']*100:+.1f} pp, "
                  f"CI [{cmp_fs['paired_diff']['lo']*100:+.1f}, "
                  f"{cmp_fs['paired_diff']['hi']*100:+.1f}], "
                  f"p={cmp_fs['mcnemar']['p_value']:.4f})")

        n_met = sum(1 for c in criterion if c["meets_v1_primary_criterion"])
        n_seeds = len(criterion)
        if n_met == n_seeds:
            verdict = "survives"
        elif n_met * 2 > n_seeds:
            verdict = "partially survives"
        else:
            verdict = "does not survive"

        # ---- aggregate across seeds ---------------------------------------
        shared_ids = None
        for data in seeds.values():
            ids = {r["example_id"] for r in data["_preds"]}
            shared_ids = ids if shared_ids is None else (shared_ids & ids)
        shared = sorted(shared_ids or [])

        def vectors(metric_key: str) -> list[list[float]]:
            out = []
            for _, data in sorted(seeds.items()):
                index = {r["example_id"]: r for r in data["_preds"]}
                out.append([1.0 if index[e][metric_key] else 0.0
                            for e in shared])
            return out

        aggregate = {}
        for label, key in (("exact_match", "exact_match"),
                           ("schema_valid_rate", "schema_valid"),
                           ("constraint_violation_rate", "constraint_violation")):
            aggregate[label] = M.hierarchical_bootstrap(vectors(key))

        per_field = {}
        for field in FIELD_ORDER:
            vecs = []
            for _, data in sorted(seeds.items()):
                index = {r["example_id"]: r for r in data["_preds"]}
                vecs.append([1.0 if index[e]["field_correct"].get(field) else 0.0
                             for e in shared])
            per_field[field] = M.hierarchical_bootstrap(vecs, n_resamples=4000)

        payload = {
            "experiment": "E1 multi-seed variance of the v1 primary configuration",
            "n_seeds": n_seeds,
            "seeds": {s: {k: v for k, v in d.items() if k != "_preds"}
                      for s, d in seeds.items()},
            "per_seed_criterion": criterion,
            "n_seeds_meeting_v1_criterion": n_met,
            "verdict": verdict,
            "aggregate": aggregate,
            "per_field": per_field,
            "baseline_rates": {n: M.compute_metrics(b)["exact_match"]
                               for n, b in baselines.items()},
            "deviation_from_preregistration": (
                "Seed 1337 reuses the v1 training run rather than repeating it. "
                "It is the identical configuration and seed, so it is a valid "
                "member of the seed set, but the pre-registration said it would "
                "be re-run as a reproducibility check and it was not. Reason: "
                "the machine has 5.9 GB of RAM and a repeat run costs ~40 "
                "minutes of strictly serial time."),
        }

        # ---- markdown ------------------------------------------------------
        L: list[str] = []
        add = L.append
        add("# E1 -- multi-seed variance of the primary configuration")
        add("")
        add(f"**Verdict: the v1 conclusion {verdict}.** "
            f"{n_met} of {n_seeds} seeds independently meet v1's pre-declared "
            f"primary criterion.")
        add("")
        add("The rule above was fixed in `PREREGISTRATION.md` before any of "
            "these runs existed. Every seed is reported; none is promoted.")
        add("")

        add("## Per seed")
        add("")
        add("| Seed | Selected epoch | Best val loss | Runtime | Exact match | "
            "Schema valid | Zero-scoring families | Adapter sha256 |")
        add("|---|---|---|---|---|---|---|---|")
        for seed, data in sorted(seeds.items()):
            t = data["training"]
            m = data["metrics"]
            sha = data["adapter_sha256"]
            add(f"| {seed} | {t.get('selected_epoch', '?')} "
                f"| {t.get('best_eval_loss') and round(t['best_eval_loss'], 4)} "
                f"| {t.get('runtime_seconds') and round(t['runtime_seconds'])}s "
                f"| **{fmt_pct(m['exact_match'])}** "
                f"| {fmt_pct(m['schema_valid_rate'])} "
                f"| {data['n_families_zero_correct']}/{data['n_families']} "
                f"| `{sha[:12] if sha else 'n/a'}` |")
        add("")

        add("## Does each seed clear v1's bar?")
        add("")
        add(f"v1's criterion: beat **both** baselines on exact match "
            f"(zero-shot {fmt_pct(payload['baseline_rates']['zeroshot'])}, "
            f"few-shot {fmt_pct(payload['baseline_rates']['fewshot'])}) **and** "
            f"have the paired 95% CI against few-shot exclude zero.")
        add("")
        add("| Seed | Exact match | Diff vs few-shot | 95% CI | McNemar p | "
            "Meets criterion |")
        add("|---|---|---|---|---|---|")
        for c in criterion:
            d = c["vs_fewshot"]
            add(f"| {c['seed']} | {fmt_pct(c['exact_match'])} "
                f"| {d['diff']*100:+.1f} pp "
                f"| [{d['lo']*100:+.1f}, {d['hi']*100:+.1f}] pp "
                f"| {c['mcnemar_p']:.4f} "
                f"| {'**yes**' if c['meets_v1_primary_criterion'] else '**no**'} |")
        add("")

        add("## Across seeds")
        add("")
        add("Intervals below resample **both** test examples and training "
            "seeds. v1's interval resampled test examples only.")
        add("")
        add("| Metric | Mean | Seed min | Seed max | Seed spread | Seed SD | "
            "95% CI (both sources) |")
        add("|---|---|---|---|---|---|---|")
        for label, stats in aggregate.items():
            add(f"| {label} | {fmt_pct(stats['mean'])} "
                f"| {fmt_pct(stats['seed_min'])} | {fmt_pct(stats['seed_max'])} "
                f"| {stats['seed_spread']*100:.1f} pp "
                f"| {stats['seed_std'] and round(stats['seed_std']*100, 1)} pp "
                f"| [{fmt_pct(stats['lo'])}, {fmt_pct(stats['hi'])}] |")
        add("")
        add(f"With {n_seeds} seeds the seed level of that bootstrap is coarse. "
            f"The raw spread column is the more truthful summary.")
        add("")

        add("### Per field, across seeds")
        add("")
        add("| Field | Mean | Seed min | Seed max | Spread |")
        add("|---|---|---|---|---|")
        for field, stats in per_field.items():
            add(f"| `{field}` | {fmt_pct(stats['mean'])} "
                f"| {fmt_pct(stats['seed_min'])} | {fmt_pct(stats['seed_max'])} "
                f"| {stats['seed_spread']*100:.1f} pp |")
        add("")

        add("## Deviation from the pre-registration")
        add("")
        add(payload["deviation_from_preregistration"])
        add("")

        dataio.write_json(payload, OUT / "multiseed.json")
        (OUT / "MULTISEED.md").write_text("\n".join(L), encoding="utf-8")

        run.metrics = {"verdict": verdict, "n_seeds": n_seeds,
                       "n_meeting_criterion": n_met,
                       "exact_match": aggregate["exact_match"]}
        run.finish("success")
        print(f"\nVERDICT: the v1 conclusion {verdict} "
              f"({n_met}/{n_seeds} seeds meet the criterion)")
        print(f"exact match across seeds: "
              f"{aggregate['exact_match']['per_seed_means']} "
              f"(spread {aggregate['exact_match']['seed_spread']*100:.1f} pp)")
        print(f"wrote {(OUT / 'MULTISEED.md').relative_to(REPO_ROOT)}")
        return 0

    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
