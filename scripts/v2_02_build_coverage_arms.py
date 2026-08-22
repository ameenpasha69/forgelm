"""E2 -- build two training arms that differ ONLY in scenario coverage.

v1's data-size ablation was informative but confounded: the smaller arm lost a
scenario family as a side effect of stratifying by category. This builds arms
where coverage is the manipulated variable and everything else is pinned.

    Arm A  high coverage   32 families x 2 examples = 64
    Arm B  low  coverage   16 families x 4 examples = 64

Both arms hold exactly 8 examples per category, so the category distribution is
identical by construction rather than approximately.

Why 64 and not more: equal counts with category balance require Arm B's
families to supply 4 examples each, and only 6 of the 32 training families hold
6 or more. 32x3 / 16x6 = 96 is therefore infeasible; 32x2 / 16x4 = 64 is the
largest clean design the v1 training split supports. That limit was recorded in
the pre-registration before any arm was run.

    python scripts/v2_02_build_coverage_arms.py

Outputs experiments/v2/configs/arm_{high,low}_coverage.json, consumed by
03_train_lora.py --families.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio  # noqa: E402
from forgelm.ledger import REPO_ROOT, Run  # noqa: E402
from forgelm.seeding import rng  # noqa: E402
from forgelm.splits import apply_split, load_manifest  # noqa: E402

CONFIGS = REPO_ROOT / "experiments" / "v2" / "configs"

EXAMPLES_PER_CATEGORY = 8          # identical in both arms
HIGH_FAMILIES_PER_CATEGORY = 4     # all of them
LOW_FAMILIES_PER_CATEGORY = 2
HIGH_PER_FAMILY = EXAMPLES_PER_CATEGORY // HIGH_FAMILIES_PER_CATEGORY   # 2
LOW_PER_FAMILY = EXAMPLES_PER_CATEGORY // LOW_FAMILIES_PER_CATEGORY     # 4


def pick(records: list[dict], per_family: int, families: list[str],
         context: str) -> list[dict]:
    """Deterministically take `per_family` examples from each named family."""
    by_family = defaultdict(list)
    for record in sorted(records, key=lambda r: r["example_id"]):
        by_family[record["scenario_family"]].append(record)

    chosen: list[dict] = []
    for family in sorted(families):
        pool = list(by_family[family])
        if len(pool) < per_family:
            raise RuntimeError(
                f"family {family} has {len(pool)} examples, need {per_family}")
        rng("training", context, family).shuffle(pool)
        chosen.extend(pool[:per_family])
    chosen.sort(key=lambda r: r["example_id"])
    return chosen


def summarise(records: list[dict]) -> dict:
    return {
        "n": len(records),
        "n_families": len({r["scenario_family"] for r in records}),
        "by_category": dict(sorted(Counter(r["category"]
                                           for r in records).items())),
        "by_priority": dict(sorted(Counter(r["expected_output"]["priority"]
                                           for r in records).items())),
        "by_service": dict(sorted(Counter(r["expected_output"]["affected_service"]
                                          for r in records).items())),
        "security_incidents": sum(
            1 for r in records if r["expected_output"]["is_security_incident"]),
        "examples_per_family": round(
            len(records) / len({r["scenario_family"] for r in records}), 3),
    }


def main() -> int:
    run = Run(kind="build_coverage_arms").start()
    try:
        records = dataio.read_jsonl(dataio.PROCESSED_DATASET)
        manifest = load_manifest(dataio.SPLIT_MANIFEST)
        train = apply_split(records, manifest)["train"]

        by_category = defaultdict(set)
        for record in train:
            by_category[record["category"]].add(record["scenario_family"])

        high_families: list[str] = []
        low_families: list[str] = []
        for category in sorted(by_category):
            families = sorted(by_category[category])
            if len(families) != HIGH_FAMILIES_PER_CATEGORY:
                raise RuntimeError(
                    f"{category} has {len(families)} training families, "
                    f"expected {HIGH_FAMILIES_PER_CATEGORY}")
            high_families.extend(families)
            # Which two families Arm B keeps is decided by a dedicated seed, so
            # it is neither alphabetical nor chosen after seeing any result.
            shuffled = list(families)
            rng("split_assignment", "coverage_arm_low", category).shuffle(shuffled)
            low_families.extend(sorted(shuffled[:LOW_FAMILIES_PER_CATEGORY]))

        high = pick(train, HIGH_PER_FAMILY, high_families, "coverage_high")
        low = pick(train, LOW_PER_FAMILY, low_families, "coverage_low")

        if len(high) != len(low):
            raise RuntimeError(
                f"arms must be equal size: high={len(high)} low={len(low)}")

        high_stats, low_stats = summarise(high), summarise(low)
        if high_stats["by_category"] != low_stats["by_category"]:
            raise RuntimeError(
                f"category distributions differ:\n  high={high_stats['by_category']}"
                f"\n  low ={low_stats['by_category']}")

        for arm, families, chosen, stats in (
                ("high_coverage", high_families, high, high_stats),
                ("low_coverage", low_families, low, low_stats)):
            dataio.write_json({
                "arm": arm,
                "experiment": "E2 coverage vs depth",
                "families": sorted(families),
                "example_ids": [r["example_id"] for r in chosen],
                "n_examples": len(chosen),
                "examples_per_family": (HIGH_PER_FAMILY if arm == "high_coverage"
                                        else LOW_PER_FAMILY),
                "stats": stats,
                "controlled": [
                    "equal total example count",
                    "identical category distribution (8 per category)",
                    "identical prompt, LoRA settings, decoding",
                    "identical validation and test splits",
                    "identical training seeds",
                ],
            }, CONFIGS / f"arm_{arm}.json")

        print(f"Arm A high coverage: {high_stats['n']} examples, "
              f"{high_stats['n_families']} families, "
              f"{high_stats['examples_per_family']}/family")
        print(f"Arm B low  coverage: {low_stats['n']} examples, "
              f"{low_stats['n_families']} families, "
              f"{low_stats['examples_per_family']}/family")
        print(f"category distribution identical: "
              f"{high_stats['by_category'] == low_stats['by_category']}")
        print(f"  high priorities: {high_stats['by_priority']}")
        print(f"  low  priorities: {low_stats['by_priority']}")
        print(f"  overlap in example ids: "
              f"{len(set(r['example_id'] for r in high) & set(r['example_id'] for r in low))}")

        run.metrics = {"high": high_stats, "low": low_stats}
        run.finish("success")
        print(f"\nwrote {CONFIGS.relative_to(REPO_ROOT)}/arm_high_coverage.json "
              f"and arm_low_coverage.json")
        return 0

    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
