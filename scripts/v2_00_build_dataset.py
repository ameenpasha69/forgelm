"""Build, validate, split and SEAL the v2 dataset.

Deterministic and CPU-only. Beyond the v1 checks it additionally proves that
the v2 catalogue is not a reworded v1 catalogue, by comparing every v2 ticket
against every v1 ticket and reporting the maximum similarity observed.

    python scripts/v2_00_build_dataset.py

Outputs
    experiments/v2/data/tickets_v2.jsonl
    experiments/v2/data/split_manifest_v2.json     includes the seal checksum
    experiments/v2/data/DATASET_VERSION_V2.json
    experiments/v2/reports/dataset_validation_v2.json
    experiments/v2/reports/cross_version_similarity.json

Exits non-zero on any error-severity finding.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio, validate  # noqa: E402
from forgelm.datagen_v2 import (  # noqa: E402
    CATALOGUE_VERSION, DATASET_VERSION_V2, FAMILIES_V2, TARGET_TOTAL_V2,
    family_table_v2, generate_dataset_v2,
)
from forgelm.ledger import REPO_ROOT, Run, sha256_file  # noqa: E402
from forgelm.seeding import SEEDS  # noqa: E402
from forgelm.splits_v2 import apply_split_v2, build_manifest_v2  # noqa: E402
from forgelm.validate import _char_ngrams, jaccard  # noqa: E402

V2 = REPO_ROOT / "experiments" / "v2"
DATA = V2 / "data"
REPORTS = V2 / "reports"

DATASET = DATA / "tickets_v2.jsonl"
MANIFEST = DATA / "split_manifest_v2.json"
VERSION_FILE = DATA / "DATASET_VERSION_V2.json"


def cross_version_similarity(v2_records, v1_records, top_n: int = 25):
    """Every v2 ticket against every v1 ticket.

    192 x 300 = 57,600 exact comparisons. The claim "v2 families are new
    situations, not paraphrases" is worth nothing unless it is measured.
    """
    v1_grams = [(r, _char_ngrams(r["ticket_text"])) for r in v1_records]
    pairs = []
    worst = 0.0
    worst_pair = None
    for v2_rec in v2_records:
        g2 = _char_ngrams(v2_rec["ticket_text"])
        for v1_rec, g1 in v1_grams:
            sim = jaccard(g2, g1)
            if sim > worst:
                worst, worst_pair = sim, (v2_rec["example_id"],
                                          v1_rec["example_id"])
            if sim >= 0.60:
                pairs.append({"v2": v2_rec["example_id"],
                              "v1": v1_rec["example_id"],
                              "similarity": round(sim, 4)})
    pairs.sort(key=lambda p: -p["similarity"])
    return {
        "n_comparisons": len(v2_records) * len(v1_records),
        "max_similarity": round(worst, 4),
        "max_pair": worst_pair,
        "pairs_at_or_above_0.60": pairs[:top_n],
        "n_pairs_at_or_above_0.60": len(pairs),
        "family_overlap": sorted(
            {r["scenario_family"] for r in v2_records}
            & {r["scenario_family"] for r in v1_records}),
    }


def main() -> int:
    run = Run(kind="build_dataset_v2", config={
        "dataset_version": DATASET_VERSION_V2,
        "catalogue_version": CATALOGUE_VERSION,
        "n_families": len(FAMILIES_V2),
    }, seeds=dict(SEEDS)).start()

    try:
        records = generate_dataset_v2()
        print(f"generated {len(records)} v2 examples from {len(FAMILIES_V2)} "
              f"new scenario families")
        if len(records) != TARGET_TOTAL_V2:
            raise RuntimeError(f"expected {TARGET_TOTAL_V2}, got {len(records)}")

        # ---- structural validation (same checks as v1) -------------------
        pre = validate.run_all(records, manifest=None)
        print(f"pre-split validation: {pre['errors']} error(s), "
              f"{pre['warnings']} warning(s)")
        for f in pre["findings"]:
            if f["severity"] == "error":
                print(f"  ERROR {f['check']}: {f['message']}")
        if pre["errors"]:
            run.metrics["validation"] = pre
            run.finish("failed")
            return 1

        # ---- not-a-paraphrase check --------------------------------------
        v1_records = dataio.read_jsonl(dataio.PROCESSED_DATASET)
        cross = cross_version_similarity(records, v1_records)
        print(f"cross-version similarity: {cross['n_comparisons']:,} "
              f"comparisons, max = {cross['max_similarity']} "
              f"(pair {cross['max_pair']})")
        print(f"  v1/v2 scenario-family name overlap: "
              f"{cross['family_overlap'] or 'none'}")
        if cross["family_overlap"]:
            raise RuntimeError(
                f"v2 reuses v1 scenario families: {cross['family_overlap']}")
        if cross["max_similarity"] >= 0.80:
            raise RuntimeError(
                f"a v2 ticket is {cross['max_similarity']} similar to a v1 "
                f"ticket; the v2 catalogue would be a paraphrase, not a new "
                f"evaluation surface")

        # ---- split and seal ----------------------------------------------
        manifest = build_manifest_v2(records)
        print(f"split checksum          : {manifest['checksum'][:32]}...")
        print(f"SEALED test membership  : "
              f"{manifest['test_membership_checksum'][:32]}...")
        print(f"counts: {manifest['counts']}")

        post = validate.run_all(records, manifest=manifest)
        for f in post["findings"]:
            if f["check"] in ("cross_split_near_duplicate",
                              "scenario_family_leakage"):
                print(f"  {f['severity'].upper():5s} {f['check']}: "
                      f"{f['message']}")
        if post["errors"]:
            run.metrics["validation"] = post
            run.finish("failed")
            return 1

        for rec in records:
            rec["qc_status"] = "auto_validated"

        by_split = apply_split_v2(records, manifest)
        for name in ("train", "validation", "test"):
            stats = validate.summarise(by_split[name])
            seal = " [SEALED]" if name == "test" else ""
            print(f"  {name:11s} n={stats['n']:4d} "
                  f"families={stats['scenario_families']:3d} "
                  f"priorities={stats['priorities']}{seal}")

        # ---- write --------------------------------------------------------
        dataio.write_jsonl(records, DATASET)
        dataio.write_json(manifest, MANIFEST)
        dataio.write_json({
            **post,
            "cross_version_similarity": cross,
            "family_table": family_table_v2(),
            "by_split": {n: validate.summarise(rows)
                         for n, rows in by_split.items()},
        }, REPORTS / "dataset_validation_v2.json")
        dataio.write_json(cross, REPORTS / "cross_version_similarity.json")

        dataio.write_json({
            "dataset_version": DATASET_VERSION_V2,
            "catalogue_version": CATALOGUE_VERSION,
            "n_examples": len(records),
            "n_scenario_families": len(FAMILIES_V2),
            "split_checksum": manifest["checksum"],
            "test_membership_checksum": manifest["test_membership_checksum"],
            "dataset_sha256": sha256_file(DATASET),
            "manifest_sha256": sha256_file(MANIFEST),
            "max_similarity_to_v1": cross["max_similarity"],
            "licence": "MIT (synthetic, generated by this repository)",
            "sealed_split": "test",
            "seal_policy": (
                "v2 test predictions are not inspected until configurations, "
                "seeds, prompts, decoding and checkpoint-selection rules are "
                "frozen. splits_v2.assert_not_sealed enforces this in code."),
        }, VERSION_FILE)

        run.metrics = {
            "n_examples": len(records),
            "counts": manifest["counts"],
            "test_membership_checksum": manifest["test_membership_checksum"],
            "max_similarity_to_v1": cross["max_similarity"],
            "validation_errors": post["errors"],
        }
        for name, path in (("dataset", DATASET), ("manifest", MANIFEST),
                           ("version", VERSION_FILE)):
            run.add_artifact(name, path)
        run.finish("success")

        print(f"\nrun_id: {run.run_id}")
        print("v2 dataset built and test split SEALED.")
        return 0

    except Exception as exc:  # noqa: BLE001
        run.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
