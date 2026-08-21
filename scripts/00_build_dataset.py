"""Generate, validate, split and freeze the ForgeLM dataset.

Run this first. It is fully deterministic and CPU-only: no model is loaded and
no network access is required.

    python scripts/00_build_dataset.py

Outputs
    data/raw/tickets_v1.jsonl              generated examples
    data/processed/tickets_v1.validated.jsonl   same records, qc_status stamped
    data/splits/split_manifest_v1.json     frozen example -> split assignment
    data/DATASET_VERSION.json              version + checksums
    reports/dataset_validation.json        every QC / leakage finding
    reports/dataset_stats.json             per-split composition
    runs/<run_id>/run.json                 ledger record

The script exits non-zero if any error-severity finding is raised, so a broken
dataset cannot silently flow into training.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgelm import dataio, validate  # noqa: E402
from forgelm.datagen import (  # noqa: E402
    DATASET_VERSION, FAMILIES, TEMPLATE_FAMILIES, family_table, generate_dataset,
)
from forgelm.ledger import REPO_ROOT, Run, sha256_file  # noqa: E402
from forgelm.schema import SCHEMA_VERSION  # noqa: E402
from forgelm.seeding import SEEDS  # noqa: E402
from forgelm.splits import apply_split, write_manifest  # noqa: E402

REPORTS = REPO_ROOT / "reports"


def main() -> int:
    run = Run(kind="build_dataset", config={
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "n_scenario_families": len(FAMILIES),
        "n_template_families": len(TEMPLATE_FAMILIES),
    }, seeds=dict(SEEDS)).start()

    try:
        # ---- 1. generate -------------------------------------------------
        records = generate_dataset()
        print(f"generated {len(records)} examples from {len(FAMILIES)} scenario "
              f"families")

        dataio.write_jsonl(records, dataio.RAW_DATASET)
        print(f"wrote {dataio.RAW_DATASET.relative_to(REPO_ROOT)}")

        # ---- 2. structural validation (pre-split) ------------------------
        pre = validate.run_all(records, manifest=None)
        print(f"pre-split validation: {pre['errors']} error(s), "
              f"{pre['warnings']} warning(s), {pre['info']} info")
        for f in pre["findings"]:
            if f["severity"] == "error":
                print(f"  ERROR  {f['check']}: {f['message']}")

        if pre["errors"]:
            run.metrics["pre_split_validation"] = pre
            run.finish("failed")
            print("\nABORTING: dataset failed structural validation.", file=sys.stderr)
            return 1

        # ---- 3. freeze the split ----------------------------------------
        manifest = write_manifest(records, dataio.SPLIT_MANIFEST)
        print(f"split manifest checksum: {manifest['checksum']}")
        print(f"split counts: {manifest['counts']}")

        # ---- 4. leakage validation (post-split) -------------------------
        post = validate.run_all(records, manifest=manifest)
        print(f"post-split validation: {post['errors']} error(s), "
              f"{post['warnings']} warning(s), {post['info']} info")
        for f in post["findings"]:
            if f["severity"] in ("error", "info") and f["check"] in (
                "cross_split_near_duplicate", "scenario_family_leakage",
                "template_family_spread",
            ):
                print(f"  {f['severity'].upper():7s} {f['check']}: {f['message']}")

        if post["errors"]:
            run.metrics["post_split_validation"] = post
            run.finish("failed")
            print("\nABORTING: dataset failed leakage validation.", file=sys.stderr)
            return 1

        # ---- 5. stamp QC status and write processed copy ----------------
        for rec in records:
            rec["qc_status"] = "auto_validated"
        dataio.write_jsonl(records, dataio.PROCESSED_DATASET)

        # ---- 6. statistics ----------------------------------------------
        by_split = apply_split(records, manifest)
        stats = {
            "overall": validate.summarise(records),
            "by_split": {name: validate.summarise(rows)
                         for name, rows in by_split.items()},
            "family_table": family_table(),
        }
        dataio.write_json(stats, REPORTS / "dataset_stats.json")
        dataio.write_json(
            {**post,
             "near_duplicate_pairs_within_family":
                 validate.near_duplicate_pairs(records, same_family_only=True)[:50],
             "near_duplicate_pairs_across_families":
                 validate.near_duplicate_pairs(records, same_family_only=False)[:50]},
            REPORTS / "dataset_validation.json",
        )

        for name, rows in by_split.items():
            s = stats["by_split"][name]
            print(f"  {name:11s} n={s['n']:4d}  families={s['scenario_families']:3d}  "
                  f"categories={len(s['categories'])}  "
                  f"priorities={s['priorities']}")

        # ---- 7. version file --------------------------------------------
        version_payload = {
            "dataset_version": DATASET_VERSION,
            "schema_version": SCHEMA_VERSION,
            "n_examples": len(records),
            "n_scenario_families": len(FAMILIES),
            "n_template_families": len(TEMPLATE_FAMILIES),
            "generation_seed": SEEDS["dataset_generation"],
            "split_seed": SEEDS["split_assignment"],
            "split_manifest_checksum": manifest["checksum"],
            "raw_sha256": sha256_file(dataio.RAW_DATASET),
            "processed_sha256": sha256_file(dataio.PROCESSED_DATASET),
            "split_manifest_sha256": sha256_file(dataio.SPLIT_MANIFEST),
            "licence": "MIT (synthetic data, generated by this repository)",
        }
        dataio.write_json(version_payload, dataio.DATASET_VERSION_FILE)

        run.metrics = {
            "n_examples": len(records),
            "split_counts": manifest["counts"],
            "validation_errors": post["errors"],
            "validation_warnings": post["warnings"],
            "split_manifest_checksum": manifest["checksum"],
        }
        run.inputs = {"dataset_fingerprint": dataio.dataset_fingerprint()}
        for name, p in (("raw", dataio.RAW_DATASET),
                        ("processed", dataio.PROCESSED_DATASET),
                        ("manifest", dataio.SPLIT_MANIFEST),
                        ("validation_report", REPORTS / "dataset_validation.json"),
                        ("stats", REPORTS / "dataset_stats.json")):
            run.add_artifact(name, p)
        run.finish("success")

        print(f"\nrun_id: {run.run_id}")
        print("dataset frozen. Do not regenerate without bumping DATASET_VERSION.")
        return 0

    except Exception as exc:  # noqa: BLE001 - we want the ledger to record it
        run.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
