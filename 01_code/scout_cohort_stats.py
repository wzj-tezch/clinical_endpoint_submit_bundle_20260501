"""One-off cohort size scout for deep experiment (merged labels × seg, feature extraction)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from deep_experiment_full_gpu import extract_features_from_seg, list_patients


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip nibabel-heavy feature scans; write label×seg merge sizes only.",
    )
    args = ap.parse_args()
    data_root = Path(r"E:\可用数据")
    label_xlsx = Path(r"E:\可用数据\survival_clean(1).xlsx")
    labels = pd.read_excel(label_xlsx)
    labels["patient_id"] = labels["patient_id"].astype(str).str.zfill(10)
    labels["time"] = pd.to_numeric(labels["time"], errors="coerce")
    labels["endpoint"] = pd.to_numeric(labels["endpoint"], errors="coerce")
    labels = labels.dropna(subset=["time", "endpoint"]).copy()
    labels["endpoint"] = labels["endpoint"].astype(int)

    pats = list_patients(data_root / "seg")
    p_rows = [{"sample_id": p.sample_id, "cohort": p.cohort, "patient_id": p.patient_id} for p in pats]
    p_df = pd.DataFrame(p_rows).merge(labels[["patient_id", "endpoint", "time"]], on="patient_id", how="inner")
    p_df = p_df.sort_values(["endpoint", "time"], ascending=[False, True]).copy()

    summary: dict = {
        "seg_patients_discovered": len(pats),
        "label_merge_rows": int(len(p_df)),
        "unique_patients_merged": int(p_df["patient_id"].nunique()),
        "note_merge_only_skipped_feature_audit": False,
        "feature_extract_success_rows": None,
        "feature_extract_failed_or_missing": None,
    }

    if args.merge_only:
        summary["note_merge_only_skipped_feature_audit"] = True
    else:
        p_map = {p.sample_id: p for p in pats}
        ok = 0
        fail = 0
        for sid in p_df["sample_id"].tolist():
            pf = p_map.get(sid)
            if pf is None:
                fail += 1
                continue
            try:
                f = extract_features_from_seg(pf)
                if f is not None:
                    ok += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
        summary["feature_extract_success_rows"] = ok
        summary["feature_extract_failed_or_missing"] = fail
    out = Path(__file__).resolve().parent / "cohort_scout_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
