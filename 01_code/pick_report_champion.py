"""Pick best experiment output folder by ensemble composite score; copy champion bundle."""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def composite_score(r: pd.Series) -> float:
    return float(
        0.4 * r["auc"] + 0.3 * r["balanced_accuracy_050"] + 0.3 * r["f1_050"]
    )


def load_ensemble_metrics(out_dir: Path) -> Tuple[float, Dict[str, float]]:
    p = out_dir / "metrics_endpoint_binary_test.csv"
    if not p.exists():
        return float("-inf"), {}
    df = pd.read_csv(p)
    rows = df[df["model"] == "model6_ensemble_endpoint"]
    if rows.empty:
        return float("-inf"), {}
    r = rows.iloc[0]
    keys = ["auc", "balanced_accuracy_050", "f1_050", "mcc_050", "brier", "ece"]
    vals = {k: float(r[k]) if pd.notna(r[k]) else float("nan") for k in keys if k in r}
    if not all(math.isfinite(vals.get(k, float("nan"))) for k in ["auc", "balanced_accuracy_050", "f1_050"]):
        return float("-inf"), vals
    return composite_score(r), vals


def main() -> None:
    root = Path(__file__).resolve().parent
    candidates: List[Path] = [
        root / "outputs_clinical_endpoint_n320_bs400",
        root / "outputs_clinical_endpoint_full_bs400",
        root / "outputs_clinical_endpoint_full_bs400_seeds5",
        root / "outputs_deep_full_gpu_clinical_endpoint",
    ]
    scored: List[Dict[str, object]] = []
    for d in candidates:
        if not d.is_dir():
            continue
        sc, vals = load_ensemble_metrics(d)
        scored.append({"dir": str(d), "composite_0.4AUC_0.3BA_0.3F1": sc, **vals})

    if not scored:
        raise SystemExit("no valid candidate outputs found")

    def sort_key(x: Dict[str, object]) -> Tuple[float, float, float]:
        comp = float(x["composite_0.4AUC_0.3BA_0.3F1"])
        mcc = float(x.get("mcc_050", float("nan")))
        brier = float(x.get("brier", float("nan")))
        mcc = mcc if math.isfinite(mcc) else -1e9
        brier = brier if math.isfinite(brier) else 1e9
        return (comp, mcc, -brier)

    scored.sort(key=sort_key, reverse=True)
    winner = Path(str(scored[0]["dir"]))
    tag = datetime.now().strftime("%Y%m%d_%H%M")
    dest = root / f"report_champion_bundle_{tag}"

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(winner, dest)

    scout = root / "cohort_scout_summary.json"
    if scout.exists():
        shutil.copy2(scout, dest / "cohort_scout_summary.json")

    sel = {
        "selection_rule": "max 0.4*AUC+0.3*balanced_accuracy_050+0.3*f1_050 on model6_ensemble_endpoint; tie-break mcc_050 then lower brier",
        "winner": str(winner),
        "bundle": str(dest),
        "ranked": scored,
    }
    (root / "report_champion_selection.json").write_text(
        json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    readme = dest / "README_CHAMPION.txt"
    readme.write_text(
        "Champion output copied by pick_report_champion.py\n"
        f"Source: {winner}\n"
        f"Rule: {sel['selection_rule']}\n"
        "See report_champion_selection.json in repo root for full ranking.\n",
        encoding="utf-8",
    )
    print(json.dumps({"winner": str(winner), "bundle": str(dest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
