import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.ndimage import center_of_mass
from scipy.signal import hilbert
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from report_figure_utils import (
    compute_dca_arrays,
    configure_academic_mpl,
    plot_bullseye_academic,
    plot_calibration_composite,
    plot_decision_curve,
    plot_roc_endpoint_multimodel,
    plot_time_dependent_aucs,
)

try:
    import xgboost as xgb
except Exception:
    xgb = None


FRAME_RE = re.compile(r"_(\d+)(?:_SEG)?\.nii\.gz$", re.IGNORECASE)


def resolve_cpu_jobs(requested: int) -> int:
    if requested is not None and requested > 0:
        return int(requested)
    total = os.cpu_count() or 4
    return max(1, total - 1)


def choose_compute_mode(mode: str, sample_count: int, cuda_available: bool) -> str:
    if mode == "cpu":
        return "cpu"
    if mode == "gpu":
        return "gpu" if cuda_available else "cpu"
    # auto mode: small sample often benefits from CPU due to transfer overhead.
    if cuda_available and sample_count >= 180:
        return "gpu"
    return "cpu"


def predict_xgb_proba(model, x_df: pd.DataFrame, use_gpu: bool) -> np.ndarray:
    if xgb is None:
        raise RuntimeError("xgboost is not available")
    if use_gpu:
        try:
            import cupy as cp

            x_cu = cp.asarray(x_df.values, dtype=cp.float32)
            pred = model.get_booster().inplace_predict(x_cu)
            return cp.asnumpy(pred).astype(float)
        except Exception:
            pass
    dm = xgb.DMatrix(x_df.values, feature_names=list(x_df.columns))
    return model.get_booster().predict(dm).astype(float)


def parse_eval_seeds(seed_text: str, base_seed: int) -> List[int]:
    if seed_text.strip():
        vals = []
        for x in seed_text.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                vals.append(int(x))
            except Exception:
                pass
        if vals:
            return vals
    return [base_seed, base_seed + 37, base_seed + 73]


def add_segment_boost_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seg_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    tr = train_df.copy()
    te = test_df.copy()
    eps = 1e-6

    ev = tr[tr["endpoint"] == 1]
    nv = tr[tr["endpoint"] == 0]
    if len(ev) == 0 or len(nv) == 0:
        top5 = seg_cols[:5]
    else:
        diffs = {c: float(ev[c].mean() - nv[c].mean()) for c in seg_cols}
        top5 = [k for k, _ in sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]]
    top3 = top5[:3]

    def add_block(df: pd.DataFrame) -> pd.DataFrame:
        df["seg_top3_mean"] = df[top3].mean(axis=1)
        df["seg_top5_mean"] = df[top5].mean(axis=1)
        df["seg_top5_std"] = df[top5].std(axis=1).fillna(0.0)
        df["seg_top5_max"] = df[top5].max(axis=1)
        df["seg_base_mean"] = df[[f"seg{i:02d}_amp" for i in range(1, 7)]].mean(axis=1)
        df["seg_mid_mean"] = df[[f"seg{i:02d}_amp" for i in range(7, 13)]].mean(axis=1)
        df["seg_apex_mean"] = df[[f"seg{i:02d}_amp" for i in range(13, 18)]].mean(axis=1)
        df["seg_base_apex_ratio"] = df["seg_base_mean"] / (np.abs(df["seg_apex_mean"]) + eps)
        df["seg_dispersion_cv"] = df[seg_cols].std(axis=1) / (np.abs(df[seg_cols].mean(axis=1)) + eps)
        df["DeltaTheta_x_top5"] = df["DeltaTheta"] * df["seg_top5_mean"]
        df["Rdiast_x_ratio"] = df["Rdiast"] * df["seg_base_apex_ratio"]
        return df

    tr = add_block(tr)
    te = add_block(te)
    boost_cols = [
        "seg_top3_mean",
        "seg_top5_mean",
        "seg_top5_std",
        "seg_top5_max",
        "seg_base_mean",
        "seg_mid_mean",
        "seg_apex_mean",
        "seg_base_apex_ratio",
        "seg_dispersion_cv",
        "DeltaTheta_x_top5",
        "Rdiast_x_ratio",
    ]
    return tr, te, boost_cols, top5


def composite_clinical_tune_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Inner-CV tuning: 0.4*AUC + 0.3*BalancedAcc + 0.3*F1 (prob阈值=0.5)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = float("nan")
    yhat = (probs >= 0.5).astype(int)
    ba = float(balanced_accuracy_score(y_true, yhat))
    f1 = float(f1_score(y_true, yhat, zero_division=0))
    if not np.isfinite(auc):
        auc = ba
    return 0.4 * auc + 0.3 * ba + 0.3 * f1


def clinical_binary_at_threshold(y_true: np.ndarray, probs: np.ndarray, thr: float) -> Dict[str, float]:
    yhat = (probs >= thr).astype(int)
    tp = float(np.sum((yhat == 1) & (y_true == 1)))
    tn = float(np.sum((yhat == 0) & (y_true == 0)))
    fp = float(np.sum((yhat == 1) & (y_true == 0)))
    fn = float(np.sum((yhat == 0) & (y_true == 1)))
    sens = tp / max(tp + fn, 1.0)
    spec = tn / max(tn + fp, 1.0)
    ppv = tp / max(tp + fp, 1.0)
    npv = tn / max(tn + fn, 1.0)
    lr_pos = sens / max(1.0 - spec, 1e-8)
    lr_neg = (1.0 - sens) / max(spec, 1e-8)
    return {
        "threshold": float(thr),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, yhat)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, yhat)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "lr_pos": float(lr_pos),
        "lr_neg": float(lr_neg),
        "youden_j": float(sens + spec - 1.0),
    }


def scan_thresholds_for_clinical(
    y_true: np.ndarray, probs: np.ndarray, sens_floor: float = 0.75, spec_floor: float = 0.75
) -> Tuple[Dict[str, float], List[Tuple[str, Dict[str, float]]]]:
    """主阈值=max balanced accuracy；另给出 Youden、偏保守（高特异）、偏积极（高敏感）。"""
    p = np.asarray(probs, dtype=float)
    uniq = np.sort(np.unique(p))
    cand: List[Tuple[str, Dict[str, float]]] = []
    if uniq.size < 2:
        thr = float(np.median(p)) if p.size else 0.5
        m = clinical_binary_at_threshold(y_true, probs, thr)
        m["rule"] = "single_prob_mass"
        cand.append(("single_prob_mass", dict(m)))
        return dict(m), cand
    mids = [(uniq[i] + uniq[i + 1]) / 2.0 for i in range(len(uniq) - 1)]
    mids = list(dict.fromkeys(mids))[:301]
    if not mids:
        m = clinical_binary_at_threshold(y_true, probs, 0.5)
        m["rule"] = "fallback_0.5"
        cand.append(("fallback_0.5", dict(m)))
        return dict(m), cand
    rows: List[Tuple[float, Dict[str, float]]] = []
    for thr in mids:
        m = clinical_binary_at_threshold(y_true, probs, float(thr))
        rows.append((float(thr), dict(m)))
    best_bal = max(rows, key=lambda x: x[1]["balanced_accuracy"])
    best_youd = max(rows, key=lambda x: x[1]["youden_j"])
    cons_pool = [m for _, m in rows if m["sensitivity"] >= sens_floor]
    best_cons = max(cons_pool, key=lambda m: m["specificity"]) if cons_pool else best_bal[1]
    agg_pool = [m for _, m in rows if m["specificity"] >= spec_floor]
    best_agg = max(agg_pool, key=lambda m: m["sensitivity"]) if agg_pool else best_bal[1]
    primary = dict(best_bal[1])
    primary["rule"] = "max_balanced_accuracy"
    youden = dict(best_youd[1])
    youden["rule"] = "max_youden_j"
    conservative = dict(best_cons)
    conservative["rule"] = "conservative_high_specificity"
    aggressive = dict(best_agg)
    aggressive["rule"] = "aggressive_high_sensitivity"
    cand = [
        ("max_balanced_accuracy", primary),
        ("max_youden_j", youden),
        ("conservative_high_specificity", conservative),
        ("aggressive_high_sensitivity", aggressive),
    ]
    return primary, cand


def gather_oof_tree_mlp(
    train_df: pd.DataFrame,
    features: List[str],
    y_bin: np.ndarray,
    seed: int,
    use_gpu: bool,
    n_jobs: int,
    xgb_params: Dict[str, float],
    mlp_params: Dict[str, float],
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    oof_tree = np.full(len(train_df), np.nan, dtype=float)
    oof_mlp = np.full(len(train_df), np.nan, dtype=float)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for fold_idx, (tr, va) in enumerate(skf.split(train_df, y_bin)):
        tr_df = train_df.iloc[tr].copy()
        va_df = train_df.iloc[va].copy()
        y_tr = y_bin[tr]
        if len(np.unique(y_tr)) < 2:
            continue
        tree_p, _ = fit_tree_predict(
            train_df=tr_df,
            pred_df=va_df,
            features=features,
            y_train=y_tr,
            seed=seed + fold_idx + 77,
            use_gpu=use_gpu,
            n_jobs=n_jobs,
            xgb_params=xgb_params,
        )
        mlp_p = train_torch_binary(
            tr_df[features].values,
            y_tr.astype(float),
            va_df[features].values,
            device=device,
            seed=seed + fold_idx + 177,
            **mlp_params,
        )
        oof_tree[va] = tree_p
        oof_mlp[va] = mlp_p
    return oof_tree, oof_mlp


def tune_ensemble_weight(
    y_bin: np.ndarray,
    oof_tree: np.ndarray,
    oof_mlp: np.ndarray,
    train_mask: np.ndarray,
) -> float:
    if len(np.unique(y_bin[train_mask])) < 2:
        return 0.5
    best_w = 0.5
    best_s = -1e9
    for w in np.linspace(0.0, 1.0, 21):
        p = np.full(len(y_bin), np.nan, dtype=float)
        mt = train_mask & np.isfinite(oof_tree) & np.isfinite(oof_mlp)
        p[mt] = float(w) * oof_mlp[mt] + (1.0 - float(w)) * oof_tree[mt]
        s = composite_clinical_tune_score(y_bin[mt], p[mt])
        if np.isfinite(s) and s > best_s:
            best_s = s
            best_w = float(w)
    return best_w


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true).astype(float)
    p = np.asarray(probs).astype(float)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p)
    p_sorted, y_sorted = p[order], y[order]
    n = len(p_sorted)
    bin_size = max(1, n // n_bins)
    ece = 0.0
    lo = 0
    while lo < n:
        hi = min(n, lo + bin_size)
        mask = slice(lo, hi)
        conf = float(np.mean(p_sorted[mask]))
        acc = float(np.mean(y_sorted[mask]))
        w = float((hi - lo) / n)
        ece += w * abs(acc - conf)
        lo = hi
    return float(ece)


def calibration_logit_slope_intercept(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    """对预测概率做 logit，拟合单层 logistic 校准：logit(calibrated) ~ slope * logit(p) + intercept。"""
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=500)
    lr.fit(logit, y)
    return float(lr.coef_.ravel()[0]), float(lr.intercept_.ravel()[0])


@dataclass
class PatientFrames:
    cohort: str
    patient_id: str
    seg_frames: Dict[int, Path]

    @property
    def sample_id(self) -> str:
        return f"{self.cohort}__{self.patient_id}"


def parse_frame_index(path: Path) -> Optional[int]:
    m = FRAME_RE.search(path.name)
    if m is None:
        return None
    return int(m.group(1))


def list_patients(seg_root: Path) -> List[PatientFrames]:
    out: List[PatientFrames] = []
    for cohort_dir in sorted([d for d in seg_root.iterdir() if d.is_dir()]):
        for p_dir in sorted([d for d in cohort_dir.iterdir() if d.is_dir()]):
            seg_files = {}
            for f in p_dir.glob("*.nii.gz"):
                idx = parse_frame_index(f)
                if idx is not None:
                    seg_files[idx] = f
            common = sorted(seg_files.keys())
            if len(common) < 10:
                continue
            out.append(
                PatientFrames(
                    cohort=cohort_dir.name,
                    patient_id=p_dir.name.zfill(10),
                    seg_frames={i: seg_files[i] for i in common},
                )
            )
    return out


def assign_aha17_segment(z_idx: int, z_len: int, angle: float, is_apex_center: bool) -> int:
    if is_apex_center:
        return 17
    basal_cut = int(round(z_len * 0.33))
    mid_cut = int(round(z_len * 0.66))
    angle = (angle + 2 * math.pi) % (2 * math.pi)
    if z_idx < basal_cut:
        return int(angle / (2 * math.pi / 6)) + 1
    if z_idx < mid_cut:
        return int(angle / (2 * math.pi / 6)) + 7
    return int(angle / (2 * math.pi / 4)) + 13


def segment_time_series(mask_seq: List[np.ndarray]) -> Dict[int, np.ndarray]:
    t_len = len(mask_seq)
    series = {k: np.zeros(t_len, dtype=float) for k in range(1, 18)}
    if t_len == 0:
        return series
    ed_mask = mask_seq[0]
    vox = np.argwhere(ed_mask > 0)
    if vox.size == 0:
        return series
    ed_com = vox.mean(axis=0)
    z_len = ed_mask.shape[2]
    for t, mask in enumerate(mask_seq):
        cur = np.argwhere(mask > 0)
        if cur.size == 0:
            continue
        bucket = {k: [] for k in range(1, 18)}
        for x, y, z in cur:
            dx = x - ed_com[0]
            dy = y - ed_com[1]
            ang = math.atan2(dy, dx)
            is_apex_center = (
                z >= int(round(z_len * 0.9))
                and abs(dx) < mask.shape[0] * 0.05
                and abs(dy) < mask.shape[1] * 0.05
            )
            seg = assign_aha17_segment(z, z_len, ang, is_apex_center)
            bucket[seg].append(math.sqrt(dx * dx + dy * dy))
        for seg in range(1, 18):
            if bucket[seg]:
                series[seg][t] = float(np.mean(bucket[seg]))
            elif t > 0:
                series[seg][t] = series[seg][t - 1]
    return series


def phase_desync_index(seg_series: Dict[int, np.ndarray]) -> float:
    phases = []
    for seg in range(1, 18):
        s = seg_series[seg]
        if np.std(s) < 1e-8:
            continue
        feat = np.column_stack([s, np.gradient(s)])
        u, _, _ = np.linalg.svd(feat - feat.mean(0), full_matrices=False)
        pc1 = u[:, 0]
        ana = hilbert(pc1 - pc1.mean())
        phase = np.unwrap(np.angle(ana))
        phases.append(phase)
    if len(phases) < 2:
        return 0.0
    mat = np.stack(phases, axis=0)
    vals = []
    for t in range(mat.shape[1]):
        phi = mat[:, t]
        d = np.abs(phi[:, None] - phi[None, :])
        vals.append(float(np.max(d)))
    return float(np.std(vals))


def discrete_curvature(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.zeros(1, dtype=float)
    p_prev = points[:-2]
    p_cur = points[1:-1]
    p_nxt = points[2:]
    v1 = p_cur - p_prev
    v2 = p_nxt - p_cur
    cr = np.linalg.norm(np.cross(v1, v2), axis=1)
    dn = np.maximum(np.linalg.norm(v1, axis=1) ** 3, 1e-8)
    return cr / dn


def extract_features_from_seg(pf: PatientFrames) -> Optional[Dict[str, float]]:
    frame_ids = sorted(pf.seg_frames.keys())
    myo_seq = []
    for idx in frame_ids:
        seg = nib.load(str(pf.seg_frames[idx])).get_fdata()
        myo = (seg == 2).astype(np.uint8)
        if myo.sum() == 0:
            return None
        myo_seq.append(myo)
    t_len = len(myo_seq)
    vols = np.array([float(m.sum()) for m in myo_seq], dtype=float)
    es_idx = int(np.argmax(vols))
    diast = list(range(es_idx, t_len))
    syst = list(range(0, es_idx + 1))
    centers = np.array([center_of_mass(m) for m in myo_seq], dtype=float)
    disp = np.linalg.norm(centers - centers[0], axis=1)
    ldiast = float(np.sum(np.abs(np.diff(disp[diast])))) if len(diast) > 1 else 0.0
    rdiast = float(ldiast / max(len(diast) - 1, 1))
    kappa = float(np.mean(discrete_curvature(centers[syst]))) if len(syst) >= 3 else 0.0
    eclose = float(np.linalg.norm(centers[-1] - centers[0]))
    seg_series = segment_time_series(myo_seq)
    delta_theta = phase_desync_index(seg_series)
    out = {
        "sample_id": pf.sample_id,
        "cohort": pf.cohort,
        "patient_id": pf.patient_id,
        "n_frames": float(t_len),
        "es_idx": float(es_idx),
        "myo_vol_mean": float(np.mean(vols)),
        "myo_vol_std": float(np.std(vols)),
        "Ldiast": ldiast,
        "Rdiast": rdiast,
        "kappa": kappa,
        "Eclose": eclose,
        "DeltaTheta": delta_theta,
    }
    for seg in range(1, 18):
        out[f"seg{seg:02d}_amp"] = float(np.max(seg_series[seg]) - np.min(seg_series[seg]))
    return out


def cindex_raw_and_adjusted(time_arr: np.ndarray, event_arr: np.ndarray, risk_arr: np.ndarray) -> Tuple[float, float]:
    c_raw = float(concordance_index(time_arr, -risk_arr, event_arr))
    return c_raw, float(max(c_raw, 1.0 - c_raw))


def nri(r_old: np.ndarray, r_new: np.ndarray, y: np.ndarray) -> float:
    em = y == 1
    nm = y == 0
    if em.sum() == 0 or nm.sum() == 0:
        return float("nan")
    ue = np.mean((r_new[em] - r_old[em]) > 0)
    de = np.mean((r_new[em] - r_old[em]) < 0)
    un = np.mean((r_new[nm] - r_old[nm]) > 0)
    dn = np.mean((r_new[nm] - r_old[nm]) < 0)
    return float((ue - de) + (dn - un))


def idi(r_old: np.ndarray, r_new: np.ndarray, y: np.ndarray) -> float:
    em = y == 1
    nm = y == 0
    if em.sum() == 0 or nm.sum() == 0:
        return float("nan")
    return float((r_new[em].mean() - r_new[nm].mean()) - (r_old[em].mean() - r_old[nm].mean()))


def fit_cox(train_df: pd.DataFrame, cov: List[str], penalizer: float, l1_ratio: float) -> Tuple[CoxPHFitter, List[str]]:
    usable = [c for c in cov if c in train_df.columns and float(train_df[c].var()) > 1e-4]
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(train_df[usable + ["time", "endpoint"]], duration_col="time", event_col="endpoint", show_progress=False)
    return cph, usable


def select_cox_params(train_df: pd.DataFrame, cov: List[str], seed: int) -> Tuple[float, float]:
    grid = [(0.01, 0.0), (0.05, 0.0), (0.1, 0.0), (0.2, 0.5), (0.5, 0.5), (1.0, 0.8)]
    y = train_df["endpoint"].values.astype(int)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best = grid[0]
    best_score = -1e9
    for p, l1 in grid:
        vals = []
        for tr, va in skf.split(train_df, y):
            tr_df = train_df.iloc[tr].copy()
            va_df = train_df.iloc[va].copy()
            try:
                m, u = fit_cox(tr_df, cov, p, l1)
                risk = m.predict_partial_hazard(va_df[u]).values.ravel()
                _, c = cindex_raw_and_adjusted(va_df["time"].values, va_df["endpoint"].values, risk)
                vals.append(c)
            except Exception:
                vals.append(np.nan)
        s = np.nanmean(vals)
        if np.isfinite(s) and s > best_score:
            best_score = s
            best = (p, l1)
    return best


class TorchMLP(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_torch_binary(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    device: str,
    seed: int,
    hidden1: int = 64,
    hidden2: int = 32,
    dropout: float = 0.15,
    lr: float = 1e-3,
    epochs: int = 120,
    patience: int = 20,
) -> np.ndarray:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    model = TorchMLP(x_tr.shape[1]).to(device)
    # override simple architecture widths
    model.net = torch.nn.Sequential(
        torch.nn.Linear(x_tr.shape[1], hidden1),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden1, hidden2),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden2, 1),
    ).to(device)
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([max(neg / max(pos, 1.0), 1.0)], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    xt = torch.tensor(x_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    xv = torch.tensor(x_va, dtype=torch.float32, device=device)
    model.train()
    best_loss = float("inf")
    stale = 0
    for _ in range(epochs):
        opt.zero_grad()
        lg = model(xt)
        loss = loss_fn(lg, yt)
        loss.backward()
        opt.step()
        cur = float(loss.detach().cpu().item())
        if cur + 1e-6 < best_loss:
            best_loss = cur
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(xv)).detach().cpu().numpy()
    return prob


def select_xgb_params(
    train_df: pd.DataFrame,
    features: List[str],
    y_bin: np.ndarray,
    seed: int,
    device: str,
    n_jobs: int,
) -> Dict[str, float]:
    if xgb is None:
        return {}
    grid = [
        {"max_depth": 3, "min_child_weight": 1.0, "learning_rate": 0.05, "n_estimators": 350, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.0},
        {"max_depth": 4, "min_child_weight": 1.0, "learning_rate": 0.03, "n_estimators": 550, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.5},
        {"max_depth": 5, "min_child_weight": 2.0, "learning_rate": 0.03, "n_estimators": 700, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {"max_depth": 6, "min_child_weight": 2.0, "learning_rate": 0.02, "n_estimators": 900, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 3.0},
        {"max_depth": 4, "min_child_weight": 3.0, "learning_rate": 0.04, "n_estimators": 500, "subsample": 0.95, "colsample_bytree": 0.9, "reg_lambda": 1.0},
    ]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best = grid[0]
    best_score = -1e9
    for g in grid:
        fold_scores = []
        for tr, va in skf.split(train_df, y_bin):
            xtr = train_df.iloc[tr][features]
            xva = train_df.iloc[va][features]
            ytr = y_bin[tr]
            yva = y_bin[va]
            if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
                continue
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                device=device,
                n_jobs=n_jobs,
                random_state=seed,
                **g,
            )
            model.fit(xtr, ytr)
            p = predict_xgb_proba(model, xva, use_gpu=(device == "cuda"))
            fold_scores.append(composite_clinical_tune_score(yva, p))
        score = np.mean(fold_scores) if fold_scores else np.nan
        if np.isfinite(score) and score > best_score:
            best_score = score
            best = g
    return best


def select_rf_params(train_df: pd.DataFrame, features: List[str], y_bin: np.ndarray, seed: int, n_jobs: int) -> Dict[str, float]:
    grid = [
        {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 2},
        {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 2},
        {"n_estimators": 700, "max_depth": 10, "min_samples_leaf": 1},
    ]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best = grid[0]
    best_score = -1e9
    for g in grid:
        fold_scores = []
        for tr, va in skf.split(train_df, y_bin):
            xtr = train_df.iloc[tr][features]
            xva = train_df.iloc[va][features]
            ytr = y_bin[tr]
            yva = y_bin[va]
            if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
                continue
            m = RandomForestClassifier(random_state=seed, class_weight="balanced", n_jobs=n_jobs, **g)
            m.fit(xtr, ytr)
            p = m.predict_proba(xva)[:, 1]
            fold_scores.append(composite_clinical_tune_score(yva, p))
        score = np.mean(fold_scores) if fold_scores else np.nan
        if np.isfinite(score) and score > best_score:
            best_score = score
            best = g
    return best


def select_mlp_params(train_df: pd.DataFrame, features: List[str], y_bin: np.ndarray, seed: int, device: str) -> Dict[str, float]:
    grid = [
        {"hidden1": 64, "hidden2": 32, "dropout": 0.15, "lr": 1e-3, "epochs": 140, "patience": 20},
        {"hidden1": 96, "hidden2": 48, "dropout": 0.2, "lr": 8e-4, "epochs": 180, "patience": 25},
        {"hidden1": 128, "hidden2": 64, "dropout": 0.25, "lr": 6e-4, "epochs": 220, "patience": 30},
        {"hidden1": 160, "hidden2": 96, "dropout": 0.3, "lr": 5e-4, "epochs": 240, "patience": 30},
        {"hidden1": 80, "hidden2": 40, "dropout": 0.1, "lr": 1.2e-3, "epochs": 120, "patience": 18},
    ]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best = grid[0]
    best_score = -1e9
    for g in grid:
        fold_scores = []
        for tr, va in skf.split(train_df, y_bin):
            xtr = train_df.iloc[tr][features].values
            xva = train_df.iloc[va][features].values
            ytr = y_bin[tr].astype(float)
            yva = y_bin[va].astype(int)
            if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
                continue
            p = train_torch_binary(xtr, ytr, xva, device=device, seed=seed, **g)
            fold_scores.append(composite_clinical_tune_score(yva, p))
        score = np.mean(fold_scores) if fold_scores else np.nan
        if np.isfinite(score) and score > best_score:
            best_score = score
            best = g
    return best


def fit_tree_predict(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    features: List[str],
    y_train: np.ndarray,
    seed: int,
    use_gpu: bool,
    n_jobs: int,
    xgb_params: Dict[str, float],
) -> Tuple[np.ndarray, str]:
    uniq = np.unique(y_train)
    if uniq.size < 2:
        return np.full(len(pred_df), float(uniq[0]), dtype=float), "constant_fallback_single_class"
    if xgb is not None:
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            n_jobs=n_jobs,
            random_state=seed,
            **xgb_params,
        )
        model.fit(train_df[features], y_train)
        pred = predict_xgb_proba(model, pred_df[features], use_gpu=use_gpu)
        return pred, "xgboost"

    rf = RandomForestClassifier(
        n_estimators=700,
        max_depth=10,
        min_samples_leaf=1,
        random_state=seed,
        class_weight="balanced",
        n_jobs=n_jobs,
    )
    rf.fit(train_df[features], y_train)
    probs = rf.predict_proba(pred_df[features])
    if probs.shape[1] == 1:
        cls = int(rf.classes_[0])
        return np.full(len(pred_df), 1.0 if cls == 1 else 0.0, dtype=float), "random_forest_fallback"
    pos_idx = int(np.where(rf.classes_ == 1)[0][0])
    return probs[:, pos_idx], "random_forest_fallback"


def select_ensemble_weight(
    train_df: pd.DataFrame,
    features: List[str],
    y_bin: np.ndarray,
    seed: int,
    use_gpu: bool,
    n_jobs: int,
    xgb_params: Dict[str, float],
    mlp_params: Dict[str, float],
) -> Tuple[float, np.ndarray, np.ndarray]:
    if len(np.unique(y_bin)) < 2:
        return 0.5, np.full(len(train_df), np.nan), np.full(len(train_df), np.nan)
    device = "cuda" if use_gpu else "cpu"
    oof_tree, oof_mlp = gather_oof_tree_mlp(
        train_df,
        features,
        y_bin,
        seed,
        use_gpu,
        n_jobs,
        xgb_params,
        mlp_params,
        device,
    )
    mask = np.isfinite(oof_tree) & np.isfinite(oof_mlp)
    if mask.sum() < 10 or len(np.unique(y_bin[mask])) < 2:
        return 0.5, oof_tree, oof_mlp
    w = tune_ensemble_weight(y_bin, oof_tree, oof_mlp, mask)
    return w, oof_tree, oof_mlp


def bootstrap_ci(values: List[float]) -> Tuple[float, float, float]:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description="扩样本GPU深化实验")
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\可用数据"))
    parser.add_argument("--label-xlsx", type=Path, default=Path(r"E:\可用数据\survival_clean(1).xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"e:\本基中间结果\outputs_deep_full_gpu_clinical_endpoint"))
    parser.add_argument("--min-samples", type=int, default=320)
    parser.add_argument(
        "--full-cohort",
        action="store_true",
        help="不使用 min-samples 截断；纳入 label×seg 合并后可匹配的全部患者（去重 patient_id）。",
    )
    parser.add_argument("--min-events-3y", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--compute", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--cpu-jobs", type=int, default=0, help="CPU线程数，0表示自动(核数-1)")
    parser.add_argument(
        "--eval-seeds",
        type=str,
        default="",
        help="多seed复跑列表，逗号分隔；留空则使用默认3个seed",
    )
    parser.add_argument(
        "--oof-isotonic-calibration",
        action="store_true",
        help="在训练折 OOF 融合概率上拟合 isotonic；外测导出校准附加指标与阈值表（不改变主 preds 文件名）。",
    )
    args = parser.parse_args()
    cpu_jobs = resolve_cpu_jobs(args.cpu_jobs)
    torch.set_num_threads(cpu_jobs)
    eval_seeds = parse_eval_seeds(args.eval_seeds, args.seed)

    out = args.output_dir
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    configure_academic_mpl()

    labels = pd.read_excel(args.label_xlsx)
    labels["patient_id"] = labels["patient_id"].astype(str).str.zfill(10)
    labels["time"] = pd.to_numeric(labels["time"], errors="coerce")
    labels["endpoint"] = pd.to_numeric(labels["endpoint"], errors="coerce")
    labels = labels.dropna(subset=["time", "endpoint"]).copy()
    labels["endpoint"] = labels["endpoint"].astype(int)

    pats = list_patients(args.data_root / "seg")
    p_rows = [{"sample_id": p.sample_id, "cohort": p.cohort, "patient_id": p.patient_id} for p in pats]
    p_df = pd.DataFrame(p_rows).merge(labels[["patient_id", "endpoint", "time"]], on="patient_id", how="inner")
    p_df = p_df.sort_values(["endpoint", "time"], ascending=[False, True]).copy()

    if args.full_cohort:
        sel_df = p_df.drop_duplicates(subset=["patient_id"]).copy().reset_index(drop=True)
    else:
        # event-priority + cohort balance sampling
        event_df = p_df[(p_df["endpoint"] == 1) & (p_df["time"] <= 1095)].copy()
        nonevent_df = p_df.drop(event_df.index).copy()
        selected = []
        used = set()
        cohorts = sorted(p_df["cohort"].unique().tolist())
        quota = max(int(args.min_samples / max(len(cohorts), 1)), 1)
        for c in cohorts:
            sub_e = event_df[event_df["cohort"] == c]
            for _, r in sub_e.iterrows():
                if r["patient_id"] in used:
                    continue
                selected.append(r)
                used.add(r["patient_id"])
                if len([x for x in selected if x["cohort"] == c]) >= quota:
                    break
        for _, r in event_df.iterrows():
            if len(selected) >= args.min_samples:
                break
            if r["patient_id"] in used:
                continue
            selected.append(r)
            used.add(r["patient_id"])
        for _, r in nonevent_df.iterrows():
            if len(selected) >= args.min_samples:
                break
            if r["patient_id"] in used:
                continue
            selected.append(r)
            used.add(r["patient_id"])

        sel_df = pd.DataFrame(selected).drop_duplicates(subset=["patient_id"]).copy()
        ev3 = int(((sel_df["endpoint"] == 1) & (sel_df["time"] <= 1095)).sum())
        if ev3 < args.min_events_3y:
            # top up by 3y events first
            remain = event_df[~event_df["patient_id"].isin(sel_df["patient_id"])]
            for _, r in remain.iterrows():
                sel_df = pd.concat([sel_df, r.to_frame().T], ignore_index=True)
                ev3 = int(((sel_df["endpoint"] == 1) & (sel_df["time"] <= 1095)).sum())
                if ev3 >= args.min_events_3y:
                    break

        sel_df = sel_df.drop_duplicates(subset=["patient_id"]).reset_index(drop=True)

    sel_df.to_csv(out / "selected_cohort.csv", index=False, encoding="utf-8-sig")

    # map sample_id->PatientFrames
    p_map = {p.sample_id: p for p in pats}
    feats = []
    for sid in sel_df["sample_id"].tolist():
        pf = p_map.get(sid)
        if pf is None:
            continue
        try:
            f = extract_features_from_seg(pf)
            if f is not None:
                feats.append(f)
        except Exception:
            continue
    feat_df = pd.DataFrame(feats)
    lab_mini = sel_df[["sample_id", "cohort", "endpoint", "time"]].drop_duplicates(subset=["sample_id"], keep="first")
    feat_base = feat_df.drop(columns=["cohort"], errors="ignore")
    data = feat_base.merge(lab_mini, on="sample_id", how="inner")
    data.to_csv(out / "features_with_labels_full.csv", index=False, encoding="utf-8-sig")

    # protocol freeze
    protocol = {
        "seed": args.seed,
        "time_unit": "days",
        "horizons_days": [365, 730, 1095],
        "selected_samples": int(len(data)),
        "selected_events_total": int(data["endpoint"].sum()),
        "selected_events_3y": int(((data["endpoint"] == 1) & (data["time"] <= 1095)).sum()),
        "endpoint_binary_task_note": "树/MLP/融合的训练标签仅为 endpoint∈{0,1}，不再使用3年二元代理标签。",
        "compute_requested": args.compute,
        "cpu_jobs": cpu_jobs,
        "eval_seeds": eval_seeds,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "xgboost_available": bool(xgb is not None),
        "full_cohort_mode": bool(args.full_cohort),
        "requested_min_samples_cap": args.min_samples,
        "label_seg_merge_candidates_unique_patients": int(p_df["patient_id"].nunique()),
    }
    with open(out / "protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    base_cov = ["myo_vol_mean", "myo_vol_std"]
    geom_cov = base_cov + ["Ldiast", "Rdiast", "kappa", "Eclose", "DeltaTheta"]
    seg_cov = [f"seg{i:02d}_amp" for i in range(1, 18)]
    full_cov = geom_cov + seg_cov

    # fixed outer split
    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.25,
        random_state=args.seed,
        stratify=data["endpoint"].values.astype(int),
    )
    train_df = data.iloc[train_idx].copy().reset_index(drop=True)
    test_df = data.iloc[test_idx].copy().reset_index(drop=True)
    train_df.to_csv(out / "outer_train.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(out / "outer_test.csv", index=False, encoding="utf-8-sig")

    # robust scale using train stats
    scaler = RobustScaler()
    train_df[full_cov] = scaler.fit_transform(train_df[full_cov])
    test_df[full_cov] = scaler.transform(test_df[full_cov])

    train_df, test_df, boost_cov, stable_top5 = add_segment_boost_features(train_df, test_df, seg_cov)
    boost_cov_core = [
        "seg_top3_mean",
        "seg_top5_mean",
        "seg_base_apex_ratio",
        "seg_dispersion_cv",
        "DeltaTheta_x_top5",
    ]
    model_cov = full_cov + [c for c in boost_cov_core if c in boost_cov]
    train_df.to_csv(out / "outer_train_enhanced.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(out / "outer_test_enhanced.csv", index=False, encoding="utf-8-sig")

    y_train_bin = train_df["endpoint"].astype(int).values
    y_test_bin = test_df["endpoint"].astype(int).values
    protocol_endpoint = {
        "task": "binary_endpoint",
        "label": "endpoint (1=event occurred, 0=no event by study definition)",
        "train_n": int(len(train_df)),
        "train_positive_n": int(y_train_bin.sum()),
        "train_positive_rate": float(y_train_bin.mean()),
        "test_n": int(len(test_df)),
        "test_positive_n": int(y_test_bin.sum()),
        "test_positive_rate": float(y_test_bin.mean()),
        "outer_test_size": 0.25,
        "outer_stratify": "endpoint",
        "outer_split_random_state": int(args.seed),
        "cohort_counts_selected": data.groupby("cohort").size().astype(int).to_dict(),
    }
    with open(out / "protocol_endpoint.json", "w", encoding="utf-8") as f:
        json.dump(protocol_endpoint, f, ensure_ascii=False, indent=2)

    compute_mode = choose_compute_mode(args.compute, sample_count=len(train_df), cuda_available=torch.cuda.is_available())
    use_gpu = compute_mode == "gpu"

    # Cox models with nested tuning
    p1, l1_1 = select_cox_params(train_df, base_cov, seed=args.seed + 1)
    m1, u1 = fit_cox(train_df, base_cov, p1, l1_1)
    risk1 = m1.predict_partial_hazard(test_df[u1]).values.ravel()
    c1_raw, c1 = cindex_raw_and_adjusted(test_df["time"].values, test_df["endpoint"].values, risk1)

    p2, l1_2 = select_cox_params(train_df, geom_cov, seed=args.seed + 2)
    m2, u2 = fit_cox(train_df, geom_cov, p2, l1_2)
    risk2 = m2.predict_partial_hazard(test_df[u2]).values.ravel()
    c2_raw, c2 = cindex_raw_and_adjusted(test_df["time"].values, test_df["endpoint"].values, risk2)

    p3, l1_3 = select_cox_params(train_df, model_cov, seed=args.seed + 3)
    m3, u3 = fit_cox(train_df, model_cov, p3, l1_3)
    risk3 = m3.predict_partial_hazard(test_df[u3]).values.ravel()
    c3_raw, c3 = cindex_raw_and_adjusted(test_df["time"].values, test_df["endpoint"].values, risk3)

    # XGBoost / RF + MLP: 标签为 endpoint 二分类（非 3 年窗代理）
    y_bin_unique = np.unique(y_train_bin)
    xgb_best = {}
    tree_backend = "xgboost"
    if y_bin_unique.size < 2:
        const_prob = float(y_bin_unique[0])
        xgb_prob = np.full(len(test_df), const_prob, dtype=float)
        tree_backend = "constant_fallback_single_class"
        xgb_best = {"constant_prob": const_prob}
    elif xgb is not None:
        xgb_best = select_xgb_params(
            train_df=train_df,
            features=model_cov,
            y_bin=y_train_bin,
            seed=args.seed + 10,
            device="cuda" if use_gpu else "cpu",
            n_jobs=cpu_jobs,
        )
        xgb_prob, tree_backend = fit_tree_predict(
            train_df=train_df,
            pred_df=test_df,
            features=model_cov,
            y_train=y_train_bin,
            seed=args.seed,
            use_gpu=use_gpu,
            n_jobs=cpu_jobs,
            xgb_params=xgb_best,
        )
    else:
        tree_backend = "random_forest_fallback"
        rf_best = select_rf_params(
            train_df=train_df,
            features=model_cov,
            y_bin=y_train_bin,
            seed=args.seed + 11,
            n_jobs=cpu_jobs,
        )
        xgb_best = rf_best
        rf_model = RandomForestClassifier(
            random_state=args.seed,
            class_weight="balanced",
            n_jobs=cpu_jobs,
            **rf_best,
        )
        rf_model.fit(train_df[model_cov], y_train_bin)
        probs = rf_model.predict_proba(test_df[model_cov])
        if probs.shape[1] == 1:
            cls = int(rf_model.classes_[0])
            xgb_prob = np.full(len(test_df), 1.0 if cls == 1 else 0.0, dtype=float)
        else:
            pos_idx = int(np.where(rf_model.classes_ == 1)[0][0])
            xgb_prob = probs[:, pos_idx]
    c_xgb_raw, c_xgb = cindex_raw_and_adjusted(test_df["time"].values, test_df["endpoint"].values, xgb_prob)

    # Torch MLP with inner tuning
    device = "cuda" if use_gpu else "cpu"
    if y_bin_unique.size < 2:
        mlp_best = {"constant_prob": float(y_bin_unique[0])}
        mlp_prob = np.full(len(test_df), float(y_bin_unique[0]), dtype=float)
    else:
        mlp_best = select_mlp_params(
            train_df=train_df,
            features=model_cov,
            y_bin=y_train_bin,
            seed=args.seed + 20,
            device=device,
        )
        mlp_prob = train_torch_binary(
            train_df[model_cov].values,
            y_train_bin.astype(float),
            test_df[model_cov].values,
            device=device,
            seed=args.seed + 9,
            **mlp_best,
        )
    c_mlp_raw, c_mlp = cindex_raw_and_adjusted(test_df["time"].values, test_df["endpoint"].values, mlp_prob)

    if y_bin_unique.size < 2:
        ensemble_w = 0.5
        oof_tree = np.full(len(train_df), np.nan, dtype=float)
        oof_mlp = np.full(len(train_df), np.nan, dtype=float)
        oof_mask = np.zeros(len(train_df), dtype=bool)
        oof_ens = np.full(len(train_df), np.nan, dtype=float)
        thr_primary_oob = {"threshold": 0.5, "rule": "fallback"}
        threshold_named: List[Tuple[str, Dict[str, float]]] = []
    else:
        ensemble_w, oof_tree, oof_mlp = select_ensemble_weight(
            train_df=train_df,
            features=model_cov,
            y_bin=y_train_bin,
            seed=args.seed + 30,
            use_gpu=use_gpu,
            n_jobs=cpu_jobs,
            xgb_params=xgb_best,
            mlp_params=mlp_best,
        )
        oof_mask = np.isfinite(oof_tree) & np.isfinite(oof_mlp)
        oof_ens = ensemble_w * oof_mlp + (1.0 - ensemble_w) * oof_tree
        thr_primary_oob, threshold_named = scan_thresholds_for_clinical(y_train_bin[oof_mask], oof_ens[oof_mask])
    ens_prob = ensemble_w * mlp_prob + (1.0 - ensemble_w) * xgb_prob
    c_ens_raw, c_ens = cindex_raw_and_adjusted(test_df["time"].values, test_df["endpoint"].values, ens_prob)

    horizons = [365, 730, 1095]
    metric_rows = []
    preds = {
        "model1_base_cox": risk1,
        "model2_geom_cox": risk2,
        "model3_full_cox": risk3,
        "model4_tree_auto": xgb_prob,
        "model5_mlp_auto": mlp_prob,
        "model6_ensemble_auto": ens_prob,
    }
    c_raw_map = {
        "model1_base_cox": c1_raw,
        "model2_geom_cox": c2_raw,
        "model3_full_cox": c3_raw,
        "model4_tree_auto": c_xgb_raw,
        "model5_mlp_auto": c_mlp_raw,
        "model6_ensemble_auto": c_ens_raw,
    }
    c_adj_map = {
        "model1_base_cox": c1,
        "model2_geom_cox": c2,
        "model3_full_cox": c3,
        "model4_tree_auto": c_xgb,
        "model5_mlp_auto": c_mlp,
        "model6_ensemble_auto": c_ens,
    }
    for name, pred in preds.items():
        row = {"model": name, "cindex_raw": c_raw_map[name], "cindex_adjusted": c_adj_map[name]}
        for h in horizons:
            y_h = ((test_df["endpoint"] == 1) & (test_df["time"] <= h)).astype(int).values
            if y_h.sum() == 0 or y_h.sum() == len(y_h):
                row[f"auc_{h}d"] = np.nan
            else:
                row[f"auc_{h}d"] = float(roc_auc_score(y_h, pred))
        metric_rows.append(row)
    metrics_fixed = pd.DataFrame(metric_rows)
    metrics_fixed.to_csv(out / "metrics_fixed_test.csv", index=False, encoding="utf-8-sig")

    # endpoint 二分类：测试集上以概率区分 endpoint（阈值=0.5 的诊断统计 + Brier/ECE）
    def endpoint_summary(model_name: str, pr: np.ndarray) -> Dict:
        yt = y_test_bin
        row: Dict[str, object] = {"model": model_name}
        if len(np.unique(yt)) < 2:
            row.update({"auc": np.nan, "balanced_accuracy_050": np.nan, "f1_050": np.nan, "mcc_050": np.nan, "brier": np.nan, "ece": np.nan})
            return row
        row["auc"] = float(roc_auc_score(yt, pr))
        m05 = clinical_binary_at_threshold(yt, pr, 0.5)
        row["balanced_accuracy_050"] = m05["balanced_accuracy"]
        row["f1_050"] = m05["f1"]
        row["mcc_050"] = m05["mcc"]
        row["brier"] = float(brier_score_loss(yt, pr))
        row["ece"] = float(expected_calibration_error(yt, pr))
        slope, intr = calibration_logit_slope_intercept(yt, pr)
        row["calibration_slope_logit"] = slope
        row["calibration_intercept_logit"] = intr
        return row

    ep_rows = [
        endpoint_summary("model4_tree_endpoint", xgb_prob),
        endpoint_summary("model5_mlp_endpoint", mlp_prob),
        endpoint_summary("model6_ensemble_endpoint", ens_prob),
    ]
    pd.DataFrame(ep_rows).to_csv(out / "metrics_endpoint_binary_test.csv", index=False, encoding="utf-8-sig")

    pred_bundle = pd.DataFrame(
        {
            "sample_id": test_df["sample_id"].astype(str).values,
            "patient_id": test_df["patient_id"].astype(str).values,
            "endpoint": y_test_bin,
            "time": test_df["time"].values,
            "prob_tree_endpoint": xgb_prob,
            "prob_mlp_endpoint": mlp_prob,
            "prob_ensemble_endpoint": ens_prob,
            "risk_cox_base": risk1,
            "risk_cox_geom": risk2,
            "risk_cox_full": risk3,
        }
    )
    pred_bundle.to_csv(out / "outer_test_predictions_bundle.csv", index=False, encoding="utf-8-sig")

    thr_fixed_str = thr_primary_oob.get("threshold")
    thr_fixed = float(thr_fixed_str) if thr_fixed_str is not None and np.isfinite(thr_fixed_str) else 0.5

    thr_test_rows = []
    for nm, d_oob in threshold_named:
        tt = clinical_binary_at_threshold(y_test_bin, ens_prob, float(d_oob["threshold"]))
        tt_row = dict(tt)
        tt_row["rule_name_train_oob"] = nm
        tt_row["auc_probability"] = float(roc_auc_score(y_test_bin, ens_prob)) if len(np.unique(y_test_bin)) > 1 else np.nan
        thr_test_rows.append(tt_row)
    if not thr_test_rows and len(np.unique(y_test_bin)) > 1:
        tt = clinical_binary_at_threshold(y_test_bin, ens_prob, thr_fixed)
        tt["rule_name_train_oob"] = "fallback"
        tt["auc_probability"] = float(roc_auc_score(y_test_bin, ens_prob))
        thr_test_rows.append(tt)
    pd.DataFrame(thr_test_rows).to_csv(out / "clinical_thresholds_test_performance.csv", index=False, encoding="utf-8-sig")

    # 训练折 OOF 上选出的阈值摘要（与同名列便于对照）
    thr_oob_flat = [{**dict(d), "rule_name": nm} for nm, d in threshold_named]
    if thr_oob_flat:
        pd.DataFrame(thr_oob_flat).to_csv(out / "clinical_thresholds_oob_train.csv", index=False, encoding="utf-8-sig")

    # 校准、DCA 与 ROC（固定外测；出版风格复合图）
    if len(np.unique(y_test_bin)) > 1:
        plot_calibration_composite(
            y_test_bin,
            ens_prob,
            fig_dir / "calibration_curve_endpoint_ensemble.png",
            title="Calibration diagnostic — ensemble fusion on binary endpoint",
        )
        plot_decision_curve(y_test_bin, ens_prob, fig_dir / "decision_curve_endpoint_ensemble.png")

        roc_curves_primary = (
            ("model4_tree_endpoint", xgb_prob),
            ("model5_mlp_endpoint", mlp_prob),
            ("model6_ensemble_endpoint", ens_prob),
            ("model3_full_cox_partial_hazard", risk3),
        )
        plot_roc_endpoint_multimodel(
            y_test_bin,
            roc_curves_primary,
            fig_dir / "roc_endpoint_multimodel_test.png",
        )

    # OOF isotonic calibration on fused probability (extra deliverables only)
    if args.oof_isotonic_calibration and y_bin_unique.size >= 2 and int(oof_mask.sum()) >= 20 and len(np.unique(y_test_bin)) > 1:
        iso_ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        xv = np.asarray(oof_ens[oof_mask], dtype=float)
        yv = np.asarray(y_train_bin[oof_mask], dtype=float)
        iso_ir.fit(xv, yv)
        ens_prob_iso = iso_ir.predict(np.asarray(ens_prob, dtype=float))
        oof_iso_pred = iso_ir.predict(xv)
        _, named_iso = scan_thresholds_for_clinical(yv, oof_iso_pred)
        ep_iso_row = endpoint_summary("model6_ensemble_isotonic_oob", ens_prob_iso)
        pd.DataFrame([ep_iso_row]).to_csv(out / "metrics_endpoint_isotonic_test.csv", index=False, encoding="utf-8-sig")
        thr_iso_rows = []
        for nm, d_oob in named_iso:
            tt = clinical_binary_at_threshold(y_test_bin, ens_prob_iso, float(d_oob["threshold"]))
            tt_row = dict(tt)
            tt_row["rule_name_train_oob"] = nm
            tt_row["auc_probability"] = float(roc_auc_score(y_test_bin, ens_prob_iso))
            thr_iso_rows.append(tt_row)
        pd.DataFrame(thr_iso_rows).to_csv(out / "clinical_thresholds_test_performance_isotonic.csv", index=False, encoding="utf-8-sig")

        plot_calibration_composite(
            y_test_bin,
            ens_prob_iso,
            fig_dir / "calibration_curve_endpoint_ensemble_isotonic.png",
            title="Calibration — OOF isotonic adjusted ensemble probabilities",
            n_bins=10,
        )
        plot_decision_curve(
            y_test_bin,
            ens_prob_iso,
            fig_dir / "decision_curve_endpoint_ensemble_isotonic.png",
            title="DCA (endpoint, probabilities after OOF isotonic calibration)",
        )

    # NRI/IDI vs Cox baseline risk（二元 endpoint 口径）
    y_endpoint_test = y_test_bin
    nri_rows = []
    for name in ["model2_geom_cox", "model3_full_cox", "model4_tree_auto", "model5_mlp_auto", "model6_ensemble_auto"]:
        nri_rows.append(
            {
                "model": name,
                "nri_vs_base_endpoint": nri(preds["model1_base_cox"], preds[name], y_endpoint_test),
                "idi_vs_base_endpoint": idi(preds["model1_base_cox"], preds[name], y_endpoint_test),
            }
        )
    pd.DataFrame(nri_rows).to_csv(out / "nri_idi_fixed_test.csv", index=False, encoding="utf-8-sig")

    # bootstrap CI on fixed test
    rng = np.random.RandomState(args.seed)
    b_rows = []
    n = len(test_df)
    for _ in range(args.bootstrap):
        idx = rng.randint(0, n, size=n)
        t = test_df["time"].values[idx]
        e = test_df["endpoint"].values[idx]
        yb_ep = y_test_bin[idx]
        rec = {}
        for name, pred in preds.items():
            p = pred[idx]
            c_raw, c_adj = cindex_raw_and_adjusted(t, e, p)
            rec[f"{name}_craw"] = c_raw
            rec[f"{name}_cadj"] = c_adj
            for h in horizons:
                yh = ((e == 1) & (t <= h)).astype(int)
                if yh.sum() == 0 or yh.sum() == len(yh):
                    rec[f"{name}_auc_{h}"] = np.nan
                else:
                    rec[f"{name}_auc_{h}"] = float(roc_auc_score(yh, p))
            if len(np.unique(yb_ep)) > 1:
                rec[f"{name}_auc_endpoint"] = float(roc_auc_score(yb_ep, p))
        for name in ["model2_geom_cox", "model3_full_cox", "model4_tree_auto", "model5_mlp_auto", "model6_ensemble_auto"]:
            rec[f"{name}_nri_ep"] = nri(preds["model1_base_cox"][idx], preds[name][idx], yb_ep)
            rec[f"{name}_idi_ep"] = idi(preds["model1_base_cox"][idx], preds[name][idx], yb_ep)
        if len(np.unique(yb_ep)) > 1:
            pe = preds["model6_ensemble_auto"][idx]
            ez = clinical_binary_at_threshold(yb_ep, pe, thr_fixed)
            rec["model6_ensemble_balanced_accuracy_thr_oob"] = ez["balanced_accuracy"]
            rec["model6_ensemble_f1_thr_oob"] = ez["f1"]
            rec["model6_ensemble_mcc_thr_oob"] = ez["mcc"]
            rec["model6_ensemble_sensitivity_thr_oob"] = ez["sensitivity"]
            rec["model6_ensemble_specificity_thr_oob"] = ez["specificity"]
        b_rows.append(rec)
    bdf = pd.DataFrame(b_rows)
    bdf.to_csv(out / "bootstrap_metrics.csv", index=False, encoding="utf-8-sig")

    # summarize CI
    sum_rows = []
    for col in bdf.columns:
        m, l, h = bootstrap_ci(bdf[col].tolist())
        sum_rows.append({"metric": col, "mean": m, "ci2.5": l, "ci97.5": h})
    summary_ci = pd.DataFrame(sum_rows)
    summary_ci.to_csv(out / "metrics_ci_summary.csv", index=False, encoding="utf-8-sig")

    # multi-seed replay on fixed split for robustness
    seed_rows = []
    for s in eval_seeds:
        tree_p, tree_back = fit_tree_predict(
            train_df=train_df,
            pred_df=test_df,
            features=model_cov,
            y_train=y_train_bin,
            seed=s + 1,
            use_gpu=use_gpu,
            n_jobs=cpu_jobs,
            xgb_params=xgb_best if "constant_prob" not in xgb_best else {},
        )
        if len(np.unique(y_train_bin)) < 2:
            mlp_p = np.full(len(test_df), float(y_train_bin[0]), dtype=float)
        else:
            mlp_p = train_torch_binary(
                train_df[model_cov].values,
                y_train_bin.astype(float),
                test_df[model_cov].values,
                device=device,
                seed=s + 2,
                **mlp_best,
            )
        ens_p = ensemble_w * mlp_p + (1.0 - ensemble_w) * tree_p
        c_tree_raw, c_tree = cindex_raw_and_adjusted(test_df["time"].values, test_df["endpoint"].values, tree_p)
        c_mlp_r, c_mlp_a = cindex_raw_and_adjusted(test_df["time"].values, test_df["endpoint"].values, mlp_p)
        c_ens_r, c_ens_a = cindex_raw_and_adjusted(test_df["time"].values, test_df["endpoint"].values, ens_p)
        y_h = ((test_df["endpoint"] == 1) & (test_df["time"] <= 1095)).astype(int).values
        auc_tree_td = float(roc_auc_score(y_h, tree_p)) if 0 < y_h.sum() < len(y_h) else np.nan
        auc_mlp_td = float(roc_auc_score(y_h, mlp_p)) if 0 < y_h.sum() < len(y_h) else np.nan
        auc_ens_td = float(roc_auc_score(y_h, ens_p)) if 0 < y_h.sum() < len(y_h) else np.nan
        auc_tree_ep = float(roc_auc_score(y_test_bin, tree_p)) if len(np.unique(y_test_bin)) > 1 else np.nan
        auc_mlp_ep = float(roc_auc_score(y_test_bin, mlp_p)) if len(np.unique(y_test_bin)) > 1 else np.nan
        auc_ens_ep = float(roc_auc_score(y_test_bin, ens_p)) if len(np.unique(y_test_bin)) > 1 else np.nan
        mthr = clinical_binary_at_threshold(y_test_bin, ens_p, thr_fixed) if len(np.unique(y_test_bin)) > 1 else {}
        seed_rows.append(
            {
                "seed": s,
                "tree_backend": tree_back,
                "threshold_used_from_main_oob_balanced_acc": thr_fixed,
                "model4_tree_auc_1095": auc_tree_td,
                "model5_mlp_auc_1095": auc_mlp_td,
                "model6_ens_auc_1095": auc_ens_td,
                "model4_tree_auc_endpoint": auc_tree_ep,
                "model5_mlp_auc_endpoint": auc_mlp_ep,
                "model6_ensemble_auc_endpoint": auc_ens_ep,
                "model6_ba_thr_oob": mthr.get("balanced_accuracy", np.nan),
                "model6_f1_thr_oob": mthr.get("f1", np.nan),
                "model6_sens_thr_oob": mthr.get("sensitivity", np.nan),
                "model6_spec_thr_oob": mthr.get("specificity", np.nan),
                "model6_mcc_thr_oob": mthr.get("mcc", np.nan),
                "model4_tree_cadj": c_tree,
                "model5_mlp_cadj": c_mlp_a,
                "model6_ens_cadj": c_ens_a,
                "model4_tree_craw": c_tree_raw,
                "model5_mlp_craw": c_mlp_r,
                "model6_ens_craw": c_ens_r,
            }
        )
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(out / "seed_repeat_metrics.csv", index=False, encoding="utf-8-sig")
    seed_summary_rows = []
    for c in [x for x in seed_df.columns if x != "seed" and x != "tree_backend"]:
        m, l, h = bootstrap_ci(seed_df[c].tolist())
        seed_summary_rows.append({"metric": c, "mean": m, "ci2.5": l, "ci97.5": h})
    pd.DataFrame(seed_summary_rows).to_csv(out / "seed_repeat_summary.csv", index=False, encoding="utf-8-sig")

    # segment stability by bootstrapping cox full on train
    seg_counter = {f"seg{i:02d}_amp": 0 for i in range(1, 18)}
    seg_rows = []
    for _ in range(min(200, args.bootstrap)):
        idx = rng.randint(0, len(train_df), size=len(train_df))
        tr = train_df.iloc[idx].copy()
        try:
            m, u = fit_cox(tr, full_cov, p3, l1_3)
            co = m.params_.to_dict()
            ranked = sorted([(k, abs(v)) for k, v in co.items() if k.startswith("seg")], key=lambda x: x[1], reverse=True)[:5]
            for k, v in ranked:
                seg_counter[k] += 1
                seg_rows.append({"segment": k, "abs_coef": float(v)})
        except Exception:
            pass
    seg_df = pd.DataFrame(
        [{"segment": k, "top5_freq": v, "top5_rate": v / max(min(200, args.bootstrap), 1)} for k, v in seg_counter.items()]
    ).sort_values("top5_freq", ascending=False)
    seg_df.to_csv(out / "segment_stability.csv", index=False, encoding="utf-8-sig")

    # bullseye standard: event-nonevent difference on selected full set
    diff_vals = {}
    ev = data[data["endpoint"] == 1]
    nv = data[data["endpoint"] == 0]
    for i in range(1, 18):
        c = f"seg{i:02d}_amp"
        diff_vals[i] = float(ev[c].mean() - nv[c].mean())
    plot_bullseye_academic(
        diff_vals,
        fig_dir / "bullseye_standard_diff.png",
        "Spatial pattern — mean Hilbert-derived segment amplitude difference (endpoint events vs non-events)",
    )

    # bullseye annotated by stability
    stab_vals = {}
    for i in range(1, 18):
        c = f"seg{i:02d}_amp"
        val = float(seg_df.loc[seg_df["segment"] == c, "top5_rate"].values[0])
        stab_vals[i] = val
    plot_bullseye_academic(
        stab_vals,
        fig_dir / "bullseye_top5_stability.png",
        "Selection frequency — Cox full-model |coef| within top-5 segments (training bootstrap)",
    )

    preds_td_order = [
        ("model1_base_cox", preds["model1_base_cox"]),
        ("model2_geom_cox", preds["model2_geom_cox"]),
        ("model3_full_cox", preds["model3_full_cox"]),
        ("model4_tree_auto", preds["model4_tree_auto"]),
        ("model5_mlp_auto", preds["model5_mlp_auto"]),
        ("model6_ensemble_auto", preds["model6_ensemble_auto"]),
    ]
    h_td = [365, 730, 1095]
    plot_time_dependent_aucs(
        horizons_days=h_td,
        preds_by_model=preds_td_order,
        test_endpoint=test_df["endpoint"].values,
        test_time=test_df["time"].values,
        out_path=fig_dir / "time_dependent_auc_curve.png",
    )

    # feature importance stability (xgb + cox)
    if xgb is not None and "constant_prob" not in xgb_best:
        imp_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            n_jobs=cpu_jobs,
            random_state=args.seed + 999,
            **xgb_best,
        )
        imp_model.fit(train_df[model_cov], y_train_bin)
        imp = pd.DataFrame({"feature": model_cov, "importance": imp_model.feature_importances_}).sort_values("importance", ascending=False)
        imp.to_csv(out / "xgb_feature_importance.csv", index=False, encoding="utf-8-sig")
    cox_coef = pd.DataFrame({"feature": list(m3.params_.index), "coef": list(m3.params_.values)})
    cox_coef.to_csv(out / "cox_full_coef.csv", index=False, encoding="utf-8-sig")

    protocol["compute_selected"] = compute_mode
    protocol["tree_backend"] = tree_backend
    protocol["tree_device"] = "cuda" if use_gpu and xgb is not None else "cpu"
    protocol["mlp_device"] = device
    protocol["ensemble_weight_mlp"] = ensemble_w
    protocol["inner_cv_tune_composite"] = "0.4*AUC + 0.3*balanced_accuracy + 0.3*F1 (inner decision threshold 0.5)"
    protocol["oob_ensemble_primary_threshold_balanced_accuracy"] = float(thr_fixed)
    protocol["oob_threshold_candidate_rules"] = [nm for nm, _ in threshold_named]
    ens_ep_metrics = {k: v for k, v in ep_rows[2].items() if k != "model"}
    protocol["endpoint_ensemble_probability_metrics_fixed_test"] = ens_ep_metrics
    protocol["boost_features"] = [c for c in model_cov if c not in full_cov]
    protocol["stable_top5_segments_train"] = stable_top5
    protocol["tree_best_params"] = xgb_best
    protocol["mlp_best_params"] = mlp_best
    protocol["oof_isotonic_calibration_enabled"] = bool(args.oof_isotonic_calibration)
    with open(out / "protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    # report
    top_seg_text = "\n".join(
        [f"- {r.segment}: {int(r.top5_freq)} 次 / {min(200, args.bootstrap)} 次（{r.top5_rate:.2%}）" for r in seg_df.head(5).itertuples(index=False)]
    )
    metrics_text = metrics_fixed.to_string(index=False)
    ci_text = summary_ci.sort_values("metric").head(30).to_string(index=False)
    seed_summary_text = pd.read_csv(out / "seed_repeat_summary.csv").sort_values("metric").to_string(index=False)
    ep_tab = pd.read_csv(out / "metrics_endpoint_binary_test.csv").to_string(index=False)
    thr_tab = pd.read_csv(out / "clinical_thresholds_test_performance.csv").to_string(index=False)
    iso_append = ""
    iso_path = out / "metrics_endpoint_isotonic_test.csv"
    if iso_path.exists():
        iso_append = (
            "\n## OOF Isotonic 校准（附加，不改变主表文件）\n```\n"
            + pd.read_csv(iso_path).to_string(index=False)
            + "\n```\n"
        )
    cob_path = out / "clinical_thresholds_oob_train.csv"
    cob_txt = pd.read_csv(cob_path).to_string(index=False) if cob_path.exists() else ""

    oof_thr_section = ""
    if cob_path.exists():
        oof_thr_section = (
            "### 训练侧 OOF 上各规则对应的阈值与指标摘要\n```\n"
            + cob_txt
            + "\n```\n\n"
        )

    baseline_note = (
        "对照提示：若以此前 **`outputs_deep_full_gpu_sprint3`** 中用「≤3年时间窗二元代理标签」训练的树/MLP 为基线，"
        "本轮已将监督标签统一为 **`endpoint` 本体二分类**（仍保留 Cox 与时间依赖 AUC 作为辅助解读）。在同一外层划分与可比 seed 设置下，"
        "应重点对比 **`metrics_endpoint_binary_test.csv`**（AUC/BalancedAcc/F1 @0.5、Brier/ECE）与 **`clinical_thresholds_test_performance.csv`**。"
    )

    clinical_md = (
        "# 临床可读汇报 — endpoint 二分类\n\n"
        f"{baseline_note}\n\n"
        "## 阈值推荐（训练折 OOF 上定义；外测复述）\n"
        "- 主方案：`max_balanced_accuracy`。\n"
        "- 备选：`max_youden_j`、`conservative_high_specificity`、`aggressive_high_sensitivity`。\n\n"
        f"{oof_thr_section}"
        "### 固定外测（复述各阈值对应的 Sens/Spec/PPV/NPV/LR+/LR-/F1/MCC）\n```\n{thr_tab}\n```\n\n"
        "## 概率层面的 endpoint 判别（阈值=0.5 及校准）\n"
        f"```\n{ep_tab}\n```\n\n"
        "## 图示\n"
        "- `figures/roc_endpoint_multimodel_test.png`\n"
        "- `figures/calibration_curve_endpoint_ensemble.png`\n"
        "- `figures/decision_curve_endpoint_ensemble.png`\n"
        "（DCA、校准与 ROC 均基于同一固定外测的概率 / 评分。逐例输出见 `outer_test_predictions_bundle.csv`。）\n"
        f"{iso_append}"
    )
    with open(out / "report_clinical_endpoint.md", "w", encoding="utf-8") as f:
        f.write(clinical_md)

    with open(out / "report_for_teacher_full_gpu.md", "w", encoding="utf-8") as f:
        f.write(
            "# 全量GPU深化实验终版报告\n\n"
            "## 数据与协议\n"
            "- 时间单位：天；endpoint作为事件标签（1=事件发生，0=未发生）。\n"
            "- 评估时间窗：1y=365天，2y=730天，3y=1095天。\n"
            f"- 选入样本：{len(data)}；总事件：{int(data['endpoint'].sum())}；3年事件：{int(((data['endpoint']==1)&(data['time']<=1095)).sum())}。\n"
            "- **树/MLP/融合**的监督标签为 `endpoint` 二分类（非 3 年窗代理）；内层 CV 优化目标：`0.4*AUC+0.3*BalancedAcc+0.3*F1`（概率阈值 0.5）。\n"
            "- 评估口径：固定外层测试集 + 内层嵌套调参；同时报告原始 C-index、方向校正 C-index，以及 Cox 与时间依赖 AUC。\n"
            f"- OOF 上选定的融合阈值（Balanced Accuracy 主规则）：**{thr_fixed:.6g}**（详见 `clinical_thresholds_*.csv`）。\n"
            f"- 计算策略：requested={args.compute}, selected={compute_mode}, cpu_jobs={cpu_jobs}。\n"
            f"- GPU状态：torch.cuda={torch.cuda.is_available()}，xgboost可用={xgb is not None}。\n\n"
            f"- 树模型后端：{tree_backend}（device={protocol['tree_device']}）。\n"
            f"- MLP设备：{device}。\n\n"
            "## Cox / 生存视角（固定外层测试）\n"
            f"```\n{metrics_text}\n```\n\n"
            "## endpoint 二分类（固定外层测试 — 概率与阈值）\n"
            "### 阈值（OOF→外测复述）与临床指标\n"
            f"```\n{thr_tab}\n```\n\n"
            "### 树 / MLP / 融合的概率型指标（含 Brier/ECE/logit校准）\n"
            f"```\n{ep_tab}\n```\n\n"
            "## 关键区间结果（bootstrap）\n"
            "含 `*_auc_endpoint`、按 OOF 主阈值裁切的 `model6_ensemble_*_thr_oob`，以及 Cox/NRI。\n\n"
            f"```\n{ci_text}\n```\n\n"
            "## 多seed复跑稳定性（固定外测）\n"
            "含 endpoint AUC（`auc_endpoint`）与主阈值 Sens/Spec/BA/F1（`model6_*_thr_oob`）。\n\n"
            f"```\n{seed_summary_text}\n```\n\n"
            "## 增益指标\n"
            "- NRI/IDI（二元 **endpoint** 标签）已写入 `nri_idi_fixed_test.csv`。\n\n"
            "## 解释性与定位\n"
            f"{top_seg_text}\n\n"
            "## 图表与交付清单\n"
            "- `figures/time_dependent_auc_curve.png`\n"
            "- `figures/roc_endpoint_multimodel_test.png`\n"
            "- `figures/calibration_curve_endpoint_ensemble.png` / `figures/decision_curve_endpoint_ensemble.png`\n"
            "- `figures/bullseye_standard_diff.png`\n"
            "- `figures/bullseye_top5_stability.png`\n"
            "- `outer_test_predictions_bundle.csv`（外测逐例标签与各模型分数）\n"
            "- `metrics_endpoint_binary_test.csv` · `clinical_thresholds_test_performance.csv`\n"
            "- `protocol_endpoint.json` · `report_clinical_endpoint.md`\n"
            "- `seed_repeat_metrics.csv` / `bootstrap_metrics.csv` 等同 seed 可比文件。\n"
        )

    print("完成: outputs ->", out)


if __name__ == "__main__":
    main()
