"""
Publication-style figures shared by deep_experiment_full_gpu.py and offline regeneration.
Avoid importing deep_experiment from here (no cycles).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve


# Colorblind-safe palette (Wong / Nature-ish)
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def configure_academic_mpl(font_scale: float = 1.0) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Microsoft YaHei", "SimHei"],
            "axes.unicode_minus": False,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.facecolor": "#FAFAFA",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.55,
            "grid.linestyle": "-",
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#CCCCCC",
        }
    )
    fs = 10 * font_scale
    mpl.rcParams["axes.titlesize"] = fs + 2
    mpl.rcParams["axes.labelsize"] = fs + 1
    mpl.rcParams["legend.fontsize"] = fs - 1
    mpl.rcParams["xtick.labelsize"] = fs - 1
    mpl.rcParams["ytick.labelsize"] = fs - 1


def compute_dca_arrays(
    y_true: np.ndarray, probs: np.ndarray, n_points: int = 81
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.linspace(1e-3, 1.0 - 1e-3, n_points)
    n = len(y_true)
    p_disease = float(np.mean(y_true))
    nb_net = []
    for pt in pts:
        yhat = (probs >= pt).astype(int)
        tp = float(np.sum((yhat == 1) & (y_true == 1)))
        fp = float(np.sum((yhat == 1) & (y_true == 0)))
        nb_net.append(tp / max(n, 1) - fp / max(n, 1) * (pt / max(1.0 - pt, 1e-8)))
    nb_all = []
    nb_none = []
    for pt in pts:
        nb_all.append(p_disease - (1.0 - p_disease) * (pt / max(1.0 - pt, 1e-8)))
        nb_none.append(0.0)
    return pts, np.array(nb_net), np.array(nb_all), np.array(nb_none)


def _short_model_label(raw: str) -> str:
    m = raw.lower().replace("_", " ")
    if "model6" in m or "ensemble" in m:
        return "Ensemble"
    if "model5" in m or "mlp" in m:
        return "MLP"
    if "model4" in m or "tree" in m:
        return "Tree"
    if "model3" in m or "full" in m:
        return "Cox full features"
    if "model2" in m or "geom" in m:
        return "Cox geometry"
    if "model1" in m or "base" in m:
        return "Cox baseline"
    return raw


def plot_roc_endpoint_multimodel(
    y_endpoint: np.ndarray,
    curves: Sequence[Tuple[str, np.ndarray]],
    out_path: Path,
    *,
    subtitle: str = "",
) -> None:
    """FPR vs TPR; each curve tuple is (name, scores). Cox risks are valid ROC ranking."""
    fig, ax = plt.subplots(figsize=(6.2, 5.8), constrained_layout=True)
    aucs: List[str] = []
    for idx, (name, scores) in enumerate(curves):
        y = np.asarray(y_endpoint).astype(int)
        s = np.asarray(scores, dtype=float)
        if len(np.unique(y)) < 2:
            continue
        auc = roc_auc_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        lab = _short_model_label(name)
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(fpr, tpr, lw=2.4, label=f"{lab} (AUC = {auc:.3f})", color=color)
        aucs.append(f"{lab}:{auc:.3f}")
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=1.2, color="#888888", label="Reference")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curves for endpoint discrimination (fixed hold-out test)" + (f"\n{subtitle}" if subtitle else ""))
    ax.legend(loc="lower right")
    ax.set_facecolor("#FFFFFF")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_composite(
    y_true: np.ndarray,
    probs: np.ndarray,
    out_path: Path,
    *,
    title: str = "Calibration — ensemble probability vs observed endpoint fraction",
    n_bins: int = 10,
) -> None:
    """Main panel: calibration; side: histogram of predicted probabilities."""
    fig = plt.figure(figsize=(10.8, 4.95), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    yt = np.asarray(y_true).astype(int)
    pr = np.asarray(probs, dtype=float)
    if len(np.unique(yt)) >= 2:
        prob_true, mean_pred = calibration_curve(yt, pr, n_bins=n_bins, strategy="uniform")
        ax0.plot(mean_pred, prob_true, "o-", color=PALETTE[0], ms=7, lw=2, label="Model (uniform bins)")
    ax0.plot([0, 1], [0, 1], ls=(0, (4, 3)), color="#666666", lw=1.2, label="Perfect calibration")
    ax0.set_xlabel("Predicted probability")
    ax0.set_ylabel("Fraction of positives (actual)")
    ax0.set_title("Reliability curve")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.legend(loc="upper left")
    ax1.hist(pr, bins=min(38, max(14, len(pr) // 3)), density=True, color="#BFBFBF", edgecolor="#FFFFFF", alpha=0.88)
    ax1.set_xlabel("Predicted ensemble probability")
    ax1.set_ylabel("Density")
    ax1.set_title("Probability distribution on test set")
    fig.suptitle(title, fontsize=mpl.rcParams["axes.titlesize"] + 1, y=1.03)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_decision_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    out_path: Path,
    *,
    title: str = "Decision curve analysis (net benefit)",
) -> None:
    pts, nb_net, nb_all, nb_none = compute_dca_arrays(y_true, probs)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.plot(pts, nb_net, color=PALETTE[0], lw=2.3, label="Ensemble model")
    ax.plot(pts, nb_all, ls=(0, (4, 3)), lw=1.5, color=PALETTE[3], alpha=0.85, label="Treat all")
    ax.plot(pts, nb_none, ls=(0, (2, 2)), lw=1.5, color="#666666", label="Treat none")
    ax.set_xlim(pts.min(), pts.max())
    lo = np.nanmin([np.nanmin(nb_net), np.nanmin(nb_all)])
    hi = np.nanmax([np.nanmax(nb_net), np.nanmax(nb_all)])
    pad = max(0.03, (hi - lo) * 0.12)
    ax.set_ylim(lo - pad, hi + pad)
    ax.axhline(0.0, color="#BBBBBB", lw=0.7, zorder=0)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_facecolor("#FFFFFF")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_time_dependent_aucs(
    horizons_days: Iterable[int],
    preds_by_model: Sequence[Tuple[str, np.ndarray]],
    test_endpoint: np.ndarray,
    test_time: np.ndarray,
    out_path: Path,
) -> None:
    """Line plot across horizons — already common in papers."""
    xs = np.array(list(horizons_days))
    fig, ax = plt.subplots(figsize=(7.5, 5.35), constrained_layout=True)
    e = np.asarray(test_endpoint).astype(int)
    ti = np.asarray(test_time).astype(float)
    for mi, (name, pred) in enumerate(preds_by_model):
        vals = []
        for h in xs:
            y_h = ((e == 1) & (ti <= float(h))).astype(int)
            if y_h.sum() == 0 or y_h.sum() == len(y_h):
                vals.append(np.nan)
            else:
                vals.append(float(roc_auc_score(y_h, pred)))
        ax.plot(
            xs,
            vals,
            marker="o",
            lw=2.1,
            ms=6,
            label=_short_model_label(name),
            color=PALETTE[mi % len(PALETTE)],
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d // 365} y" if d % 365 == 0 else f"{int(d)} d" for d in xs])
    ax.set_xlabel("Time horizon")
    ax.set_ylabel("Time-dependent ROC-AUC")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Time-dependent ROC at fixed horizons (IPCW-free proxy)")
    ax.legend(loc="lower right", ncol=2, fontsize=mpl.rcParams["legend.fontsize"] - 0.5)
    ax.set_facecolor("#FFFFFF")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_bullseye_academic(values: Dict[int, float], output_png: Path, title: str) -> None:
    from matplotlib.colors import Normalize
    from matplotlib.patches import Circle, Wedge

    vv = np.array(list(values.values()), dtype=float)
    vmax = float(np.max(np.abs(vv))) if vv.size and np.max(np.abs(vv)) > 0 else 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    fig, ax = plt.subplots(figsize=(7.35, 6.95), constrained_layout=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.2, 1.35)
    ax.set_ylim(-1.15, 1.12)

    def col(seg):
        return cmap(norm(values.get(seg, 0.0)))

    for i in range(6):
        st = 90 - i * 60
        ed = st - 60
        ax.add_patch(Wedge((0, 0), 1.0, ed, st, width=0.25, facecolor=col(i + 1), edgecolor="white", linewidth=1.05))
    for i in range(6):
        st = 90 - i * 60
        ed = st - 60
        ax.add_patch(Wedge((0, 0), 0.75, ed, st, width=0.25, facecolor=col(i + 7), edgecolor="white", linewidth=1.05))
    for i in range(4):
        st = 90 - i * 90
        ed = st - 90
        ax.add_patch(Wedge((0, 0), 0.5, ed, st, width=0.25, facecolor=col(i + 13), edgecolor="white", linewidth=1.05))
    ax.add_patch(Circle((0, 0), 0.25, facecolor=col(17), edgecolor="white", linewidth=1.05))
    for s in range(1, 18):
        if s <= 6:
            r = 0.88
            ang = np.deg2rad(90 - (s - 0.5) * 60)
        elif s <= 12:
            r = 0.63
            ang = np.deg2rad(90 - ((s - 6) - 0.5) * 60)
        elif s <= 16:
            r = 0.38
            ang = np.deg2rad(90 - ((s - 12) - 0.5) * 90)
        else:
            r = 0.0
            ang = 0.0
        ax.text(r * np.cos(ang), r * np.sin(ang), str(s), ha="center", va="center", fontsize=9, fontweight="medium")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.047, pad=0.022)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label("Scale")
    ax.set_title(title)
    ax.text(1.12, -0.95, "AHA 17 segments", fontsize=9, va="bottom", ha="right", style="italic", color="#555555")

    plt.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
