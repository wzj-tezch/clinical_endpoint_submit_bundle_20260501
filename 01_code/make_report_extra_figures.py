"""
从既有实验输出目录读取 CSV，生成可用于学术报告的补充统计图（出版风格）。
用法:
  python make_report_extra_figures.py [--output-dir 冠军目录] [--figures-subdir figures_extra]
若有 outer_test_predictions_bundle.csv 等 CSV，可同时刷新 figures/ 下 ROC、校准、DCA、time-dependent AUC 与两张 bullseye（与主实验脚本图示一致）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from report_figure_utils import (
    PALETTE,
    configure_academic_mpl,
    plot_bullseye_academic,
    plot_calibration_composite,
    plot_decision_curve,
    plot_roc_endpoint_multimodel,
    plot_time_dependent_aucs,
)


def plot_model_comparison(endpoint_csv: Path, out: Path) -> None:
    df = pd.read_csv(endpoint_csv)

    def short_name(m: str) -> str:
        m = str(m)
        if "model4" in m:
            return "Tree"
        if "model5" in m:
            return "MLP"
        if "model6" in m:
            return "Ensemble"
        return m

    labels = [short_name(x) for x in df["model"]]
    metrics_names = ["AUC", "Balanced Acc.@0.5", "F1@0.5"]
    cols = ["auc", "balanced_accuracy_050", "f1_050"]
    mat = df[cols].astype(float).values

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.95), gridspec_kw={"width_ratios": [1.0, 1.15]}, constrained_layout=True)
    ax0 = axes[0]
    im = ax0.imshow(mat.T, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax0.set_xticks(np.arange(len(labels)))
    ax0.set_xticklabels(labels)
    ax0.set_yticks(np.arange(len(metrics_names)))
    ax0.set_yticklabels(metrics_names)
    ax0.set_title("Metrics heatmap (hold-out)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax0.text(i, j, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=11, fontweight="medium", color="#111111")

    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04).set_label("Score")
    ax1 = axes[1]
    x = np.arange(len(df))
    w = 0.24
    hatches = ["/", "\\", ".."]
    for i, cn in enumerate(cols):
        vals = df[cn].astype(float).values
        ax1.bar(
            x + (i - 1) * w,
            vals,
            width=w * 1.06,
            label=metrics_names[i],
            color=PALETTE[i % len(PALETTE)],
            alpha=0.92,
            edgecolor="#FFFFFF",
            linewidth=1.05,
            hatch=hatches[i % len(hatches)] * 2,
        )
    ax1.axhline(0.5, color="#888888", ls=(0, (3, 3)), lw=1.05, alpha=0.9, label=None)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Grouped comparison (dual encoding)")
    ax1.set_ylabel("Score")
    ax1.legend(loc="lower right", ncol=3)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_rules(thr_csv: Path, out: Path) -> None:
    df0 = pd.read_csv(thr_csv)
    short = []
    for r in df0["rule_name_train_oob"]:
        if r == "max_balanced_accuracy":
            short.append("Max-BA")
        elif r == "max_youden_j":
            short.append("Youden")
        elif r == "conservative_high_specificity":
            short.append("Hi-Spec")
        elif r == "aggressive_high_sensitivity":
            short.append("Hi-Sens")
        else:
            short.append(r[:12])
    df = df0.copy()
    df["rule_short"] = short
    df = df.drop_duplicates(subset=["rule_short"], keep="first")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.98), gridspec_kw={"width_ratios": [1.0, 1.18]}, constrained_layout=True)

    cols_plot = ["sensitivity", "specificity", "balanced_accuracy"]
    pretty = {"sensitivity": "Sensitivity", "specificity": "Specificity", "balanced_accuracy": "Balanced accuracy"}
    n = len(df)
    xs = np.arange(n)
    w = 0.26
    for i, c in enumerate(cols_plot):
        axes[0].bar(
            xs + (i - 1) * w,
            df[c].astype(float),
            width=w * 1.05,
            label=pretty[c],
            color=PALETTE[i],
            alpha=0.9,
            edgecolor="#FFFFFF",
        )
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(df["rule_short"], rotation=12, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Value")
    axes[0].set_title("Clinical operating points by threshold rule")
    axes[0].legend(loc="lower right", ncol=3)

    xsc = df["specificity"].astype(float).values
    ysc = df["sensitivity"].astype(float).values
    baz = df["balanced_accuracy"].astype(float).values
    labs = df["rule_short"].tolist()
    sizes = (baz * 920 + 140).tolist()
    axes[1].scatter(
        xsc,
        ysc,
        s=sizes,
        c=baz,
        cmap="viridis",
        vmin=0.55,
        vmax=0.75,
        edgecolors="#FFFFFF",
        linewidths=2.15,
        zorder=10,
        alpha=0.96,
    )
    for lx, ly, lb in zip(xsc, ysc, labs):
        axes[1].annotate(lb, (lx, ly), textcoords="offset points", xytext=(10, -6), fontsize=9, clip_on=True)

    xr = np.linspace(0.45, max(1.03, np.max(xsc) + 0.05), 50)
    axes[1].plot(xr, xr, ls=(0, (4, 3)), lw=1.05, color="#999999")
    axes[1].set_xlim(0.54, max(1.01, np.max(xsc) * 1.015))
    axes[1].set_ylim(0.61, max(1.01, np.max(ysc) * 1.02))
    axes[1].set_xlabel("Specificity")
    axes[1].set_ylabel("Sensitivity")
    axes[1].set_title("ROC-like trade-off sketch (Sens vs Spec, marker size/color = BA)")
    sm = axes[1].collections[0]
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.05, pad=0.025)
    cbar.ax.set_title("Bal. acc.", fontsize=9)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_forest_ci(summary_csv: Path, out: Path) -> None:
    sm = pd.read_csv(summary_csv)
    want_prefix = (
        "model6_ensemble_auto_auc_endpoint",
        "model6_ensemble_balanced_accuracy_thr_oob",
        "model6_ensemble_sensitivity_thr_oob",
        "model6_ensemble_specificity_thr_oob",
    )
    pretty = (
        "Endpoint ROC-AUC",
        "Balanced accuracy @ OOF-threshold",
        "Sensitivity",
        "Specificity",
    )
    rows = []
    for p in want_prefix:
        hit = sm[sm["metric"] == p]
        if not hit.empty:
            rows.append(hit.iloc[0])
    if not rows:
        return
    sub = pd.DataFrame(rows).iloc[::-1]
    k = len(sub)
    y = np.arange(k)
    mean = sub["mean"].astype(float).values
    lo = sub["ci2.5"].astype(float).values
    hi = sub["ci97.5"].astype(float).values

    fig, ax = plt.subplots(figsize=(8.0, max(4.0, k * 0.95)), constrained_layout=True)
    for yi, mn, lw, hh in zip(y, mean, lo, hi):
        ax.fill_betweenx([yi - 0.42, yi + 0.42], lw, hh, color=PALETTE[0], alpha=0.28)
    ax.scatter(mean, y, zorder=12, marker="s", color=PALETTE[1], s=225, linewidths=1.6, edgecolors="#FFFFFF")
    for yi, mn in zip(y, mean):
        ax.plot([mn, mn], [yi - 0.34, yi + 0.34], color="#333333", lw=5.05, alpha=0.88, clip_on=False, zorder=8)
    lbl = pretty[len(pretty) - k :] if len(pretty) >= k else [str(u)[:50] for u in sub["metric"]]
    ax.set_yticks(y)
    ax.set_yticklabels(lbl, fontsize=10)
    ax.set_xlim(0, max(1.06, np.nanmax(hi) * 1.06))
    ax.set_xlabel("Bootstrap summaries (percentile bootstrap on fixed test cohort)")
    ax.set_title("Interval summary — fused model operating characteristics")
    ax.set_facecolor("#FFFFFF")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap_kde_combo(bootstrap_csv: Path, col: str, out: Path) -> None:
    df = pd.read_csv(bootstrap_csv)
    if col not in df.columns:
        return
    s = pd.to_numeric(df[col], errors="coerce").dropna().values
    if s.size == 0:
        return

    fig = plt.figure(figsize=(11.15, 4.92), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.05, 0.92])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    sns_like_bins = max(26, min(48, len(s) // 9))
    n, eb, patches = ax0.hist(s, bins=sns_like_bins, density=True, color="#B8CFE6", alpha=0.92, edgecolor="#FFFFFF")

    lo, mi, hi = np.percentile(s, [2.5, 50, 97.5])
    mu = float(np.mean(s))
    ymin, ymax = ax0.get_ylim()
    shaded_h = ymin + 0.04 * (ymax - ymin + 1e-9)
    ax0.fill_between([lo, hi], ymin, ymin + shaded_h - ymin, alpha=0.35, color="#888888", label="_nolegend_", clip_on=False)
    ax0.axvline(mi, ls=(0, (4, 3)), lw=2.05, color=PALETTE[2], alpha=0.95, label=f"Median = {mi:.3f}")
    ax0.axvline(mu, ls=(0, (2, 2)), lw=1.85, color=PALETTE[1], alpha=0.95, label=f"Mean = {mu:.3f}")
    kde_x = np.linspace(np.nanmin(s), np.nanmax(s), min(600, len(s) * 5))
    try:
        kde_fn = scipy_stats.gaussian_kde(s)
        ax0.plot(kde_x, kde_fn(kde_x), color=PALETTE[0], lw=2.6, label="Gaussian KDE")
    except Exception:
        pass
    ax0.set_title("Bootstrap ROC-AUC (ensemble endpoint, test resampled)")
    ax0.set_xlabel(col.replace("_", " "))
    ax0.set_ylabel("Density")
    ax0.legend(loc="upper right")

    probs = scipy_stats.probplot(s, dist="norm", plot=None)
    osm, osr = probs[0]
    slope, intercept, r_val, *_ = scipy_stats.linregress(osm, osr)
    ax1.scatter(osm, osr, s=38, alpha=0.74, edgecolors="#FFFFFF", c=PALETTE[3])
    ax1.plot(osm, intercept + slope * osm, color=PALETTE[0], lw=2.2, ls=(0, (4, 2)))
    ax1.text(0.05, 0.92, f"Pearson r ≈ {r_val:.3f}", transform=ax1.transAxes, fontsize=9)
    ax1.set_title("Q–Q vs normal (exploratory)")
    ax1.set_xlabel("Theoretical quantiles")
    ax1.set_ylabel("Ordered values")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_segment_stability(seg_csv: Path, out: Path, top_k: int = 14) -> None:
    df = pd.read_csv(seg_csv).sort_values("top5_rate", ascending=False).head(top_k)
    df = df.iloc[::-1]
    rates = df["top5_rate"].astype(float).values
    labs = df["segment"].tolist()
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9.85, max(6.05, len(df) * 0.32)), constrained_layout=True)
    ax.barh(y, rates * 0.92, height=0.34, alpha=0.38, color=PALETTE[0], edgecolor="#FFFFFF")
    ax.plot(rates, y, marker="o", ms=12.5, linestyle="-", lw=3.05, markerfacecolor=PALETTE[2], markeredgecolor="#FFFFFF", markeredgewidth=1.3, color=PALETTE[1])
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.set_xlim(0, min(1.05, np.max(rates) * 1.15))
    ax.set_xlabel("Relative frequency entering Cox-full |coef| top-5 (training bootstrap)")
    ax.set_title("Segment-level explanatory stability")
    ax.set_facecolor("#FFFFFF")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_compare(ep_a: Path, ep_b: Path, label_a: str, label_b: str, out: Path) -> None:
    a = pd.read_csv(ep_a)
    b = pd.read_csv(ep_b)
    ae = a[a["model"] == "model6_ensemble_endpoint"].iloc[0]
    be = b[b["model"] == "model6_ensemble_endpoint"].iloc[0]
    metrics_keys = ["auc", "balanced_accuracy_050", "f1_050"]
    xlabels = ["AUC", "BA@0.5", "F1@0.5"]
    x = np.arange(len(metrics_keys))

    vals_a = [float(ae[m]) for m in metrics_keys]
    vals_b = [float(be[m]) for m in metrics_keys]
    deltas = np.array(vals_b) - np.array(vals_a)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.95), gridspec_kw={"height_ratios": [1.45, 0.92]}, constrained_layout=True)
    w = 0.36
    bars1 = axes[0].bar(x - w / 2, vals_a, width=w, label=label_a, color=PALETTE[0], hatch="//", alpha=0.92, edgecolor="#FFFFFF")
    bars2 = axes[0].bar(x + w / 2, vals_b, width=w, label=label_b, color=PALETTE[1], hatch="\\\\", alpha=0.92, edgecolor="#FFFFFF")

    axes[0].scatter(x - w / 2 + 0.02, vals_a, marker="P", color="#222222", s=115, zorder=5)
    axes[0].scatter(x + w / 2 - 0.02, vals_b, marker="v", color="#222222", s=115, zorder=5)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(xlabels)
    axes[0].set_ylim(0, max(1.05, np.max(vals_a + vals_b) * 1.05))
    axes[0].set_ylabel("Score")
    axes[0].set_title("Cohort-design sensitivity -- ensemble fusion on hold-out endpoint")
    axes[0].legend(loc="upper right")

    colors = ["#4477AA", "#998833", "#CC6677"]
    axes[1].barh(xlabels, deltas, height=0.55, color=colors, alpha=0.93, edgecolor="#FFFFFF")
    axes[1].set_xlim(min(-0.12, deltas.min() * 1.2), max(0.12, deltas.max() * 1.2))
    axes[1].axvline(0, color="#333333", lw=1)
    axes[1].set_xlabel("Δ score (strategy B − strategy A)")
    axes[1].set_title("Metric deltas between enrollment policies")

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def maybe_refresh_roc_from_bundle(output_dir: Path) -> None:
    bundle_path = output_dir / "outer_test_predictions_bundle.csv"
    if not bundle_path.exists():
        return
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    bd = pd.read_csv(bundle_path)
    ye = bd["endpoint"].astype(int).values
    roc_curves = (
        ("model4_tree_endpoint", bd["prob_tree_endpoint"].astype(float).values),
        ("model5_mlp_endpoint", bd["prob_mlp_endpoint"].astype(float).values),
        ("model6_ensemble_endpoint", bd["prob_ensemble_endpoint"].astype(float).values),
        ("model3_full_cox_partial_hazard", bd["risk_cox_full"].astype(float).values),
    )
    plot_roc_endpoint_multimodel(
        ye,
        roc_curves,
        fig_dir / "roc_endpoint_multimodel_test.png",
    )


def refresh_standard_figures_from_csv(output_dir: Path) -> None:
    """从 CSV 离线重绘 `figures/` 主文图（无外测 CSV 则无操作 / 跳过子项）。"""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    feats_csv = output_dir / "features_with_labels_full.csv"
    segstab = output_dir / "segment_stability.csv"
    if feats_csv.exists():
        df = pd.read_csv(feats_csv)
        ev = df[df["endpoint"] == 1]
        nv = df[df["endpoint"] == 0]
        diff_vals: dict[int, float] = {}
        for i in range(1, 18):
            c = f"seg{i:02d}_amp"
            diff_vals[i] = float(ev[c].mean() - nv[c].mean())
        plot_bullseye_academic(
            diff_vals,
            fig_dir / "bullseye_standard_diff.png",
            "Spatial pattern — mean Hilbert-derived segment amplitude difference (endpoint events vs non-events)",
        )
        stab_vals: dict[int, float] = {}
        if segstab.exists():
            sdf = pd.read_csv(segstab)
            rates = dict(zip(sdf["segment"].astype(str), sdf["top5_rate"].astype(float)))
            for i in range(1, 18):
                k = f"seg{i:02d}_amp"
                stab_vals[i] = float(rates.get(k, 0.0))
            plot_bullseye_academic(
                stab_vals,
                fig_dir / "bullseye_top5_stability.png",
                "Selection frequency — Cox full-model |coef| within top-5 segments (training bootstrap)",
            )

    bundle_path = output_dir / "outer_test_predictions_bundle.csv"
    if bundle_path.exists():
        bd = pd.read_csv(bundle_path)
        ye = bd["endpoint"].astype(int).values
        preds_td_order = (
            ("model1_base_cox", bd["risk_cox_base"].astype(float).values),
            ("model2_geom_cox", bd["risk_cox_geom"].astype(float).values),
            ("model3_full_cox", bd["risk_cox_full"].astype(float).values),
            ("model4_tree_auto", bd["prob_tree_endpoint"].astype(float).values),
            ("model5_mlp_auto", bd["prob_mlp_endpoint"].astype(float).values),
            ("model6_ensemble_auto", bd["prob_ensemble_endpoint"].astype(float).values),
        )
        plot_time_dependent_aucs(
            [365, 730, 1095],
            preds_td_order,
            ye,
            bd["time"].astype(float).values,
            fig_dir / "time_dependent_auc_curve.png",
        )
        plot_calibration_composite(
            ye,
            bd["prob_ensemble_endpoint"].astype(float).values,
            fig_dir / "calibration_curve_endpoint_ensemble.png",
            title="Calibration diagnostic — ensemble fusion on binary endpoint",
        )
        plot_decision_curve(ye, bd["prob_ensemble_endpoint"].astype(float).values, fig_dir / "decision_curve_endpoint_ensemble.png")

    maybe_refresh_roc_from_bundle(output_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"e:\本基中间结果\outputs_clinical_endpoint_n320_bs400"),
        help="主分析实验输出目录（含 CSV）",
    )
    ap.add_argument(
        "--compare-endpoint-csv",
        type=Path,
        default=Path(r"e:\本基中间结果\outputs_clinical_endpoint_full_bs400\metrics_endpoint_binary_test.csv"),
        help="用于与主目录对比的第二份 metrics_endpoint_binary_test.csv（可选）",
    )
    ap.add_argument("--figures-subdir", type=str, default="figures_extra", help="在 output-dir 下创建的子目录名")
    args = ap.parse_args()
    configure_academic_mpl()

    refresh_standard_figures_from_csv(args.output_dir)

    out_dir = args.output_dir / args.figures_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint_csv = args.output_dir / "metrics_endpoint_binary_test.csv"
    thr_csv = args.output_dir / "clinical_thresholds_test_performance.csv"
    summary_csv = args.output_dir / "metrics_ci_summary.csv"
    boot_csv = args.output_dir / "bootstrap_metrics.csv"
    seg_csv = args.output_dir / "segment_stability.csv"

    if endpoint_csv.exists():
        plot_model_comparison(endpoint_csv, out_dir / "fig_bar_models_endpoint_metrics.png")
    if thr_csv.exists():
        plot_threshold_rules(thr_csv, out_dir / "fig_bar_threshold_rules.png")
    if summary_csv.exists():
        plot_forest_ci(summary_csv, out_dir / "fig_forest_bootstrap_ci_model6.png")
    if boot_csv.exists():
        plot_bootstrap_kde_combo(
            boot_csv, "model6_ensemble_auto_auc_endpoint", out_dir / "fig_hist_bootstrap_ensemble_auc_endpoint.png"
        )
    if seg_csv.exists():
        plot_segment_stability(seg_csv, out_dir / "fig_bar_segment_stability_top.png")

    if args.compare_endpoint_csv.exists() and endpoint_csv.exists():
        plot_cohort_compare(
            endpoint_csv,
            args.compare_endpoint_csv,
            "Min-samples 320 (event-enriched cohort)",
            "Full mergeable cohort (natural prevalence)",
            out_dir / "fig_compare_two_cohorts_ensemble.png",
        )

    print("Figures refreshed:", args.output_dir / "figures,", out_dir / "(extras)")


if __name__ == "__main__":
    main()
