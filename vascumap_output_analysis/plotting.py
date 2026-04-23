"""Shared plotting and analysis functions for VascuMap experiment comparison notebooks."""

import colorsys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex
from matplotlib.gridspec import GridSpec
from PIL import ImageColor
from scipy.stats import gaussian_kde, kruskal, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# ── Colour utilities ──────────────────────────────────────────────────────────

def create_n_valued_palette(base_colour, n=5):
    """Create n shades of a base colour (name or hex) for per-image lines."""
    r, g, b = ImageColor.getcolor(base_colour, "RGB")
    r, g, b = r / 255, g / 255, b / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    adjustments = np.linspace(0.5, 1.4, n)
    palette = []
    for a in adjustments:
        l_adj = min(max(l * a, 0), 1)
        rgb = colorsys.hls_to_rgb(h, l_adj, s)
        palette.append(to_hex(rgb))
    return palette


# ── Data loading ──────────────────────────────────────────────────────────────

def combine_outputs(root_dir):
    """Recursively find analysis/branch CSVs under root_dir and concatenate them.

    Returns (combined_analysis_metrics, combined_branch_metrics).
    Also saves combined CSVs back into root_dir.
    """
    root_dir = Path(root_dir)
    csv_files = list(root_dir.rglob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return None, None

    combined_analysis_metrics = []
    combined_branch_metrics = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = str(f)
            df["source_folder"] = str(f.parent)
            if "analysis" in str(f):
                combined_analysis_metrics.append(df)
            elif "branch" in str(f):
                combined_branch_metrics.append(df)
            else:
                print(f"Found extra uncategorised csv {f}")
        except Exception as e:
            print(f"Failed to read {f}: {e}")

    if combined_analysis_metrics:
        combined_analysis_metrics = pd.concat(combined_analysis_metrics, ignore_index=True, sort=False)
        combined_analysis_metrics.to_csv(root_dir / "combined_analysis_metrics.csv", index=False)
    else:
        print("No analysis metric files could be read successfully.")
        combined_analysis_metrics = None

    if combined_branch_metrics:
        combined_branch_metrics = pd.concat(combined_branch_metrics, ignore_index=True, sort=False)
        combined_branch_metrics.to_csv(root_dir / "combined_branch_metrics.csv", index=False)
    else:
        print("No branch metric files could be read successfully.")
        combined_branch_metrics = None

    print(f"Total files combined: {len(combined_analysis_metrics)} analysis files "
          f"and {len(combined_branch_metrics)} branch files")
    return combined_analysis_metrics, combined_branch_metrics


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_experiment_comparisons(df, y_metrics, xorder, plot_title, save_dir,
                                condition_colors, save_type="png"):
    """Strip + box plot of each metric, one panel per metric."""
    fig, ax = plt.subplots(ncols=len(y_metrics), figsize=(4 * len(y_metrics), 5))
    if len(y_metrics) == 1:
        ax = [ax]
    for i, metric in enumerate(y_metrics):
        sns.stripplot(data=df, x="experiment", y=metric, ax=ax[i], zorder=100,
                      hue="experiment", palette=condition_colors, order=xorder, legend=False)
        sns.boxplot(data=df, x="experiment", y=metric, ax=ax[i],
                    fill=False, color="#000000", order=xorder)
    plt.tight_layout()
    out_path = Path(save_dir) / f"{plot_title}.{save_type}"
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.show()


def plot_orientation_kde(branch_df, condition_colors, conditions=None, save_dir=None):
    """Two-panel KDE of branch orientation: left weighted by length, right by volume.

    Automatically detects conditions from condition_colors present in the data.
    """
    if conditions is None:
        conditions = [c for c in condition_colors if c in branch_df["experiment"].unique()]

    img_colors = {}
    for condition in conditions:
        images = sorted(branch_df[branch_df["experiment"] == condition]["image_name"].unique())
        shades = create_n_valued_palette(condition_colors[condition], max(len(images), 2))
        for i, img_name in enumerate(images):
            img_colors[img_name] = shades[i % len(shades)]

    weight_configs = [
        ("path_length_um", "Proportional vessel length (a.u.)", "Branch orientation weighted by length"),
        ("branch_volume_um3", "Proportional vessel volume (a.u.)", "Branch orientation weighted by volume"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

    for ax, (weight_col, ylabel, title) in zip(axes, weight_configs):
        sub = branch_df.dropna(subset=["orientation_to_device_axis_deg", weight_col]).copy()
        sub = sub[sub["experiment"].isin(conditions)]

        x_eval = np.linspace(sub["orientation_to_device_axis_deg"].min(),
                             sub["orientation_to_device_axis_deg"].max(), 500)

        condition_kdes = {c: [] for c in conditions}

        for condition in conditions:
            images = sorted(sub[sub["experiment"] == condition]["image_name"].unique())
            for img_name in images:
                img_sub = sub[sub["image_name"] == img_name]
                angles = img_sub["orientation_to_device_axis_deg"].values
                weights_raw = img_sub[weight_col].values
                if len(angles) < 2 or weights_raw.sum() == 0:
                    continue
                weights = weights_raw / weights_raw.sum()
                kde = gaussian_kde(angles, weights=weights, bw_method=0.3)
                y = kde(x_eval)
                y = y / y.sum()
                ax.plot(x_eval, y, color=img_colors.get(img_name, "gray"), alpha=0.35, linewidth=1)
                condition_kdes[condition].append(y)

        for condition in conditions:
            if condition_kdes[condition]:
                mean_y = np.mean(condition_kdes[condition], axis=0)
                n = len(condition_kdes[condition])
                ax.plot(x_eval, mean_y, color=condition_colors[condition], linewidth=2.5,
                        label=f"Mean {condition.capitalize()}  (n={n})")

        ax.set_xlabel("Orientation to device axis (°)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        sns.despine(ax=ax)

    if save_dir is not None:
        fig.savefig(Path(save_dir) / "orientation_kde.png", dpi=150)
    plt.show()


def pca_plots(combined_analysis_metrics, condition_colors, save_dir=None, save_type="png"):
    """PCA scatter, volcano/importance-significance plot, and top-features bar chart.

    Uses PCA loadings (weighted by explained variance) for feature importance,
    and Mann-Whitney U / Kruskal-Wallis for statistical significance.
    Works for both 2-class and multiclass setups automatically.
    Returns statistically significant features (p < 0.05), or top 5 if none.
    """
    X = combined_analysis_metrics.drop(
        ["experiment", "image_name", "source_file", "image_index",
         "chip_volume_um3", "convex_hull_volume_um3", "vessel_volume_um3", "source_folder"],
        axis=1, errors="ignore",
    )
    y = combined_analysis_metrics["experiment"]

    # Drop columns that are entirely NaN, then drop rows with any remaining NaN
    X = X.dropna(axis=1, how="all")
    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    conditions = [c for c in condition_colors if c in y.unique()]
    is_binary = len(conditions) == 2

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # --- Feature importance from PCA loadings (weighted by explained variance) ---
    loadings = pca.components_
    ev = pca.explained_variance_ratio_

    if is_binary:
        importance_raw = loadings[0] * ev[0] + loadings[1] * ev[1]
        mean_pc1 = {c: X_pca[y.values == c, 0].mean() for c in conditions}
        pos_class = max(mean_pc1, key=mean_pc1.get)
        neg_class = min(mean_pc1, key=mean_pc1.get)
        feat_imp = pd.Series(importance_raw, index=X.columns).sort_values(key=abs, ascending=False)
    else:
        weighted_loadings = loadings * ev[:, None]
        importance_raw = np.sqrt((weighted_loadings ** 2).sum(axis=0))
        feat_imp = pd.Series(importance_raw, index=X.columns).sort_values(ascending=False)
        group_means = pd.DataFrame({c: X.loc[y == c].mean() for c in conditions})
        overall_mean = X.mean()
        deviations = group_means.subtract(overall_mean, axis=0).abs()
        dominant_class = deviations.idxmax(axis=1)

    # --- Statistical significance ---
    p_values = []
    if is_binary:
        for col in X.columns:
            groups = [X.loc[y == c, col] for c in conditions]
            _, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            p_values.append(p)
    else:
        for col in X.columns:
            groups = [X.loc[y == c, col].values for c in conditions]
            _, p = kruskal(*groups)
            p_values.append(p)

    p_values = np.array(p_values)
    neg_log_p = -np.log10(np.clip(p_values, 1e-300, None))
    sig_threshold = -np.log10(0.05)
    importance = feat_imp.reindex(X.columns).values

    sig_mask_all = p_values < 0.05
    sig_features = feat_imp[sig_mask_all[feat_imp.index.map(lambda f: list(X.columns).index(f))]]

    # --- Layout ---
    fig = plt.figure(figsize=(20, 6), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.4])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # --- 1. PCA scatter ---
    for name in conditions:
        mask = y == name
        ax0.scatter(X_pca[mask.values, 0], X_pca[mask.values, 1],
                    c=condition_colors[name], label=name.capitalize(),
                    s=80, edgecolors="k", linewidths=0.4, alpha=0.8)
    ax0.set_xlabel(f"PC1 ({ev[0]:.1%} var.)")
    ax0.set_ylabel(f"PC2 ({ev[1]:.1%} var.)")
    ax0.set_title("PCA – first two components")
    ax0.legend()

    # --- 2. Importance vs significance ---
    if is_binary:
        sig_pos = (neg_log_p >= sig_threshold) & (importance > 0)
        sig_neg = (neg_log_p >= sig_threshold) & (importance < 0)
        not_sig = neg_log_p < sig_threshold
        ax1.scatter(importance[not_sig], neg_log_p[not_sig], c="grey", alpha=0.5, s=40, label="Not significant")
        ax1.scatter(importance[sig_pos], neg_log_p[sig_pos], c=condition_colors[pos_class],
                    alpha=0.8, s=60, label=f"Significant → {pos_class}")
        ax1.scatter(importance[sig_neg], neg_log_p[sig_neg], c=condition_colors[neg_class],
                    alpha=0.8, s=60, label=f"Significant → {neg_class}")
        ax1.axvline(0, color="grey", linewidth=0.5)
        ax1.set_xlabel(f"PCA loading importance (+ → {pos_class}, − → {neg_class})")
        ax1.set_title("Volcano plot – PCA loading vs significance")
    else:
        sig_mask = neg_log_p >= sig_threshold
        ax1.scatter(importance[~sig_mask], neg_log_p[~sig_mask], c="grey", alpha=0.5, s=40, label="Not significant")
        for name in conditions:
            cls_mask = sig_mask & (dominant_class.reindex(X.columns) == name).values
            if cls_mask.any():
                ax1.scatter(importance[cls_mask], neg_log_p[cls_mask], c=condition_colors[name],
                            alpha=0.8, s=60, label=f"Sig. dominant → {name}")
        ax1.set_xlabel("PCA loading importance (variance-weighted magnitude)")
        ax1.set_title("Importance vs significance (Kruskal-Wallis)")

    ax1.axhline(sig_threshold, color="grey", linestyle="--", linewidth=0.8, label="p = 0.05")
    top_idx = np.argsort(neg_log_p * np.abs(importance))[-5:]
    for ti in top_idx:
        ax1.annotate(X.columns[ti], (importance[ti], neg_log_p[ti]),
                     fontsize=7, ha="center", va="bottom",
                     textcoords="offset points", xytext=(0, 5))
    ax1.set_ylabel("$-\\log_{10}$(p-value)")
    ax1.legend(fontsize=8)

    # --- 3. Top feature importances bar chart ---
    top = feat_imp.head(10)
    if is_binary:
        colors = [condition_colors[pos_class] if v > 0 else condition_colors[neg_class] for v in top]
        ax2.set_xlabel(f"Loading importance (+ → {pos_class}, − → {neg_class})")
    else:
        colors = [condition_colors.get(dominant_class[feat], "grey") for feat in top.index]
        ax2.set_xlabel("Loading importance (coloured by dominant class)")
    ax2.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax2.set_title("Top 10 discriminating features")
    ax2.axvline(0, color="grey", linewidth=0.8)
    ax2.tick_params(axis="y", labelsize=9)

    if save_dir is not None:
        fig.savefig(Path(save_dir) / f"pca_plots.{save_type}", dpi=150)
    plt.show()

    if len(sig_features) == 0:
        print("No significant features (p < 0.05). Returning top 5 most discriminating features instead.")
        return feat_imp.head(5).index.to_list()

    print(f"Significant features: {len(sig_features)} / {len(X.columns)}")
    return (sig_features.index.to_list(), feat_imp.head(5).index.to_list())
