"""
src/validation.py
=================
CAT411 Vulnerability Validation Framework — Anik Das (T1a / T1b)

Purpose:
    Compare model-predicted damage states against observed damage states
    for the 1994 Northridge earthquake bridge portfolio.

Functions:
    - load_observed(csv_path)           → pd.DataFrame
    - load_predicted(csv_path)          → pd.DataFrame
    - merge_predictions(observed, predicted) → pd.DataFrame
    - confusion_matrix_ds(predicted_ds, observed_ds) → np.ndarray
    - compute_metrics(conf_matrix, damage_states) → dict
    - log_residual_analysis(merged_df)  → dict
    - plot_confusion_matrix(conf_matrix, damage_states, save_path)
    - plot_per_class_accuracy(metrics, save_path)
    - plot_log_residuals(residuals, save_path)
    - run_validation(observed_csv, predicted_csv, output_dir)

Design note (data swap):
    All I/O goes through load_observed() / load_predicted().
    To swap synthetic → real data, just change the csv_path argument.
    Column schema is fixed: bridge_id, latitude, longitude,
    observed_damage / predicted_damage (optional: sa_predicted).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# ── Damage state ordering ──────────────────────────────────────────────────
DAMAGE_STATES = ["none", "slight", "moderate", "extensive", "complete"]
DS_INDEX = {ds: i for i, ds in enumerate(DAMAGE_STATES)}


# ══════════════════════════════════════════════════════════════════════════
# 1.  I / O  LAYER  (swap synthetic ↔ real here)
# ══════════════════════════════════════════════════════════════════════════

def load_observed(csv_path: str) -> pd.DataFrame:
    """
    Load observed damage CSV.

    Expected columns:
        bridge_id       : str  — unique bridge identifier
        latitude        : float
        longitude       : float
        observed_damage : str  — one of DAMAGE_STATES

    Returns
    -------
    pd.DataFrame  with dtypes validated and damage labels lowercased.

    DATA-SWAP NOTE:
        Replace csv_path with Sirisha's real northridge_observed.csv
        when available (Sprint 2).  Schema must match the above.
    """
    df = pd.read_csv(csv_path, dtype={"bridge_id": str})
    df["observed_damage"] = df["observed_damage"].str.lower().str.strip()
    _validate_damage_column(df, "observed_damage", csv_path)
    return df


def load_predicted(csv_path: str) -> pd.DataFrame:
    """
    Load model-predicted damage CSV.

    Expected columns:
        bridge_id        : str
        predicted_damage : str  — most-likely damage state
        sa_predicted     : float (optional) — Sa(1.0s) used for residuals

    Returns
    -------
    pd.DataFrame with dtypes validated.

    DATA-SWAP NOTE:
        In Sprint 2, plug in the real pipeline output CSV from
        output/analysis/bridge_damage_results.csv (adjust column names
        if needed — only bridge_id and predicted_damage are required).
    """
    df = pd.read_csv(csv_path, dtype={"bridge_id": str})
    df["predicted_damage"] = df["predicted_damage"].str.lower().str.strip()
    _validate_damage_column(df, "predicted_damage", csv_path)
    return df


def _validate_damage_column(df: pd.DataFrame, col: str, path: str) -> None:
    bad = set(df[col].unique()) - set(DAMAGE_STATES)
    if bad:
        raise ValueError(
            f"[{path}] Column '{col}' contains invalid damage states: {bad}.\n"
            f"Allowed values: {DAMAGE_STATES}"
        )


# ══════════════════════════════════════════════════════════════════════════
# 2.  MERGE
# ══════════════════════════════════════════════════════════════════════════

def merge_predictions(observed: pd.DataFrame,
                      predicted: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join observed and predicted on bridge_id.

    Returns
    -------
    pd.DataFrame with columns:
        bridge_id, latitude, longitude,
        observed_damage, predicted_damage
        sa_predicted (if present in predicted)
    """
    keep_cols = ["bridge_id", "predicted_damage"]
    if "sa_predicted" in predicted.columns:
        keep_cols.append("sa_predicted")

    merged = observed.merge(predicted[keep_cols], on="bridge_id", how="inner")
    n_obs = len(observed)
    n_pred = len(predicted)
    n_merged = len(merged)
    print(f"[merge] observed={n_obs}  predicted={n_pred}  matched={n_merged}")
    return merged


# ══════════════════════════════════════════════════════════════════════════
# 3.  CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════

def confusion_matrix_ds(predicted_ds: pd.Series,
                        observed_ds: pd.Series) -> np.ndarray:
    """
    Build a 5×5 confusion matrix (rows = observed, cols = predicted).

    Returns
    -------
    np.ndarray  shape (5, 5), integer counts.
    """
    n = len(DAMAGE_STATES)
    matrix = np.zeros((n, n), dtype=int)
    for obs, pred in zip(observed_ds, predicted_ds):
        i = DS_INDEX[obs]
        j = DS_INDEX[pred]
        matrix[i, j] += 1
    return matrix


# ══════════════════════════════════════════════════════════════════════════
# 4.  METRICS
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(conf_matrix: np.ndarray,
                    damage_states: list = DAMAGE_STATES) -> dict:
    """
    Compute per-class and overall accuracy, precision, recall, F1.

    Returns
    -------
    dict with keys:
        overall_accuracy : float
        per_class        : dict[ds] -> {precision, recall, f1, support}
    """
    total = conf_matrix.sum()
    overall_accuracy = np.diag(conf_matrix).sum() / max(total, 1)

    per_class = {}
    for i, ds in enumerate(damage_states):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp   # predicted as ds, actually not
        fn = conf_matrix[i, :].sum() - tp   # actually ds, predicted as not
        support = conf_matrix[i, :].sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall / max(precision + recall, 1e-9))

        per_class[ds] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }

    return {"overall_accuracy": overall_accuracy, "per_class": per_class}


# ══════════════════════════════════════════════════════════════════════════
# 5.  LOG-RESIDUAL ANALYSIS  (L2 — bias / spread in sa_predicted)
# ══════════════════════════════════════════════════════════════════════════

def log_residual_analysis(merged_df: pd.DataFrame) -> dict:
    """
    Compute log-residuals between numeric damage-state index of predicted
    vs observed.  Also supports sa_predicted column if present.

    Returns
    -------
    dict with keys: mean_residual, std_residual, rmse, values (np.ndarray)
    """
    obs_idx = merged_df["observed_damage"].map(DS_INDEX).values.astype(float)
    pred_idx = merged_df["predicted_damage"].map(DS_INDEX).values.astype(float)

    # Shift by 1 to avoid log(0); residual in ordinal space
    residuals = pred_idx - obs_idx
    mean_r = residuals.mean()
    std_r = residuals.std()
    rmse = np.sqrt(np.mean(residuals ** 2))

    return {
        "mean_residual": mean_r,
        "std_residual": std_r,
        "rmse": rmse,
        "values": residuals,
    }


# ══════════════════════════════════════════════════════════════════════════
# 6.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(conf_matrix: np.ndarray,
                          damage_states: list,
                          save_path: str,
                          title: str = "Confusion Matrix — Damage State Prediction") -> None:
    """
    Save a colour-coded confusion matrix heatmap.
    Rows = Observed, Columns = Predicted.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_matrix, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="Count")

    # Annotate cells
    max_val = conf_matrix.max() if conf_matrix.max() > 0 else 1
    for i in range(len(damage_states)):
        for j in range(len(damage_states)):
            val = conf_matrix[i, j]
            text_color = "white" if val > 0.6 * max_val else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=11, color=text_color, fontweight="bold")

    labels = [ds.capitalize() for ds in damage_states]
    ax.set_xticks(range(len(damage_states)))
    ax.set_yticks(range(len(damage_states)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Damage State", fontsize=12)
    ax.set_ylabel("Observed Damage State", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {save_path}")


def plot_per_class_accuracy(metrics: dict,
                            save_path: str,
                            title: str = "Per-Class Metrics") -> None:
    """
    Save a grouped bar chart of precision / recall / F1 per damage state.
    """
    ds_list = DAMAGE_STATES
    precision = [metrics["per_class"][ds]["precision"] for ds in ds_list]
    recall = [metrics["per_class"][ds]["recall"] for ds in ds_list]
    f1 = [metrics["per_class"][ds]["f1"] for ds in ds_list]
    support = [metrics["per_class"][ds]["support"] for ds in ds_list]

    x = np.arange(len(ds_list))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="Precision", color="steelblue", edgecolor="k")
    ax.bar(x,         recall,    width, label="Recall",    color="darkorange", edgecolor="k")
    ax.bar(x + width, f1,        width, label="F1-Score",  color="seagreen",   edgecolor="k")

    # Annotate support counts above bars
    for i, s in enumerate(support):
        ax.text(x[i], max(precision[i], recall[i], f1[i]) + 0.03,
                f"n={s}", ha="center", fontsize=8, color="gray")

    ax.axhline(metrics["overall_accuracy"], color="crimson", ls="--", lw=1.5,
               label=f"Overall Accuracy = {metrics['overall_accuracy']:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([ds.capitalize() for ds in ds_list])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Damage State", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {save_path}")


def plot_log_residuals(residual_dict: dict,
                       save_path: str,
                       title: str = "Ordinal Residual Distribution") -> None:
    """
    Histogram of (predicted_DS_index − observed_DS_index) with normal fit.
    """
    residuals = residual_dict["values"]
    mean_r = residual_dict["mean_residual"]
    std_r = residual_dict["std_residual"]
    rmse = residual_dict["rmse"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(residuals.min() - 0.5, residuals.max() + 1.5, 1)
    ax.hist(residuals, bins=bins, color="steelblue", edgecolor="k",
            alpha=0.7, density=True, label="Residuals")

    x_norm = np.linspace(residuals.min() - 1, residuals.max() + 1, 200)
    ax.plot(x_norm, stats.norm.pdf(x_norm, mean_r, max(std_r, 1e-9)),
            "r-", lw=2, label="Normal fit")
    ax.axvline(0, color="k", ls="--", lw=1, label="Zero bias")
    ax.axvline(mean_r, color="darkorange", ls="--", lw=1.5,
               label=f"Mean = {mean_r:.2f}")

    ax.set_xlabel("Predicted DS index − Observed DS index", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.text(0.72, 0.93, f"RMSE = {rmse:.3f}\nStd  = {std_r:.3f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {save_path}")


def plot_damage_distribution(merged_df: pd.DataFrame,
                             save_path: str) -> None:
    """
    Side-by-side bar chart: observed vs predicted damage fractions.
    Mirrors the L2 comparison in notebook 06_validation.
    """
    obs_counts = merged_df["observed_damage"].value_counts()
    pred_counts = merged_df["predicted_damage"].value_counts()
    n = len(merged_df)

    obs_frac = [obs_counts.get(ds, 0) / n for ds in DAMAGE_STATES]
    pred_frac = [pred_counts.get(ds, 0) / n for ds in DAMAGE_STATES]

    x = np.arange(len(DAMAGE_STATES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, obs_frac,  width, label="Observed",  color="coral",     edgecolor="k")
    ax.bar(x + width / 2, pred_frac, width, label="Predicted", color="steelblue", edgecolor="k")

    ax.set_xticks(x)
    ax.set_xticklabels([ds.capitalize() for ds in DAMAGE_STATES])
    ax.set_ylabel("Fraction of Bridges", fontsize=12)
    ax.set_xlabel("Damage State", fontsize=12)
    ax.set_title("L2 — Observed vs Predicted Damage Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════
# 7.  ACCEPTANCE CRITERIA TABLE
# ══════════════════════════════════════════════════════════════════════════

def build_acceptance_table(metrics: dict,
                           residual_dict: dict) -> pd.DataFrame:
    """
    Return a summary DataFrame matching the format in notebook 06_validation.

    Thresholds (from notebook):
        Overall accuracy  ≥ 0.60
        |Mean residual|   < 0.50
        RMSE              < 1.50
    """
    rows = [
        {
            "Metric": "Overall Accuracy",
            "Value": f"{metrics['overall_accuracy']:.3f}",
            "Threshold": ">= 0.60",
            "Pass": metrics["overall_accuracy"] >= 0.60,
        },
        {
            "Metric": "Mean Residual (|bias|)",
            "Value": f"{residual_dict['mean_residual']:.3f}",
            "Threshold": "|bias| < 0.50",
            "Pass": abs(residual_dict["mean_residual"]) < 0.50,
        },
        {
            "Metric": "RMSE (ordinal)",
            "Value": f"{residual_dict['rmse']:.3f}",
            "Threshold": "RMSE < 1.50",
            "Pass": residual_dict["rmse"] < 1.50,
        },
    ]
    # Add per-class recall
    for ds in DAMAGE_STATES:
        recall = metrics["per_class"][ds]["recall"]
        rows.append({
            "Metric": f"Recall - {ds.capitalize()}",
            "Value": f"{recall:.3f}",
            "Threshold": ">= 0.30 (informational)",
            "Pass": recall >= 0.30,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 8.  TOP-LEVEL RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_validation(observed_csv: str,
                   predicted_csv: str,
                   output_dir: str = "output/validation") -> dict:
    """
    Full validation pipeline.

    Parameters
    ----------
    observed_csv  : path to CSV with observed damage states
    predicted_csv : path to CSV with model-predicted damage states
    output_dir    : directory for output plots and summary CSV

    Returns
    -------
    dict with keys: merged, conf_matrix, metrics, residuals, acceptance
    """
    os.makedirs(output_dir, exist_ok=True)

    # -- Load --
    observed  = load_observed(observed_csv)
    predicted = load_predicted(predicted_csv)

    # -- Merge --
    merged = merge_predictions(observed, predicted)

    # -- Confusion matrix --
    conf_matrix = confusion_matrix_ds(
        merged["predicted_damage"], merged["observed_damage"]
    )

    # -- Metrics --
    metrics = compute_metrics(conf_matrix)

    # -- Residuals --
    residuals = log_residual_analysis(merged)

    # -- Acceptance table --
    acceptance = build_acceptance_table(metrics, residuals)

    # -- Plots --
    plot_confusion_matrix(
        conf_matrix, DAMAGE_STATES,
        save_path=os.path.join(output_dir, "01_confusion_matrix.png"),
    )
    plot_per_class_accuracy(
        metrics,
        save_path=os.path.join(output_dir, "02_per_class_metrics.png"),
    )
    plot_log_residuals(
        residuals,
        save_path=os.path.join(output_dir, "03_residual_distribution.png"),
    )
    plot_damage_distribution(
        merged,
        save_path=os.path.join(output_dir, "04_damage_distribution_comparison.png"),
    )

    # -- Save acceptance table --
    acceptance_path = os.path.join(output_dir, "acceptance_criteria.csv")
    acceptance.to_csv(acceptance_path, index=False)
    print(f"[summary] Saved: {acceptance_path}")

    # -- Print summary --
    print("\n" + "="*55)
    print("  VALIDATION SUMMARY")
    print("="*55)
    print(f"  Bridges validated : {len(merged)}")
    print(f"  Overall Accuracy  : {metrics['overall_accuracy']:.3f}")
    print(f"  Mean Residual     : {residuals['mean_residual']:.3f}")
    print(f"  RMSE (ordinal)    : {residuals['rmse']:.3f}")
    print("-"*55)
    print(acceptance[["Metric", "Value", "Threshold", "Pass"]].to_string(index=False))
    print("="*55 + "\n")

    return {
        "merged": merged,
        "conf_matrix": conf_matrix,
        "metrics": metrics,
        "residuals": residuals,
        "acceptance": acceptance,
    }
