"""Plot/table helpers shared by the notebooks. Every function only visualises data passed in."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

from src.config import project_root

import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = project_root() / "reports" / "figures"


def _save(fig, name: Optional[str], fig_dir: Optional[Path] = None):
    if name:
        d = Path(fig_dir) if fig_dir else FIG_DIR
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png", dpi=130, bbox_inches="tight")
    return fig


def plot_label_distribution(y, name="label_distribution", fig_dir=None):
    y = pd.Series(np.asarray(y, float))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    counts = y.value_counts().sort_index()
    ax.bar([str(i) for i in counts.index], counts.values, color="#3b6ea5")
    ax.set(title="Training label distribution", xlabel="Grammar score", ylabel="Samples")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    return _save(fig, name, fig_dir)


def plot_hist(values, title, xlabel, name=None, bins=30, fig_dir=None):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(pd.Series(values).dropna(), bins=bins, color="#3b6ea5", edgecolor="white")
    ax.set(title=title, xlabel=xlabel, ylabel="Files")
    return _save(fig, name, fig_dir)


def plot_value_counts(series, title, xlabel, name=None, fig_dir=None):
    vc = pd.Series(series).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([str(i) for i in vc.index], vc.values, color="#3b6ea5")
    ax.set(title=title, xlabel=xlabel, ylabel="Files")
    return _save(fig, name, fig_dir)


def plot_waveform_spectrogram(y, sr, title="", name="waveform_spectrogram", fig_dir=None):
    import librosa
    import librosa.display

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5))
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#3b6ea5")
    axes[0].set(title=f"Waveform {title}", xlabel="Time (s)", ylabel="Amplitude")
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=axes[1])
    axes[1].set(title="Spectrogram (dB)")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    fig.tight_layout()
    return _save(fig, name, fig_dir)


def plot_actual_vs_pred(y, pred, title="Actual vs predicted (out-of-fold)", name="actual_vs_predicted", fig_dir=None):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    jitter = np.random.default_rng(0).normal(0, 0.04, len(y))
    ax.scatter(y + jitter, pred, s=14, alpha=0.5, color="#3b6ea5")
    ax.plot([0, 5], [0, 5], "r--", lw=1, label="perfect")
    ax.set(xlim=(-0.3, 5.3), ylim=(-0.3, 5.3), xlabel="Actual", ylabel="Predicted", title=title)
    ax.legend()
    return _save(fig, name, fig_dir)


def plot_residuals(y, pred, name="residual_distribution", fig_dir=None):
    res = np.asarray(pred, float) - np.asarray(y, float)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(res, bins=40, color="#c0504d", edgecolor="white")
    ax.axvline(0, color="k", lw=1)
    ax.set(title=f"Residuals (pred - actual): mean {res.mean():+.3f}, std {res.std():.3f}", xlabel="Residual")
    return _save(fig, name, fig_dir)


def plot_prediction_distribution(y, pred, name="prediction_distribution", fig_dir=None):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(0, 5, 26)
    ax.hist(y, bins=bins, alpha=0.55, label="actual", color="#3b6ea5")
    ax.hist(pred, bins=bins, alpha=0.55, label="predicted", color="#e08a1e")
    ax.set(title="Actual vs predicted score distribution", xlabel="Score")
    ax.legend()
    return _save(fig, name, fig_dir)


def error_by_range(y, pred) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y, float), "pred": np.asarray(pred, float)})
    d["range"] = pd.cut(d["y"], [-0.01, 1.0, 2.0, 3.0, 4.0, 5.0], labels=["0-1", "1-2", "2-3", "3-4", "4-5"])
    d["err"] = d["pred"] - d["y"]
    g = d.groupby("range", observed=True)
    return pd.DataFrame({"n": g.size(), "rmse": g["err"].apply(lambda e: float(np.sqrt(np.mean(e ** 2)))),
                         "mean_error(bias)": g["err"].mean()}).reset_index()


def plot_error_by_range(tbl, name="error_by_range", fig_dir=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(tbl["range"].astype(str), tbl["rmse"], color="#c0504d", label="RMSE")
    ax.plot(tbl["range"].astype(str), tbl["mean_error(bias)"], "ko-", label="mean error (bias)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(title="Error by true-score range", xlabel="True score", ylabel="Error")
    ax.legend()
    return _save(fig, name, fig_dir)


def worst_predictions(filenames, y, pred, k=15) -> pd.DataFrame:
    d = pd.DataFrame({"filename": np.asarray(filenames), "y_true": np.asarray(y, float), "y_pred": np.asarray(pred, float)})
    d["error"] = d["y_pred"] - d["y_true"]
    d["abs_error"] = d["error"].abs()
    d["direction"] = np.where(d["error"] > 0, "over-predicted", "under-predicted")
    return d.sort_values("abs_error", ascending=False).head(k).reset_index(drop=True)


def plot_model_comparison(log: pd.DataFrame, name="model_comparison", fig_dir=None):
    d = log.copy()
    d["label"] = d["experiment_id"] + " " + d["model"]
    d = d.sort_values("cv_rmse_mean", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, 0.32 * len(d) + 1)), sharey=True)
    axes[0].barh(d["label"], d["cv_rmse_mean"], xerr=d["cv_rmse_std"], color="#3b6ea5")
    axes[0].set(title="CV RMSE (lower is better)", xlabel="RMSE")
    axes[1].barh(d["label"], d["cv_pearson_mean"], xerr=d["cv_pearson_std"], color="#4f9d69")
    axes[1].set(title="CV Pearson r (higher is better)", xlabel="Pearson")
    fig.tight_layout()
    return _save(fig, name, fig_dir)


def fold_metrics_from_oof(oof: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    from src.evaluation.metrics import pearson, rmse

    rows = []
    for c in columns:
        for k, g in oof.groupby("fold"):
            rows.append({"candidate": c, "fold": int(k), "rmse": rmse(g["y_true"], g[c]), "pearson": pearson(g["y_true"], g[c])})
    return pd.DataFrame(rows)


def plot_cv_folds(fold_df: pd.DataFrame, name="cv_fold_performance", fig_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(axes, ["rmse", "pearson"], ["RMSE per fold (lower better)", "Pearson per fold (higher better)"]):
        piv = fold_df.pivot(index="fold", columns="candidate", values=col)
        piv.plot(kind="bar", ax=ax, rot=0)
        ax.set(title=title, xlabel="Fold")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, name, fig_dir)


def plot_feature_importance(imp: pd.DataFrame, top: int = 25, name="feature_importance", fig_dir=None):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(d) + 1.5))
    ax.barh(d["feature"], d["importance"], color="#3b6ea5")
    ax.set(title=f"Top {len(d)} feature importances (impurity-based; indicative, not causal)")
    return _save(fig, name, fig_dir)
