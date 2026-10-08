"""Model factory, experiment runner, experiment log, final-model fitting and selection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.cross_validation import cross_validate, make_folds, result_from_oof
from src.evaluation.metrics import clip_predictions, pearson, rmse
from src.features.preprocessing import GroupFeatureBuilder
from src.models.baseline import ConstantBaseline
from src.models.ensemble import (BASELINES, AveragingEnsemble, evaluate_oof_ensemble, members_to_json,
                                 oof_column, top_k_members)

LOG_COLUMNS = ["experiment_id", "feature_set", "model", "hyperparameters", "cv_rmse_mean", "cv_rmse_std",
               "cv_pearson_mean", "cv_pearson_std", "training_rmse", "notes", "kaggle_submission_id", "kaggle_score"]
MODEL_NAMES = ["mean_baseline", "median_baseline", "ridge", "extra_trees", "random_forest", "hist_gb", "lightgbm", "xgboost"]


# --------------------------------------------------------------------------- models
def build_model(name: str, seed: int = 42):
    """Regressors with fixed, deliberately modest hyper-parameters (769 samples: avoid over-tuning)."""
    if name == "mean_baseline":
        return ConstantBaseline("mean")
    if name == "median_baseline":
        return ConstantBaseline("median")
    if name == "ridge":
        return Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 20)))])
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=seed)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.4, n_jobs=-1, random_state=seed)
    if name == "hist_gb":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                             min_samples_leaf=15, l2_regularization=1.0, random_state=seed)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=15,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                             random_state=seed, verbose=-1, n_jobs=-1)
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.6,
                            min_child_weight=3, reg_lambda=1.0, tree_method="hist", random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")


def build_pipeline(model_name: str, groups: list[str], cfg: dict) -> Pipeline:
    seed = int(cfg["random_seed"])
    f = cfg["features"]
    if model_name in BASELINES:
        return Pipeline([("model", build_model(model_name, seed))])
    builder = GroupFeatureBuilder(groups=tuple(groups), use_pca=bool(f.get("use_pca", True)),
                                  pca_components=int(f.get("pca_components", 50)), random_state=seed)
    return Pipeline([("features", builder), ("model", build_model(model_name, seed))])


def feature_set_label(groups: list[str]) -> str:
    return "+".join(groups) if groups else "none"


def _scalar_params(est) -> dict:
    est = est.steps[-1][1] if isinstance(est, Pipeline) else est
    return {k: v for k, v in est.get_params(deep=False).items() if isinstance(v, (int, float, str, bool, type(None)))}


# --------------------------------------------------------------------------- experiments
def load_experiments(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["experiments"]


def enabled_groups(cfg: dict) -> set[str]:
    f = cfg["features"]
    out = set()
    if f.get("use_audio", True): out.add("audio")
    if f.get("use_text", True): out.add("text")
    if f.get("use_embeddings", True): out.add("embeddings")
    return out


def read_log(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p, encoding="utf-8")
    return pd.DataFrame(columns=LOG_COLUMNS)


def update_log(path: Path | str, new_rows: list[dict]) -> pd.DataFrame:
    """Upsert rows by (experiment_id, model). Manually entered Kaggle columns are preserved on re-runs."""
    old = read_log(path)
    new = pd.DataFrame(new_rows, columns=LOG_COLUMNS)
    if not old.empty and not new.empty:
        key = lambda d: d["experiment_id"].astype(str) + "||" + d["model"].astype(str)
        kaggle = old.set_index(key(old))[["kaggle_submission_id", "kaggle_score"]]
        for col in ("kaggle_submission_id", "kaggle_score"):
            mapped = key(new).map(kaggle[col])
            new[col] = new[col].where(new[col].notna(), mapped)
        old = old[~key(old).isin(key(new))]
    out = pd.concat([old, new], ignore_index=True) if not old.empty else new
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")
    return out


def run_experiments(df: pd.DataFrame, y, cfg: dict, experiments: list[dict], log_path: Path | str,
                    oof_path: Path | str, only: Optional[list[str]] = None, verbose: bool = True) -> pd.DataFrame:
    """Cross-validate every (experiment, model) on identical folds; log CV metrics + training RMSE."""
    y = np.asarray(y, dtype=float)
    lo, hi = cfg["prediction"]["min_score"], cfg["prediction"]["max_score"]
    folds = make_folds(y, int(cfg["validation"]["folds"]), int(cfg["random_seed"]), int(cfg["validation"].get("stratify_bins", 5)))
    allowed = enabled_groups(cfg)
    oof_path = Path(oof_path)
    oof_tbl = pd.read_csv(oof_path) if oof_path.exists() else pd.DataFrame()
    if oof_tbl.empty or len(oof_tbl) != len(y) or not (oof_tbl.get("filename") == df["filename"]).all():
        oof_tbl = pd.DataFrame({"filename": df["filename"].to_numpy(), "y_true": y})
    from src.evaluation.cross_validation import fold_id_vector
    oof_tbl["fold"] = fold_id_vector(folds, len(y))

    rows: list[dict] = []
    for exp in experiments:
        if only and exp["id"] not in only:
            continue
        if exp.get("type") == "ensemble":
            continue
        groups = list(exp.get("feature_set", []))
        if not set(groups) <= allowed:
            print(f"[skip] {exp['id']}: needs disabled feature groups {sorted(set(groups) - allowed)}")
            continue
        missing = [g for g in groups if not any(c.startswith(p) for c in df.columns for p in {"audio": ("aud_",), "text": ("txt_", "ling_"), "embeddings": ("emb_",)}[g])]
        if missing:
            print(f"[skip] {exp['id']}: feature columns for {missing} not present in the merged table")
            continue
        for model_name in exp["models"]:
            pipe = build_pipeline(model_name, groups, cfg)
            res = cross_validate(pipe, df, y, folds, (lo, hi))
            fitted = clone(pipe).fit(df, y)
            train_rmse = rmse(y, clip_predictions(fitted.predict(df), lo, hi))  # NOT a generalisation estimate
            oof_tbl[oof_column(exp["id"], model_name)] = res.oof
            rows.append({"experiment_id": exp["id"], "feature_set": feature_set_label(groups), "model": model_name,
                         "hyperparameters": json.dumps(_scalar_params(pipe), default=str),
                         "cv_rmse_mean": res.rmse_mean, "cv_rmse_std": res.rmse_std,
                         "cv_pearson_mean": res.pearson_mean, "cv_pearson_std": res.pearson_std,
                         "training_rmse": train_rmse, "notes": exp.get("notes", ""),
                         "kaggle_submission_id": np.nan, "kaggle_score": np.nan})
            if verbose:
                print(f"{exp['id']:8s} {feature_set_label(groups):24s} {model_name:15s} "
                      f"CV RMSE {res.rmse_mean:.4f}±{res.rmse_std:.4f} | CV r {res.pearson_mean:.4f} | train RMSE {train_rmse:.4f}")
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_tbl.to_csv(oof_path, index=False)
    log = update_log(log_path, rows)

    # EXP-07 ensemble
    for exp in experiments:
        if exp.get("type") == "ensemble" and (not only or exp["id"] in only):
            log = run_ensemble_experiment(df, y, cfg, exp, log, oof_tbl, folds, log_path, verbose)
    return log


def run_ensemble_experiment(df, y, cfg, exp, log, oof_tbl, folds, log_path, verbose=True) -> pd.DataFrame:
    members = top_k_members(log[log["experiment_id"] != exp["id"]], int(exp.get("top_k", 3)))
    if len(members) < 2:
        print("[skip] ensemble: fewer than 2 eligible members")
        return log
    lo, hi = cfg["prediction"]["min_score"], cfg["prediction"]["max_score"]
    res = evaluate_oof_ensemble(oof_tbl, members, y, folds)
    est = build_ensemble_estimator(members, cfg)
    train_rmse = rmse(y, clip_predictions(clone(est).fit(df, y).predict(df), lo, hi))
    name = f"ensemble_top{len(members)}"
    row = {"experiment_id": exp["id"], "feature_set": "mixed", "model": name,
           "hyperparameters": members_to_json(members), "cv_rmse_mean": res.rmse_mean, "cv_rmse_std": res.rmse_std,
           "cv_pearson_mean": res.pearson_mean, "cv_pearson_std": res.pearson_std, "training_rmse": train_rmse,
           "notes": exp.get("notes", ""), "kaggle_submission_id": np.nan, "kaggle_score": np.nan}
    if verbose:
        print(f"{exp['id']:8s} {'mixed':24s} {name:15s} CV RMSE {res.rmse_mean:.4f}±{res.rmse_std:.4f} | CV r {res.pearson_mean:.4f} | train RMSE {train_rmse:.4f}")
    return update_log(log_path, [row])


def build_ensemble_estimator(members: list[dict], cfg: dict) -> AveragingEnsemble:
    ms = []
    for m in members:
        groups = [] if m["feature_set"] in ("none", "") else m["feature_set"].split("+")
        ms.append((f'{m["experiment_id"]}__{m["model"]}', build_pipeline(m["model"], groups, cfg)))
    return AveragingEnsemble(members=ms)


# --------------------------------------------------------------------------- selection + final fit
def select_best(log: pd.DataFrame, include_baselines: bool = False) -> pd.Series:
    """Lowest mean CV RMSE (ties: highest CV Pearson). The ensemble competes like any other candidate."""
    d = log if include_baselines else log[~log["model"].isin(BASELINES)]
    if d.empty:
        raise ValueError("experiment log has no eligible rows; run scripts/train.py first")
    return d.sort_values(["cv_rmse_mean", "cv_pearson_mean"], ascending=[True, False]).iloc[0]


def row_to_estimator(row: pd.Series, cfg: dict):
    if str(row["model"]).startswith("ensemble"):
        return build_ensemble_estimator(json.loads(row["hyperparameters"])["members"], cfg)
    groups = [] if row["feature_set"] in ("none", "") else row["feature_set"].split("+")
    return build_pipeline(row["model"], groups, cfg)


def fit_final(row: pd.Series, df: pd.DataFrame, y, cfg: dict):
    """Fit the chosen candidate on ALL labelled training data. Returns (fitted, training_rmse)."""
    lo, hi = cfg["prediction"]["min_score"], cfg["prediction"]["max_score"]
    est = row_to_estimator(row, cfg).fit(df, np.asarray(y, float))
    return est, rmse(y, clip_predictions(est.predict(df), lo, hi))


def save_final(model, row: pd.Series, out_dir: Path | str, tag: str, extra: Optional[dict] = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.joblib"
    meta = {"experiment_id": row["experiment_id"], "model": row["model"], "feature_set": row["feature_set"],
            "cv_rmse_mean": float(row["cv_rmse_mean"]), "cv_pearson_mean": float(row["cv_pearson_mean"]), **(extra or {})}
    joblib.dump({"model": model, "meta": meta}, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def get_feature_importance(fitted) -> Optional[pd.DataFrame]:
    """Tree-model importances with real feature names (None for baselines/ensembles/ridge)."""
    if not isinstance(fitted, Pipeline) or "features" not in fitted.named_steps:
        return None
    model = fitted.named_steps["model"]
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return None
    names = fitted.named_steps["features"].get_feature_names_out()
    return pd.DataFrame({"feature": names, "importance": np.asarray(imp, float)}).sort_values("importance", ascending=False).reset_index(drop=True)
