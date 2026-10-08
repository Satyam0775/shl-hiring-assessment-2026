"""Run the experiment ladder (5-fold CV on identical folds) and optionally fit the final model."""
from _common import ROOT, base_parser, get_ctx


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--only", nargs="*", default=None, help="run only these experiment ids, e.g. EXP-04 EXP-06")
    p.add_argument("--fit-final", action="store_true", help="after CV, fit the selected model on all training data and save it")
    p.add_argument("--exp", default=None, help="with --fit-final: force this experiment id instead of the CV-best")
    p.add_argument("--model", default=None, help="with --fit-final --exp: model name within that experiment")
    p.add_argument("--no-cv", action="store_true", help="skip CV and use the existing experiment log")
    args = p.parse_args()

    from src.config import repo_dir
    from src.models.train import (fit_final, get_feature_importance, load_experiments, read_log, run_experiments,
                                  save_final, select_best)
    ctx = get_ctx(args)
    cfg = ctx.cfg
    exp_dir = repo_dir(cfg, "experiments_dir")
    rep_dir = repo_dir(cfg, "reports_dir")
    log_path = exp_dir / "experiment_log.csv"

    from src.pipeline import load_train_matrix
    X, y = load_train_matrix(ctx)
    print(f"Training matrix: {X.shape[0]} rows, {X.shape[1] - 1} feature columns")

    if not args.no_cv:
        exps = load_experiments(ROOT / cfg["experiments_file"])
        log = run_experiments(X, y, cfg, exps, log_path, rep_dir / "tables" / "oof_predictions.csv", only=args.only)
    else:
        log = read_log(log_path)

    if args.fit_final:
        if args.exp:
            d = log[log["experiment_id"] == args.exp]
            if args.model:
                d = d[d["model"] == args.model]
            if d.empty:
                raise SystemExit(f"No log row for experiment={args.exp} model={args.model}; run CV first")
            row = d.sort_values("cv_rmse_mean").iloc[0]
        else:
            row = select_best(log)
        print(f"Selected: {row['experiment_id']} / {row['model']} ({row['feature_set']}) "
              f"CV RMSE {row['cv_rmse_mean']:.4f}")
        model, train_rmse = fit_final(row, X, y, cfg)
        print(f"Training RMSE (in-sample, NOT a generalisation estimate): {train_rmse:.4f}")
        out = exp_dir.parent / cfg["paths"]["models_dir"] / "final"
        tag = f"{row['experiment_id']}__{row['model']}"
        path = save_final(model, row, out, tag, {"training_rmse": train_rmse})
        if not args.exp:
            save_final(model, row, out, "final_model", {"training_rmse": train_rmse})
        imp = get_feature_importance(model)
        if imp is not None:
            imp.to_csv(rep_dir / "tables" / "feature_importance.csv", index=False, encoding="utf-8")
        print(f"Saved model: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
