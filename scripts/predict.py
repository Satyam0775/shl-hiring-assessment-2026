"""Generate test predictions (clipped to 0-5) from a saved model."""
from _common import ROOT, base_parser, get_ctx


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--model", default=str(ROOT / "models" / "final" / "final_model.joblib"))
    p.add_argument("--manifest", choices=["test", "sample_submission"], default="test")
    p.add_argument("--out", default=str(ROOT / "reports" / "tables" / "test_predictions.csv"))
    p.add_argument("--allow-missing", action="store_true", help="predict even if some rows lack features (median-imputed)")
    args = p.parse_args()

    from pathlib import Path
    from src.models.predict import load_model, predict_scores
    from src.pipeline import build_merged
    ctx = get_ctx(args)
    model, meta = load_model(args.model)
    print(f"Model: {meta['experiment_id']} / {meta['model']} ({meta['feature_set']})")
    feats = build_merged(ctx, "test", args.manifest, strict=not args.allow_missing)
    pred = predict_scores(model, feats, ctx.cfg["prediction"]["min_score"], ctx.cfg["prediction"]["max_score"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote {len(pred)} predictions -> {args.out}  (min {pred.label.min():.2f}, max {pred.label.max():.2f}, mean {pred.label.mean():.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
