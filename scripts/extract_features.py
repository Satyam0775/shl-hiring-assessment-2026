"""Acoustic + text + embedding features, then fusion into merged feature tables."""
from _common import add_split_args, base_parser, get_ctx, splits


def main() -> int:
    p = base_parser(__doc__)
    add_split_args(p)
    p.add_argument("--allow-missing", action="store_true", help="do not stop if some rows lack features")
    args = p.parse_args()
    from src.pipeline import build_merged, run_audio_features, run_embeddings, run_text_features
    ctx = get_ctx(args)
    f = ctx.cfg["features"]
    for split in splits(args.split):
        if f.get("use_audio", True):
            run_audio_features(ctx, split, args.manifest, args.force, args.limit)
        if f.get("use_text", True):
            run_text_features(ctx, split, args.force)
        if f.get("use_embeddings", True):
            run_embeddings(ctx, split, args.force)
        merged = build_merged(ctx, split, args.manifest, strict=not args.allow_missing)
        print(f"[{split}] merged feature table: {merged.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
