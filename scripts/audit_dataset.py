"""Data + audio integrity audit. Reports inconsistencies; never fixes them silently."""
from _common import ROOT, base_parser, get_ctx


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--no-audio", action="store_true", help="audit CSV metadata only")
    p.add_argument("--hash-check", action="store_true", help="MD5-compare same-named train/test audio files")
    p.add_argument("--out-dir", default=str(ROOT / "reports" / "tables"))
    args = p.parse_args()

    from src.data.validator import run_audit
    ctx = get_ctx(args)
    exp = ctx.cfg.get("expected", {})
    report = run_audit(ctx.paths, exp.get("train_rows"), exp.get("test_rows"),
                       check_audio=not args.no_audio, hash_check=args.hash_check)
    from pathlib import Path
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report.to_json(out / "dataset_audit.json")
    (out / "dataset_audit.md").write_text(report.to_markdown(), encoding="utf-8")
    if report.inventory is not None:
        report.inventory.drop(columns=["path"]).to_csv(out / "audio_inventory.csv", index=False, encoding="utf-8")
    print(report.to_markdown().split("## Statistics")[0])
    print(f"Saved: {out / 'dataset_audit.md'}")
    return 1 if report.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
