"""Batch ASR (faster-whisper) with caching and resume."""
from _common import add_split_args, base_parser, get_ctx, splits


def main() -> int:
    p = base_parser(__doc__)
    add_split_args(p)
    p.add_argument("--retry-failed", action="store_true", help="re-run files that previously failed")
    args = p.parse_args()
    from src.pipeline import run_transcription
    ctx = get_ctx(args)
    for split in splits(args.split):
        df = run_transcription(ctx, split, args.manifest, args.force, args.retry_failed, args.limit)
        n_bad = int((df["status"] != "ok").sum()) if len(df) else 0
        print(f"[{split}] transcripts cached: {len(df)} (failed: {n_bad})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
