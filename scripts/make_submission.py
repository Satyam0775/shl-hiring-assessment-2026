"""Validate predictions against the official sample_submission.csv and write a versioned submission."""
from _common import ROOT, base_parser, get_ctx


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--predictions", default=str(ROOT / "reports" / "tables" / "test_predictions.csv"))
    p.add_argument("--version", type=int, default=None, help="e.g. 4 -> submission_v04_<name>.csv")
    p.add_argument("--name", default="final", help="suffix, e.g. baseline | audio | text | fusion | final")
    p.add_argument("--out-dir", default=str(ROOT / "submissions"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    import pandas as pd
    from src.data.loader import read_metadata_csv
    from src.data.validator import MetadataInconsistencyError, assert_test_matches_sample
    from src.submission import SubmissionError, submission_path, validate_submission, write_submission

    ctx = get_ctx(args)
    sample = read_metadata_csv(ctx.paths.sample_submission_csv, "sample_submission.csv")
    test = read_metadata_csv(ctx.paths.test_csv, "test.csv")
    pred = pd.read_csv(args.predictions, encoding="utf-8")
    pc = ctx.cfg["prediction"]
    try:
        # Hard stop #1: test.csv vs sample_submission.csv. Skipped only if predictions were made explicitly
        # for the sample_submission filenames (predict.py --manifest sample_submission).
        if set(pred["filename"]) != set(sample["filename"]):
            assert_test_matches_sample(test, sample)
        valid = validate_submission(pred, sample, pc["min_score"], pc["max_score"])
        path = write_submission(valid, submission_path(args.out_dir, args.version, args.name),
                                protected=ctx.paths.sample_submission_csv, overwrite=args.overwrite)
    except (MetadataInconsistencyError, SubmissionError) as exc:
        print(f"STOP - no submission written.\n{exc}")
        return 2
    print(f"Valid submission written: {path}  ({len(valid)} rows, columns {list(valid.columns)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
