# Final Report - SHL Grammar Scoring Engine

## 1. Executive Summary
This repository predicts a continuous grammar score (0-5) from spoken-English audio clips using a leakage-safe local pipeline:

1. Audio preprocessing and acoustic feature extraction.
2. ASR transcription using faster-whisper.
3. Transcript-derived handcrafted linguistic features and MiniLM embeddings.
4. Feature fusion and tree-based regression.
5. Selection by stratified 5-fold cross-validated RMSE (tie-break Pearson), then clipping predictions to [0, 5].

The final selected candidate is an ensemble of the top 3 EXP-06 models, logged as EXP-07.

## 2. Data and Integrity Notes
- Train set: 769 labelled clips.
- Test set: 216 clips (from `data/dataset/test.csv`).
- Submission candidate integrity check (`submissions/submission_v02_shrink_08.csv`):
	- rows: 216
	- columns: `filename,label`
	- missing values: 0
	- duplicate filenames: 0
	- non-numeric labels: 0
	- label range: min 2.1296, max 4.5978
	- filename set matches `test.csv`: true

Known external metadata inconsistency remains documented in `reports/metadata_audit_initial.md` (`test.csv` vs `sample_submission.csv` row mismatch). The code intentionally does not guess; it enforces manifest checks.

## 3. Final Modeling Outcome
From `experiments/experiment_log.csv`:

| Rank | Experiment | Model | Feature set | CV RMSE (mean +/- std) | CV Pearson (mean +/- std) | Training RMSE |
|---|---|---|---|---|---|---|
| 1 | EXP-07 | ensemble_top3 | mixed | 0.6876 +/- 0.0502 | 0.8338 +/- 0.0283 | 0.1475 |
| 2 | EXP-06 | xgboost | audio+text+embeddings | 0.6921 +/- 0.0474 | 0.8302 +/- 0.0256 | 0.2901 |
| 3 | EXP-06 | lightgbm | audio+text+embeddings | 0.6942 +/- 0.0522 | 0.8290 +/- 0.0286 | 0.0928 |

Interpretation: fusion features dominate single-modality setups; shallow averaging of the top candidates provides a modest but repeatable gain on CV RMSE and CV Pearson.

## 4. Submission Artifacts
Current repository submission files:

| File | Role |
|---|---|
| `submissions/submission_final_v01.csv` | previous final snapshot |
| `submissions/submission_v02_shrink_08.csv` | validated v02 candidate |
| `submissions/submission_v02_shrink_09.csv` | alternate v02 candidate |
| `submissions/submission_v02_shrink_10.csv` | alternate v02 candidate |
| `submissions/submission_final_v02.csv` | release final, exact copy of v02_shrink_08 |

`submission_final_v02.csv` was copied byte-for-byte from `submission_v02_shrink_08.csv` and verified with matching SHA-256 hash.

## 5. Security and Publication Readiness
Focused repository scan over source/docs/config/test/report/submission paths found no hardcoded credentials or tokens. The largest non-environment tracked artifacts are model files around 19.77 MB each (excluded by `.gitignore`).

Sensitive/large local assets remain excluded via `.gitignore`, including:
- competition audio and local dataset folders,
- processed intermediate files,
- virtual environments,
- model joblib artifacts.

## 6. Limitations and Next Actions
1. Kaggle leaderboard submission IDs/scores are not auto-ingested into metadata; they should be logged manually in `experiments/experiment_log.csv` and release notes when available.
2. The official scored manifest ambiguity (`test.csv` vs `sample_submission.csv`) is an external competition governance issue and must be resolved from the official challenge source.
3. Training RMSE is intentionally reported as in-sample only; model quality decisions should continue to rely on CV RMSE and CV Pearson.
