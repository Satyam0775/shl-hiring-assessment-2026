# SHL Grammar Scoring Engine

Predict a **continuous grammar score (0-5)** from ~45-60 s of spoken English. Built for the SHL Hiring Assessment 2026 (Research Engineer). The pipeline is local, open-source, cached, leakage-safe and explainable.

> **Status: finalized local run completed.** The repository now contains the validated final experiment log, generated predictions, and the final v02 submission copy used for publication.

## Overview
`WAV -> preprocessing -> {acoustic features, ASR -> transcript -> {linguistic features, embeddings}} -> fusion -> regressor -> clip(0, 5)`

## Assignment
Build a Grammar Scoring Engine; deliver a notebook with approach, preprocessing, architecture, evaluation, the **training RMSE**, relevant metrics, visualisations, and Kaggle-format predictions. Leaderboard metrics: **RMSE (lower is better)** and **Pearson correlation (higher is better)**.

## Dataset
Local only (never committed). Expected layout under your *data root*:

```
Dataset_Final/
  train/  *.wav          (769 files)
  test/   *.wav          (216 files)
  train.csv              filename,label
  test.csv               filename,label   (label = -1, never used)
  sample_submission.csv  filename,label
```

**Known metadata inconsistency (found in the supplied CSVs, see `reports/metadata_audit_initial.md`):**
`test.csv` has 216 rows, `sample_submission.csv` has 204; only 25 filenames are shared. 212 `test.csv` names also occur in `train.csv` (names restart at `audio_0`, while train names reach `audio_5073`; probably separate numbering per folder, but verify with `--hash-check`). The code does **not** guess: `make_submission.py` stops until you decide which file defines the scored test set.

## Grammar Rubric
| Score | Meaning |
|---|---|
| 1 | Struggles with sentence structure/syntax; limited control of simple, memorised patterns |
| 2 | Limited understanding; consistent basic errors; may leave sentences incomplete |
| 3 | Decent grasp of one of structure/grammar, but errors in the other |
| 4 | Strong, consistent control; occasional minor errors, rarely impeding meaning; can self-correct |
| 5 | High accuracy, adept control of complex grammar; seldom makes noticeable mistakes; self-corrects |

The target is formulated as **regression** (ordinal, half-point labels, RMSE/Pearson reward closeness).

## Architecture
```
                       WAV AUDIO
                           |
                 AUDIO PREPROCESSING (mono, 16 kHz)
                           |
            +--------------+---------------+
            v                              v
   ACOUSTIC FEATURES (aud_)        ASR: faster-whisper
            |                              |
            |                         TRANSCRIPT (cached)
            |                  +-----------+-----------+
            |                  v                       v
            |       LINGUISTIC FEATURES        TEXT EMBEDDING (MiniLM)
            |         (txt_, ling_)                (emb_)
            |                  |                       |
            +------------------+-----------+-----------+
                                           v
                          FEATURE FUSION (per-fold: impute, PCA on emb_)
                                           v
                       REGRESSION (ExtraTrees / LightGBM / XGBoost ...)
                                           v
                              clip(prediction, 0, 5)
```

## Audio Processing
`src/audio/preprocessing.py`: load -> mono -> 16 kHz -> DC removal -> safe peak normalisation (for spectral features). Corrupt/empty/short files are recorded, not dropped. Originals are never modified.

## ASR
`faster-whisper`, `small.en` (fallback `base.en`), CUDA if available else CPU/int8. Open-source, free, offline, batch-friendly. Transcripts cached in `data/processed/transcripts/` with `filename, transcript, duration, status, error_message, asr_model`; resumable; failures are logged and skipped (`--retry-failed` re-runs them).

## Acoustic Features
Duration; RMS/ZCR stats; spectral centroid/bandwidth/rolloff/flatness/contrast; 13 MFCC + delta mean/std; speech & silence ratio; pause count/mean/max/total; onset-rate (speaking-rate proxy); rough YIN pitch stats.

## Linguistic Features
Word/sentence statistics, lexical diversity, repeated words, fillers, punctuation; function-word ratios (pronoun, auxiliary/modal, conjunction, preposition, determiner), subordination, passive/perfect/conditional markers, simple agreement/article rule flags. **These are proxies, not a grammar checker.**

## Text Embeddings
`sentence-transformers/all-MiniLM-L6-v2` (384-d). PCA (default 50) is fitted **inside each training fold only**. Optional; compared in EXP-03 / 05 / 05b.

## Model
Tree-based tabular regressors on fused features (769 samples -> compact, robust, explainable; no scaling needed; handles NaNs and non-linearity). Modest fixed hyper-parameters to avoid over-tuning. Final prediction = `clip(pred, 0, 5)`.

## Validation
Stratified 5-fold CV (stratified on equal-frequency score bins, fixed seed, identical folds for all candidates). Imputer, PCA and model are fitted on training folds only (`GroupFeatureBuilder` inside an sklearn `Pipeline`). Test data never influences selection.

## Metrics
`rmse` (lower better) and `pearson` (higher better) in `src/evaluation/metrics.py` (empty/NaN/inf/shape-mismatch raise; constant predictions give Pearson 0.0). **Training RMSE is in-sample and reported separately from CV RMSE.**

## Experiments
Defined in `experiments/configs/experiments.yaml`; results logged to `experiments/experiment_log.csv`.

| ID | Features | Models |
|---|---|---|
| EXP-00 / 01 | none | mean / median baseline |
| EXP-02 | audio | ExtraTrees, LightGBM |
| EXP-03 | text | ExtraTrees, LightGBM |
| EXP-04 | audio + text | ExtraTrees, RandomForest, HistGB, LightGBM, XGBoost |
| EXP-05 / 05b | embeddings / text + embeddings | Ridge, ExtraTrees, LightGBM |
| EXP-06 | audio + text + embeddings | ExtraTrees, LightGBM, XGBoost |
| EXP-07 | ensemble of top-3 distinct candidates | averaging (kept only if CV improves) |

Selection rule: lowest mean CV RMSE, tie-break highest mean CV Pearson.

## Results

Best model selected by the fixed rule (lowest CV RMSE; tie-break highest CV Pearson):

| Winner | Feature set | CV RMSE (mean +/- std) | CV Pearson (mean +/- std) | Training RMSE (in-sample) |
|---|---|---|---|---|
| EXP-07 `ensemble_top3` | mixed (top-3 EXP-06 models) | 0.6876 +/- 0.0502 | 0.8338 +/- 0.0283 | 0.1475 |

Top single-model candidates from `experiments/experiment_log.csv`:

| Experiment | Model | CV RMSE | CV Pearson |
|---|---|---|---|
| EXP-06 | xgboost | 0.6921 | 0.8302 |
| EXP-06 | lightgbm | 0.6942 | 0.8290 |
| EXP-06 | extra_trees | 0.7102 | 0.8222 |

Submission artifacts currently present:

| File | Purpose | Notes |
|---|---|---|
| `submissions/submission_final_v01.csv` | earlier frozen final | kept for traceability |
| `submissions/submission_v02_shrink_08.csv` | validated v02 candidate | 216 rows, schema `filename,label` |
| `submissions/submission_final_v02.csv` | exact copy of v02 candidate | SHA-256 identical to `submission_v02_shrink_08.csv` |

Kaggle public leaderboard values are not auto-recorded in this repository metadata. If you submit newer files, log submission IDs/scores in `experiments/experiment_log.csv` (columns `kaggle_submission_id`, `kaggle_score`) and `reports/final_report.md`.

## Error Analysis
Notebook 06 contains the full error-analysis workflow (worst predictions, over/under-prediction, score-band errors, residual and prediction distributions). The strongest signal from experiments is that acoustic + handcrafted text features + embeddings outperform text-only or embedding-only setups, while final ensembling gives a small but consistent CV gain over the best single model.

## Local Setup (Windows, VS Code, PowerShell)
```powershell
# 1. Python 3.11 (https://www.python.org/downloads/) - check:
py -3.11 --version

# 2. From the project folder
cd SHL-Grammar-Scoring
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```
In VS Code: *Python: Select Interpreter* -> `.venv`. Optional NVIDIA GPU: install a CUDA build of PyTorch/CTranslate2 per their docs; CPU works out of the box (CPU transcription speed depends on your hardware; time `--limit 5` first and extrapolate).

## Dataset Configuration
Either set an environment variable (current session) or pass `--data-root` each time:
```powershell
$env:SHL_DATA_ROOT = "C:/Users/USERNAME/path/Dataset_Final"
```
or edit `data_root` in `configs/config.yaml`. Precedence: `--data-root` > `SHL_DATA_ROOT` > config.

## Running the Pipeline
```powershell
python scripts/audit_dataset.py                       # add --hash-check to compare same-named train/test audio
python scripts/transcribe.py --limit 5                # smoke test, then without --limit (resumable; --retry-failed, --force)
python scripts/extract_features.py                    # acoustic + text + embeddings + fusion (cached; --force)
python scripts/train.py --fit-final                   # 5-fold CV for all experiments, select best, fit on all train data
python scripts/predict.py                             # clipped test predictions -> reports/tables/test_predictions.csv
python scripts/make_submission.py --version 4 --name fusion
```
Versioned submissions per experiment (e.g. a baseline first):
```powershell
python scripts/train.py --no-cv --fit-final --exp EXP-00 --model mean_baseline
python scripts/predict.py --model models/final/EXP-00__mean_baseline.joblib
python scripts/make_submission.py --version 1 --name baseline
```
If `sample_submission.csv` (not `test.csv`) turns out to define the scored files, pass `--manifest sample_submission` to `transcribe.py`, `extract_features.py` and `predict.py`; each listed file must exist in `test/`.

## Running Tests
```powershell
python -m pytest tests -q
```
Tests use only synthetic inputs (no dataset required).

## Generating Submission
`make_submission.py` validates against the official `sample_submission.csv` (columns + order, duplicates, NaNs, row count, filename set, range 0-5), writes `submissions/submission_vNN_<name>.csv`, never overwrites the official file or an existing version (without `--overwrite`), and **stops** with an explanation if `test.csv` and `sample_submission.csv` disagree. Final file: `submissions/submission_final.csv` (`--name final` without `--version`; notebook 06 also writes it).

For this public release, `submissions/submission_final_v02.csv` is intentionally a direct copy of `submissions/submission_v02_shrink_08.csv`.

## GitHub Instructions
```powershell
git init
git add .
git status                      # confirm: no .wav, no data/processed, no .venv, no .joblib
git commit -m "SHL grammar scoring pipeline"
git branch -M main
git remote add origin https://github.com/<user>/SHL-Grammar-Scoring.git
git push -u origin main
```
The repository must **not** contain competition data: `.gitignore` excludes `*.wav`, `data/raw/`, `data/processed/`, `*.zip`, `.env`, virtual environments and model checkpoints. Competition data is subject to the competition rules and must not be redistributed.

## Reproducibility
Fixed seed (`random_seed: 42`), identical CV folds for all candidates, cached intermediate tables, config-driven settings, pinned minimum versions in `requirements.txt`. Whisper/embedding model downloads need internet on first use. Small numeric differences across hardware/library versions are possible.

## Limitations
ASR can silently repair grammar errors; linguistic features are proxies; ~769 labels give noisy CV estimates; human-rater noise bounds RMSE; acoustic features reflect fluency not grammar; the `test.csv` / `sample_submission.csv` inconsistency must be resolved externally.

## Future Work
Grammatical-error-correction-based features (open-source), fine-tuned small speech/text encoders, ordinal/quantile calibration, repeated CV with paired tests, stacking, word-level ASR confidence features.

## Project Layout
```
configs/      config.yaml (all settings)         src/        library code
scripts/      CLI entry points                   notebooks/  01..06 (06 = final submission)
experiments/  experiment_log.csv, experiments.yaml    reports/  figures, tables, final_report.md
models/final  saved models (git-ignored)         submissions/ versioned CSVs
data/processed  cached features (git-ignored)    tests/      pytest suite
```
