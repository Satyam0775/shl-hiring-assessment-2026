# Dataset audit report

**Errors:** 3  |  **Warnings:** 1

## Findings
- **[ERROR] TEST_VS_SAMPLE_ROW_COUNT** - test.csv has 216 rows but sample_submission.csv has 204 rows
- **[ERROR] TEST_VS_SAMPLE_FILENAMES** - test.csv and sample_submission.csv filename sets differ: 25 shared, 191 only in test.csv, 179 only in sample_submission.csv
- **[WARNING] TRAIN_TEST_NAME_OVERLAP** - 212 filenames appear in BOTH train.csv and test.csv. Files live in separate folders, so this may be benign re-numbering, but verify the audio is not identical (run the audit with --hash-check).
- **[INFO] SAMPLE_TRAIN_NAME_OVERLAP** - 105 sample_submission filenames also appear in train.csv
- **[ERROR] SAMPLE_FILES_NOT_IN_TEST_DIR** - 179 sample_submission filenames have no audio in the test folder
- **[INFO] TEST_DURATION_OUTLIERS** - 3 test files are outside 30-90 s (brief says ~45-60 s)
- **[INFO] TRAIN_DURATION_OUTLIERS** - 14 train files are outside 30-90 s (brief says ~45-60 s)

## Statistics
```json
{
  "train": {
    "rows": 769,
    "columns": [
      "filename",
      "label"
    ],
    "missing_values": {
      "filename": 0,
      "label": 0
    },
    "duplicate_filenames": 0,
    "filename_pattern_match": 769,
    "filename_number_min": 0,
    "filename_number_max": 5073
  },
  "test": {
    "rows": 216,
    "columns": [
      "filename",
      "label"
    ],
    "missing_values": {
      "filename": 0,
      "label": 0
    },
    "duplicate_filenames": 0,
    "filename_pattern_match": 216,
    "filename_number_min": 0,
    "filename_number_max": 215
  },
  "sample_submission": {
    "rows": 204,
    "columns": [
      "filename",
      "label"
    ],
    "missing_values": {
      "filename": 0,
      "label": 0
    },
    "duplicate_filenames": 0,
    "filename_pattern_match": 204,
    "filename_number_min": 13,
    "filename_number_max": 1329
  },
  "train_label": {
    "min": 0.0,
    "max": 5.0,
    "mean": 3.3133940182054618,
    "std": 1.23899953424107,
    "value_counts": {
      "0.0": 37,
      "1.0": 1,
      "1.5": 3,
      "2.0": 102,
      "2.5": 79,
      "3.0": 174,
      "3.5": 64,
      "4.0": 124,
      "4.5": 52,
      "5.0": 133
    }
  },
  "test_label_values": [
    "-1"
  ],
  "sample_label_values": [
    "-1"
  ],
  "overlap": {
    "train_and_test": 212,
    "test_and_sample": 25,
    "train_and_sample": 105,
    "test_only_vs_sample": 191,
    "sample_only_vs_test": 179,
    "test_only_examples": [
      "audio_0.wav",
      "audio_1.wav",
      "audio_10.wav",
      "audio_100.wav",
      "audio_101.wav",
      "audio_102.wav",
      "audio_103.wav",
      "audio_104.wav",
      "audio_105.wav",
      "audio_106.wav"
    ],
    "sample_only_examples": [
      "audio_1001.wav",
      "audio_1006.wav",
      "audio_1011.wav",
      "audio_1025.wav",
      "audio_1028.wav",
      "audio_1039.wav",
      "audio_1041.wav",
      "audio_1046.wav",
      "audio_1055.wav",
      "audio_1059.wav"
    ],
    "train_and_test_examples": [
      "audio_0.wav",
      "audio_1.wav",
      "audio_10.wav",
      "audio_100.wav",
      "audio_101.wav",
      "audio_102.wav",
      "audio_103.wav",
      "audio_104.wav",
      "audio_105.wav",
      "audio_106.wav"
    ]
  },
  "audio": {
    "train": {
      "files_on_disk": 769,
      "listed_in_csv": 769,
      "missing_audio": 0,
      "extra_audio": 0,
      "missing_examples": [],
      "extra_examples": [],
      "invalid_files": 0,
      "invalid_examples": [],
      "duration_sec": {
        "count": 769.0,
        "mean": 55.797,
        "std": 7.9,
        "min": 20.06,
        "25%": 54.251,
        "50%": 60.075,
        "75%": 60.075,
        "max": 61.04
      },
      "sample_rates": {
        "16000": 769
      },
      "channels": {
        "1": 769
      }
    },
    "test": {
      "files_on_disk": 216,
      "listed_in_csv": 216,
      "missing_audio": 0,
      "extra_audio": 0,
      "missing_examples": [],
      "extra_examples": [],
      "invalid_files": 0,
      "invalid_examples": [],
      "duration_sec": {
        "count": 216.0,
        "mean": 48.755,
        "std": 8.182,
        "min": 6.48,
        "25%": 45.055,
        "50%": 45.06,
        "75%": 60.011,
        "max": 61.035
      },
      "sample_rates": {
        "16000": 216
      },
      "channels": {
        "1": 216
      }
    },
    "sample_submission_vs_test_audio": {
      "present_in_test_dir": 25,
      "absent_from_test_dir": 179,
      "absent_examples": [
        "audio_1001.wav",
        "audio_1006.wav",
        "audio_1011.wav",
        "audio_1025.wav",
        "audio_1028.wav",
        "audio_1039.wav",
        "audio_1041.wav",
        "audio_1046.wav",
        "audio_1055.wav",
        "audio_1059.wav"
      ]
    }
  }
}
```