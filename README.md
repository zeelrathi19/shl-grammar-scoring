# Grammar scoring for spoken English

SHL Hiring Assessment 2026 (Kaggle). The task: predict a 1–5 grammar score for short spoken-English answers (769 training clips, 216 test clips, mostly 45–60 s each). Everything runs on CPU. All pretrained weights come from GitHub releases, so Hugging Face isn't needed.

## Results

| | RMSE | Pearson |
|---|---|---|
| 5×3 repeated CV, all 732 scored clips | **0.468** | 0.887 |
| CV, test-like batch (train ids < 200) | 0.510 | 0.795 |
| CV, whole prompt topics held out | 0.488 | 0.877 |
| Training fit (in-sample) | 0.089 | 0.998 |
| Predicting the mean | 1.014 | – |

The training fit is far below the CV numbers because the SVRs nearly memorise their training data. Use the CV rows to judge how well the model generalises.

## How it works

```
audio ─┬─ Parakeet-TDT 0.6B ASR ─► transcript, word timings, token confidence
       │                            ├─ 76 features: pauses, speech rate, disfluency, lexical, syntax, prosody
       │                            └─ CoLA acceptability per sentence (RoBERTa fine-tuned on CoLA)
       │                                                               └─► LightGBM ──┐
       ├─ Parakeet encoder, layers 6 + 18, mean-pooled ────────────────► SVR ───────┤
       └─ Whisper-small.en encoder, layers 8–10, mean + std ───────────► SVR ───────┤
                                       45-second-batch flag ───────────────────────┴─► ridge stack ─► clip to [1, 5]
```

## What mattered

- **The Parakeet encoder.** Pooled at one mid and one upper layer, it's the strongest single block (0.481 CV RMSE alone) and the least hurt when whole prompt topics are held out. Whisper adds a little on top.
- **Looking at the data before modelling.**
  - The 37 clips scored 0.0 are a separate, off-rubric batch, so they're dropped.
  - The test set resembles the 45-second recording batch (train ids 0–199), which every model over-scores by 0.12–0.18, so the stack gets a flag for that batch.
  - `sample_submission.csv` is stale, so the submission follows `test.csv`.
- **Transcript features overlap with the encoders.** They reach about 0.67 RMSE alone, but add almost nothing once the encoder blocks are in.

Tried and dropped because they didn't improve the test-like CV subset: interaction terms, non-negative or LightGBM stackers, isotonic or linear stretching, kernel ridge, PCA whitening, sample weighting, a single SVR over all blocks, extra encoder layers.

## Running it

**Kaggle:** attach the competition data, turn Internet on (for the GitHub model downloads), then Run All on a CPU session. The first run takes a few hours: transcription, two encoders and a CoLA fine-tune. Every heavy step caches to `cache/`, so later runs take minutes.

**Locally:**

```bash
pip install -r requirements.txt
# competition data at ./data/Dataset_Final/{train/, test/, train.csv, test.csv}
jupyter nbconvert --to notebook --execute shl_grammar_scoring.ipynb
```

Outputs: `submission.csv`, `results.json` (metrics and stack weights) and `oof_predictions.csv`.

The committed notebook outputs come from a run with the cache already built, so the extraction cells report 0 s. Transcript excerpts were removed from the outputs, and competition audio and labels aren't included.

## Files

| | |
|---|---|
| `shl_grammar_scoring.ipynb` | Full pipeline: data checks, features, models, ablations, error analysis |
| `submission.csv` | Test predictions (216 rows, `test.csv` order) |
| `results.json` | CV and training metrics, stack weights |
| `requirements.txt` | Pinned versions the notebook was run with (Python 3.13) |

## Authorship

Built with a collaborator and with substantial AI assistance (Claude), including the code and this write-up.
