# CNN 3 s vote v1

Status: **frozen 7 November 2026 under D-045**. This is the adult lung-sound acoustic-finding model for the next locked-evaluation gate.

## Intended output

The model classifies a standardized recorded lung-sound sample as Normal, Crackle, Wheeze or Both. Three-second window predictions are combined by vote. If the winning vote fraction is below 0.66, the output is Uncertain. This is an AI second opinion about an acoustic finding, not a diagnosis.

## Frozen identity

- Model file: `model.pt`
- SHA-256: `bc58005e125291f862cc357ee45bb492cdaeea405a666c687f67581c65578c64`
- Window / hop: 3 s / 1.5 s
- Aggregation: vote
- Abstain threshold: 0.66, accepted by the project lead under D-039/D-045
- Final all-adult refit: 5 epochs

## Training evidence

The adult-only training pool contains 7,878 recordings from 178 patient groups after removing 46 pediatric ICBHI recordings. Train-only grouped out-of-fold macro-F1 is 0.4107 and accuracy at full coverage is 0.4326. At the accepted uncertainty threshold, coverage is 75.2% and retained accuracy is 45.2%.

These values were reused to select and configure the model, so they are optimistic and are not locked-test, device or clinical validation. Per-class and per-dataset results are in `reports/phase4_adult_only_cnn_3s_vote.json`; model selection is in `reports/phase4_comparison.md`.

## Known limitations

- Crackle, wheeze and both-sound performance is modest; the Both class has the lowest adult-pool F1.
- ICBHI adult-only performance is lower than HF_Lung performance.
- The accepted uncertainty rule did not reach its five-point retained-accuracy target at the 70% coverage floor; it used the declared fallback.
- No locked public test, Fraiwan external test, Hanhong recording, repeatability or clinical evaluation has run.
- Use only inside the recorded standardized auscultation workflow with human review and visible uncertainty.
