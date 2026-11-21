# CNN 3 s vote v2

Status: **frozen 22 November 2026 under D-113; adopted the same day under D-114** as the model used by the team tool
and website. v1 (`models/cnn_3s_vote_v1/`) keeps its locked results; analyses it made keep their v1 tag.
Results: `reports/model_v2/results.md`.

## Intended output
Same contract as v1. A standardized recorded adult lung-sound sample is classified as Normal, Crackle, Wheeze or Both.
Three-second windows (1.5 s hop) are combined by vote; if the winning vote fraction is below 0.62 the output is
Uncertain. An AI second opinion about an acoustic finding, never a diagnosis.

## Frozen identity
- Model file: `model.pt`, SHA-256 `943db6f8c1c9967f1434aba8e1b67fbe8f9b73d21ea627f341ad276cb9c07bc6`
- Config: `config.json` (input, augmentation, training, threshold policy, provenance hashes)
- Window / hop: 3 s / 1.5 s at 4 kHz; aggregation: vote; abstain threshold 0.62 (plan section 6, D-096)
- Loader: `stethoai.cnn_v2.load_v2`, prediction `stethoai.cnn_v2.predict_signal`

## How it was built
- Architecture: v1 SmallCNN, unchanged.
- Input (`features_v2`, D-094): log-mel window; a short window is filled by repeating its own audio; log power is
  floored 80 dB below the window peak. The input is exactly invariant to recording gain.
- Training data: the v1 adult pool, 7,878 recordings from 178 patient groups (ICBHI train adults, HF_Lung train).
  No test, external, manikin or Hanhong recording was used for training, threshold or selection.
- Arm C of three pre-registered arms (`reports/model_v2/evaluation_plan.md`, D-095): square-root source-balanced
  sampling, waveform augmentation (EQ, clipping, coloured noise 3-30 dB SNR, +/-0.5 s shift), wheeze weight 1.5.
- Chosen on training out-of-fold source-macro F1 (C 0.390, A 0.383, B 0.380; v1 reference 0.378, not directly
  comparable), tie broken by agreement under 20 dB noise (C 0.807, A 0.686). Final refit on all training recordings
  for 6 epochs, seed 42.

## Evidence (kept in separate groups)
1. Training out-of-fold (all sources): macro-F1 0.398 at threshold 0, 0.367 at 0.62, coverage 0.797.
2. HLS-CMDS manikin sanity check (161 files, not clinical validation): macro-F1 t=0 0.232 vs v1 0.285 (difference not
   established); v2 calls most files Wheeze. Output fully stable under gain change.
3. Test-informed old sets (not official): macro-F1 t=0 ICBHI 0.271 (v1 0.282), HF_Lung 0.383 (v1 0.374), Fraiwan 0.312
   (v1 0.176).
No fresh human external test has been run, so no external performance claim is made.

## Scope and limits
- Adults; lung sounds only; acoustic finding, not diagnosis; research prototype, not a medical device.
- Over-calls Wheeze on an unseen recording chain and rarely outputs Both.
- Not tested on the Hanhong stethoscope, on expert-rated recordings or for repeat-recording repeatability.
- Improves when: a fresh adult human set (FABS, on request) is scored for v1 and v2 together, and Hanhong recordings
  with expert labels exist.
