import fresh_modules

fresh_modules.refresh()

import streamlit as st

from workflow_ui import heading, footer, setup

setup('About the analysis')
heading('About the analysis')
st.markdown('''<div class="prose">

## Purpose

The analysis supports careful listening. It marks the parts of a recording that resemble crackles or wheezes, so that
the examiner can hear those parts again before signing. The examiner remains responsible for the documented finding,
and the analysis describes the sound without making a diagnosis.

## Model

The model (version cnn_3s_vote_v2, adopted on 22 November 2026) is a small convolutional neural network. It divides
each recording, resampled to 4 kHz, into overlapping 3-second segments and classifies each segment as no adventitious
sound, crackle, wheeze, or both. The recording receives the pattern that wins the most segments. If that pattern covers
less than 62% of the segments, the analysis reports "No clear pattern" instead of naming one. The result does not
change with recording volume unless the louder recording is clipped. The model was trained on adult recordings from the training portions of two public
datasets, ICBHI 2017 and HF_Lung V1, with added noise and filtering so that it tolerates differences between recordings.

## Evaluation

The first version of the model was tested once, after it was frozen, on adult recordings it had not seen. Performance
is reported as macro-F1, the average of the F1 scores of the four classes, where 1.0 means every recording was
classified correctly. Its macro-F1 was 0.374 on the HF_Lung test set (1,939 recordings), 0.282 on the ICBHI test set
(300 recordings), and 0.176 on an external hospital dataset, Fraiwan 2021 (327 recordings).

The current version was designed after those results were known, so its scores on the same sets are not an independent
test: 0.383, 0.271 and 0.312 respectively. It has not yet been tested on recordings never used in this project.

## Limitations

- On a teaching manikin, a recording source the model had never heard, the current version reported most recordings as
  wheeze, including most normal and crackle recordings.
- The current version rarely reports crackles and wheezes together.
- Added background noise at 20 dB signal-to-noise ratio changed 41% to 63% of the results, so noisy rooms are a
  real risk.
- The model has not been tested on recordings from the Hanhong stethoscope used in this study.
- The chest-site positions follow the recording guide and remain provisional until the clinical adviser
  confirms them.
- The automatic quality check detects recordings that are too short, silent or clipped. It cannot reliably detect
  speech or dominant room noise, so the examiner must listen before saving.

</div>''', unsafe_allow_html=True)
footer()
