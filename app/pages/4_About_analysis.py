import fresh_modules

fresh_modules.refresh()

import streamlit as st

from workflow_ui import footer, setup

setup('About the analysis')
st.title('About the analysis')
st.markdown('''<div class="prose">

## Purpose

The analysis supports careful listening. It marks the parts of a recording that resemble crackles or wheezes, so that
the examiner can hear those parts again before signing. The examiner remains responsible for the documented finding,
and the analysis describes the sound without making a diagnosis.

## Model

The model (version cnn_3s_vote_v1) is a small convolutional neural network. It divides each recording, resampled to
4 kHz, into overlapping 3-second segments and classifies each segment as no adventitious sound, crackle, wheeze, or
both. The recording receives the pattern that wins the most segments. If that pattern covers less than 66% of the
segments, the analysis reports "No clear pattern" instead of naming one. The model was trained on adult recordings
from the training portions of two public datasets, ICBHI 2017 and HF_Lung V1.

## Evaluation

The model was tested once, after it was frozen, on adult recordings it had not seen during training. Performance is
reported as macro-F1, the average of the F1 scores of the four classes, where 1.0 means every recording was
classified correctly. The macro-F1 was 0.374 on the HF_Lung test set (1,939 recordings), 0.282 on the ICBHI test set
(300 recordings), and 0.176 on an external hospital dataset, Fraiwan 2021 (327 recordings), where accuracy was 17%,
below the 25% expected from guessing.

## Limitations

- On recordings from sources other than HF_Lung, wheezes were mostly reported as crackles and wheezes together.
- Reporting "No clear pattern" improved accuracy on the HF_Lung test set only; it did not help on the other two sets.
- Added background noise at 20 dB signal-to-noise ratio changed 35% to 85% of the results, so noisy rooms are a
  real risk.
- The model has not been tested on recordings from the Hanhong stethoscope used in this study.
- The automatic quality check detects recordings that are too short, silent or clipped. It cannot reliably detect
  speech or dominant room noise, so the examiner must listen before saving.

</div>''', unsafe_allow_html=True)
footer()
