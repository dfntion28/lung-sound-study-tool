import streamlit as st

from workflow_ui import footer, setup

setup('About the analysis')
st.title('About the analysis')
st.markdown('''<div class="prose">

The analysis supports careful listening. It marks parts of a recording that resemble crackles or wheezes so the
examiner can hear them again. The examiner remains responsible for the documented finding.

## How it was tested

Model cnn_3s_vote_v1 is a small neural network using 3-second segments of 4 kHz sound. A pattern is named only when it
covers at least 66% of segments. It was trained on the adult training portions of ICBHI 2017 and HF_Lung V1.

Locked one-time evaluation on adult recordings, macro-F1: ICBHI test 0.282, HF_Lung test 0.374, and Fraiwan 2021
external 0.176 (accuracy 0.17).

## Known limits

- On recordings from other sources, wheezes were often reported as both crackles and wheezes.
- Holding back a result helped on only one of the three test sets (HF_Lung).
- Added background noise makes results unstable.
- It has not been tested on the Hanhong stethoscope.
- The quality check covers duration, silence and clipping. It cannot reliably identify speech or dominant noise.
- It does not provide a diagnosis.

</div>''', unsafe_allow_html=True)
footer()
