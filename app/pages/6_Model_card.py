import streamlit as st
from workflow_ui import setup

setup('What the AI was tested on')
st.subheader('Model: cnn_3s_vote_v1')
st.write('Small CNN on 4 kHz log-mel, 3 s windows, vote aggregation, Uncertain below 0.66; trained on adult ICBHI 2017 + HF_Lung V1 training splits only.')
st.subheader('Locked one-time evaluation')
st.write('Adult recordings, macro-F1: ICBHI test 0.282, HF_Lung test 0.374, Fraiwan 2021 external 0.176 (accuracy 0.17).')
st.subheader('Known limits')
st.markdown('''
- On recordings from other sources, wheezes were often reported as Both.
- Holding back an answer (Uncertain) helped on only one of the three test sets (HF_Lung).
- Added background noise makes results unstable.
- Not tested on the Hanhong stethoscope.
- Not a diagnosis; a second opinion for training and research only.
''')
st.caption('Acoustic findings describe the analyzed recording. Hanhong export format and the provisional site map still need confirmation. Built-in QC checks duration, silence and clipping; it cannot reliably identify speech or dominant noise.')
