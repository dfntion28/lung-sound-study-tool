from workflow_ui import is_hosted, setup, site_note
import streamlit as st

store = setup('Listen first. Then compare with the AI.')
st.write('Record one chest site, lock your own answer, then see the AI acoustic finding as a second opinion. '
         'For training and research with adults only. The AI never gives a diagnosis.')
st.markdown('<ol class="steps" aria-label="Workflow steps"><li class="now"><span class="n">Step 1</span>Record</li>'
            '<li><span class="n">Step 2</span>Nurse answer</li><li><span class="n">Step 3</span>AI finding</li>'
            '<li><span class="n">Step 4</span>Review</li></ol>', unsafe_allow_html=True)
with st.container(key='nextlink_home'):
    st.page_link('pages/1_Record.py', label='Start: record a site', icon=':material/mic:')
st.caption('Your answer is locked before the AI finding appears, so the AI cannot change your first answer.')
site_note(store)
st.subheader('Other tools')
st.page_link('pages/4_Review_queue.py', label='Review disagreements', icon=':material/rule:')
st.page_link('pages/5_Analytics.py', label='Study results and CSV exports', icon=':material/table_chart:')
st.page_link('pages/6_Model_card.py', label='What the AI was tested on and its limits', icon=':material/info:')
if is_hosted():
    st.caption('Recordings and study records are saved to the team\'s private Google Drive folder.')
else:
    with st.expander('Open the app on a phone (same Wi-Fi)'):
        st.code(r'.venv\Scripts\python.exe -m streamlit run app/Home.py --server.address 0.0.0.0', language='powershell')
        st.write('Keep the window running. On a phone on the same Wi-Fi, open the Network URL it prints, '
                 'usually http://<PC address>:8501. Recordings and processing stay on this PC; no internet is needed.')
        st.caption('Run one server per log folder. Set STETHOAI_LOG_ROOT before starting to choose another local log folder.')
