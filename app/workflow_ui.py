"""Shared offline UI helpers; inference remains in stethoai.app_core.

Design direction (D-050): clinical, calm, high contrast, phone-first. One accent
(deep teal) for actions; finding colours always come with an icon and a word.
"""
from html import escape
from pathlib import Path
import hmac
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from stethoai.app_core import LABELS, MANUAL_LABELS, Store, load_frozen
from stethoai.drive_store import DriveClient, DriveError, DriveState, DriveStore

STEPS = (('Record', 'pages/1_Record.py'), ('Nurse answer', 'pages/2_Nurse_answer.py'),
         ('AI finding', 'pages/3_AI_finding.py'), ('Review', 'pages/4_Review_queue.py'))

# Finding look: (css class, icon glyph, plain-word headline, explanation).
FINDING_STYLE = {
    'N': ('normal', '✓', 'Normal', 'No crackle or wheeze was detected in this recording.'),
    'C': ('crackle', '◆', 'Crackle', 'Crackle sounds were detected. Re-listen to the marked parts.'),
    'W': ('wheeze', '≈', 'Wheeze', 'Wheeze sounds were detected.'),
    'B': ('both', '◆≈', 'Crackle and wheeze', 'Both crackle and wheeze sounds were detected.'),
    'U': ('uncertain', '?', 'Uncertain', 'The AI could not decide. Re-listen and rely on your own judgment.'),
    'Q': ('qcfail', '!', 'Recording not usable', 'The AI did not analyze this recording.'),
}

QC_WORDS = {
    'protocol_too_short': 'The recording is shorter than the required length',
    'clipping': 'The sound is too loud and distorted (clipping)',
    'silence': 'The recording is silent or nearly silent',
    'mono_silence': 'The recording is silent or nearly silent',
    'nonfinite_audio': 'The file contains damaged audio values',
    'unreadable_audio': 'The file could not be read as audio',
}

CSS = '''<style>
:root{
  --ink:#10303f; --muted:#405a68; --paper:#f5f7f6; --surface:#ffffff; --line:#cfd9dd;
  --primary:#0b5d6e; --primary-ink:#ffffff; --focus:#b35c00;
  --n-fg:#1d6640; --n-bg:#e5f1ea; --c-fg:#7f4507; --c-bg:#fbeedb;
  --w-fg:#1b578a; --w-bg:#e2edf8; --b-fg:#792d60; --b-bg:#f4e4ee;
  --u-fg:#3f4a55; --u-bg:#eceff1; --q-fg:#9c241b; --q-bg:#fbe5e2;
}
html,body,.stApp{background:var(--paper);color:var(--ink);
  font-family:"Segoe UI",system-ui,-apple-system,Roboto,"Helvetica Neue",Arial,sans-serif;}
.block-container{max-width:760px;padding:3.5rem 1rem 4rem;}
h1,h2,h3{color:var(--ink);font-family:inherit;letter-spacing:-0.01em;text-wrap:balance;}
h1{font-size:1.75rem!important;font-weight:700!important;padding:.25rem 0 .5rem!important;}
h2{font-size:1.25rem!important;font-weight:700!important;margin-top:1.5rem!important;}
h3{font-size:1.05rem!important;font-weight:700!important;margin-top:1.25rem!important;}
p,li,label{font-size:1rem;line-height:1.55;}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:var(--muted)!important;opacity:1!important;font-size:.92rem;}
[data-testid="stSidebar"]{background:#e8eef0;}
[data-testid="stHeader"]{background:transparent;}
/* Large tap targets */
.stButton button,.stDownloadButton button,[data-testid="stFormSubmitButton"] button{
  min-height:52px;width:100%;font-size:1.05rem;font-weight:600;border-radius:10px;}
.stButton button[kind="primary"]{background:var(--primary);border-color:var(--primary);color:var(--primary-ink);}
.stButton button:disabled{background:#e3e8ea;color:#5d6e77;border-color:#cfd9dd;}
[data-testid="stElementContainer"]:has([data-testid="stPageLink"]),[data-testid="stPageLink"],[data-testid="stPageLink"]>div{width:100%!important;}
[data-testid="stPageLink"] a{width:100%;box-sizing:border-box;min-height:52px;display:flex;align-items:center;padding:.5rem .9rem;
  border:1px solid var(--line);border-radius:10px;background:var(--surface);margin:.25rem 0;}
[data-testid="stPageLink"] a p{font-size:1rem;font-weight:600;color:var(--primary);}
[class*="st-key-nextlink_"] [data-testid="stPageLink"] a{background:var(--primary);border-color:var(--primary);}
[class*="st-key-nextlink_"] [data-testid="stPageLink"] a p{color:var(--primary-ink);}
input[type="text"],input[type="number"],[data-baseweb="select"]>div{min-height:48px;font-size:1rem!important;}
/* Answer choices as large tiles */
[data-testid="stRadio"] [role="radiogroup"]{gap:.5rem;flex-direction:row;flex-wrap:wrap;align-items:flex-start;}
[data-testid="stRadio"] [role="radiogroup"] label{min-height:52px;padding:.6rem .9rem;margin:0;
  border:1px solid var(--line);border-radius:10px;background:var(--surface);display:flex;align-items:center;}
[data-testid="stRadio"] [role="radiogroup"]>div{flex:1 1 140px;}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked){border:2px solid var(--primary);background:#e6f1f3;}
[data-testid="stRadio"] [role="radiogroup"] label p{font-size:1.05rem;font-weight:600;}
[data-testid="stFileUploaderDropzone"]{min-height:120px;border:2px dashed var(--line);background:var(--surface);}
:focus-visible,button:focus-visible,a:focus-visible,label:focus-within{outline:3px solid var(--focus)!important;outline-offset:2px;}
/* Step indicator */
.steps{display:flex;gap:4px;list-style:none;margin:0 0 1rem;padding:0;counter-reset:s;}
.steps li{flex:1;min-width:0;border-top:5px solid var(--line);padding:.4rem .1rem 0;color:var(--muted);
  font-size:.82rem;line-height:1.25;font-weight:600;}
.steps li .n{display:block;font-size:.75rem;font-weight:700;}
.steps li.done{border-top-color:#7fa8b2;}
.steps li.now{border-top-color:var(--primary);color:var(--ink);}
/* Finding card */
.finding{border-radius:14px;padding:1.1rem 1.1rem 1rem;margin:.5rem 0 1rem;border:2px solid;}
.finding .row{display:flex;gap:.9rem;align-items:center;}
.finding .icon{flex:none;width:56px;height:56px;border-radius:50%;display:grid;place-items:center;
  font-size:1.6rem;font-weight:700;color:#fff;}
.finding .kicker{font-size:.85rem;font-weight:600;margin:0;}
.finding .word{font-size:2rem;font-weight:800;line-height:1.1;margin:0;}
.finding .why{margin:.75rem 0 0;font-size:1rem;}
.finding .note{margin:.4rem 0 0;font-size:.9rem;}
.finding.normal{background:var(--n-bg);border-color:var(--n-fg);color:var(--n-fg);} .finding.normal .icon{background:var(--n-fg);}
.finding.crackle{background:var(--c-bg);border-color:var(--c-fg);color:var(--c-fg);} .finding.crackle .icon{background:var(--c-fg);}
.finding.wheeze{background:var(--w-bg);border-color:var(--w-fg);color:var(--w-fg);} .finding.wheeze .icon{background:var(--w-fg);}
.finding.both{background:var(--b-bg);border-color:var(--b-fg);color:var(--b-fg);} .finding.both .icon{background:var(--b-fg);font-size:1.2rem;}
.finding.uncertain{background:var(--u-bg);border:2px dashed var(--u-fg);color:var(--u-fg);} .finding.uncertain .icon{background:var(--u-fg);}
.finding.qcfail{background:var(--q-bg);border-color:var(--q-fg);color:var(--q-fg);} .finding.qcfail .icon{background:var(--q-fg);}
.finding .why,.finding .note{color:var(--ink);}
.chip{display:inline-block;padding:.15rem .6rem;border-radius:999px;background:#e6f1f3;color:var(--ink);
  font-weight:600;font-size:.9rem;border:1px solid var(--line);}
.empty{border:1px dashed var(--line);border-radius:12px;padding:1rem;background:var(--surface);}
.empty b{display:block;font-size:1.05rem;margin-bottom:.25rem;}
@media (max-width:480px){h1{font-size:1.5rem!important;} .finding .word{font-size:1.7rem;}
  .block-container{padding:3.5rem 1rem 4rem;}}
</style>'''


LOGIN_CSS = '''<style>
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="stExpandSidebarButton"]{display:none!important;}
.block-container{max-width:420px;padding-top:clamp(3rem,16vh,8rem);}
.login-name{margin:0;color:var(--primary);font-weight:700;font-size:1.05rem;}
[data-testid="stForm"]{border:0;padding:0;}
[data-testid="stForm"] input{font-size:1rem!important;}
</style>'''


@st.cache_resource
def get_model():
    # load_frozen verifies bytes before torch.load. Never read a manifest here.
    return load_frozen(model_path=ROOT / 'models/cnn_3s_vote_v1/model.pt',
                       config_path=ROOT / 'models/cnn_3s_vote_v1/config.json')


def is_hosted():
    """The deploy copy carries this marker; hosted copies refuse to run unprotected."""
    return (ROOT / '.hosted').exists()


def _secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name)
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return None


def require_password():
    """One shared team password (D-068). Called by every page through setup()."""
    expected = _secret('APP_PASSWORD')
    if not expected:
        if is_hosted():
            st.error('This site is not configured yet, so it stays closed.')
            st.stop()
        return
    if st.session_state.get('signed_in'):
        return
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)
    st.markdown('<p class="login-name">Lung sound study</p>', unsafe_allow_html=True)
    st.title('Team sign-in')
    with st.form('sign_in'):
        entered = st.text_input('Team password', type='password', autocomplete='current-password')
        submitted = st.form_submit_button('Sign in', type='primary')
    if submitted:
        if hmac.compare_digest(entered.encode('utf-8'), str(expected).encode('utf-8')):
            st.session_state['signed_in'] = True
            st.rerun()
        time.sleep(1)  # Slows password guessing.
        st.error('That password is not correct. Check it and try again.')
    st.caption('Research prototype. Not a medical device. It does not diagnose.')
    st.stop()


@st.cache_resource(show_spinner='Loading the study records from Google Drive...')
def _drive_state(url, key):
    return DriveState(DriveClient(url, key))


def open_store():
    url, key = _secret('DRIVE_URL'), _secret('DRIVE_KEY')
    if url and key:
        try:
            return DriveStore(_drive_state(url, key))
        except DriveError as error:
            st.error(f'The study records could not be loaded. Nothing was changed. {error}')
            st.button('Try again', type='primary')
            st.stop()
    if is_hosted():
        st.error('Cloud storage is not configured, so saving is turned off.')
        st.stop()
    return Store(os.environ.get('STETHOAI_LOG_ROOT', ROOT / 'logs/app'))


def setup(title, step=None):
    """Page chrome. step is 1-4 for workflow pages; it draws the step indicator."""
    st.set_page_config(page_title=f'{title} | Lung sound study', page_icon='🩺', layout='centered')
    st.markdown(CSS, unsafe_allow_html=True)
    require_password()
    store = open_store()
    st.sidebar.title('Lung sound study')
    st.sidebar.caption('Recorded acoustic findings. Training and research only.')
    st.sidebar.info('Use codes only. Never enter names or personal details.')
    if step:
        items = ''.join(
            f'<li class="{"now" if i == step else "done" if i < step else ""}"'
            f'{" aria-current=step" if i == step else ""}><span class="n">Step {i}</span>{name}</li>'
            for i, (name, _) in enumerate(STEPS, 1))
        st.markdown(f'<ol class="steps" aria-label="Workflow steps">{items}</ol>', unsafe_allow_html=True)
    st.title(title)
    return store


def next_link(page, label):
    """Primary 'continue' link styled as a full-width button."""
    with st.container(key=f'nextlink_{Path(page).stem}'):
        st.page_link(page, label=label, icon=':material/arrow_forward:')


def empty_state(title, body, page=None, label=None):
    st.markdown(f'<div class="empty" role="status"><b>{escape(title)}</b>{escape(body)}</div>',
                unsafe_allow_html=True)
    if page:
        st.page_link(page, label=label, icon=':material/arrow_forward:')


def site_note(store):
    st.caption('The site order is provisional until an expert confirms the site map.')


def qc_words(reason):
    parts = []
    for raw in filter(None, str(reason).split(';')):
        code, _, detail = raw.partition(':')
        words = QC_WORDS.get(code.strip(), raw.strip())
        parts.append(f'{words} ({detail.strip()})' if detail else words)
    parts = parts[:1] + [x[0].lower() + x[1:] for x in parts[1:]]
    return '; '.join(parts) or 'The recording failed the quality check'


def select_trial(store, *, locked=False, predicted=False, key='trial_select'):
    rows = store.table('rater_study')
    if locked:
        rows = [r for r in rows if r['manual_locked_at']]
    if predicted:
        rows = [r for r in rows if r['ai_prediction_id']]
    if not rows:
        if locked:
            empty_state('No locked answers yet',
                        ' Record a sample and lock your own answer first. The AI finding stays hidden until then.',
                        'pages/2_Nurse_answer.py', 'Go to Nurse answer')
        else:
            empty_state('No recordings yet', ' Start by recording one chest site.',
                        'pages/1_Record.py', 'Go to Record')
        return None
    ids = [r['trial_id'] for r in rows]
    active = st.session_state.get('active_trial')
    chosen = st.selectbox('Recording', ids, index=ids.index(active) if active in ids else len(ids) - 1, key=key,
                          help='Coded as case_site_trial_mode_rater.')
    st.session_state['active_trial'] = chosen
    return next(r for r in rows if r['trial_id'] == chosen)


def replay(path, label='Listen to the recording'):
    st.markdown(f'**{escape(label)}**')
    if path and Path(path).exists():
        st.audio(Path(path).read_bytes(), format='audio/wav')
    else:
        st.caption('No earlier recording is available.')


def finding_card(result):
    finding = result['finding']
    cls, icon, word, why = FINDING_STYLE[finding]
    notes = []
    if finding == 'Q':
        why = f"{qc_words(result['qc_reason'])}. Re-record this site with a new trial number."
    elif finding == 'U':
        notes.append(f"The strongest answer had {result['top_score']:.0%} of window votes; "
                     f"the AI needs at least {result['threshold']:.0%} to give a finding.")
    else:
        notes.append(f"{result['top_score']:.0%} of the 3-second windows voted for this finding.")
    if result.get('relisten') and finding != 'C':
        notes.append('Crackle votes were present: re-listen to the marked parts.')
    if finding != 'Q':
        notes.append('This describes the sound in this recording only. It is not a diagnosis.')
    kicker = 'Quality check' if finding == 'Q' else 'AI acoustic finding'
    st.markdown(
        f'<div class="finding {cls}" role="status"><div class="row">'
        f'<div class="icon" aria-hidden="true">{icon}</div>'
        f'<div><p class="kicker">{kicker}</p><p class="word">{escape(word)}</p></div></div>'
        f'<p class="why">{escape(why)}</p>'
        + ''.join(f'<p class="note">{escape(n)}</p>' for n in notes) + '</div>',
        unsafe_allow_html=True)


def show_finding(result, path):
    finding_card(result)
    if result['finding'] == 'Q':
        return
    replay(path)
    with st.expander('Votes and time markers'):
        votes = pd.DataFrame({'Acoustic class': [LABELS[c] for c in ('N', 'C', 'W', 'B')],
                              'Vote fraction': [result[f'v{i}'] for i in range(4)]})
        st.bar_chart(votes, x='Acoustic class', y='Vote fraction', color='#0b5d6e', sort=False, height=220)
        st.caption('Bars show the share of window votes, not a calibrated clinical confidence.')
        windows = result['window_outputs']
        if windows:
            values, sr = sf.read(path, dtype='float32', always_2d=True)
            mono = values.mean(axis=1)
            stride = max(1, len(mono) // 16000)
            fig, wave = plt.subplots(figsize=(8, 2.2))
            wave.plot(np.arange(0, len(mono), stride) / sr, mono[::stride], color='#315b70', linewidth=.6)
            for w in windows:
                wave.axvline(w['start_s'], color='#849baa', alpha=.35, linewidth=.7)
                if int(np.argmax(w['probabilities'])) in (1, 3):
                    wave.axvspan(w['start_s'], w['end_s'], color='#b46b14', alpha=.2, hatch='//')
            wave.set(xlabel='Time (seconds)', ylabel='Amplitude', xlim=(0, len(mono) / sr))
            fig.tight_layout()
            st.markdown('**Where in the recording**')
            st.pyplot(fig)
            plt.close(fig)
            st.caption('Grey lines mark the 3-second analysis windows. Hatched orange marks windows voting '
                       'Crackle or Both. Windows overlap by 1.5 s; the last one may be padded.')
            st.line_chart(pd.DataFrame({LABELS[code]: [w['probabilities'][i] for w in windows]
                                        for i, code in enumerate(('N', 'C', 'W', 'B'))},
                                       index=pd.Index([(w['start_s'] + w['end_s']) / 2 for w in windows],
                                                      name='Window centre (seconds)')),
                          x_label='Window centre (seconds)', y_label='Window probability',
                          color=['#1d6640', '#b46b14', '#1b578a', '#792d60'], height=240)
            st.dataframe(pd.DataFrame([{'Start (s)': w['start_s'], 'End (s)': w['end_s'],
                                       **{LABELS[c]: round(w['probabilities'][i], 3)
                                          for i, c in enumerate(('N', 'C', 'W', 'B'))}}
                                      for w in windows]), hide_index=True)


def answer_select(label, *, key):
    """Large tap tiles; returns a MANUAL_LABELS code or None."""
    return st.radio(label, list(MANUAL_LABELS), format_func=MANUAL_LABELS.get, index=None, key=key,
                    horizontal=True)
