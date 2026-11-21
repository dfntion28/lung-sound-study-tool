"""Shared UI for the auscultation record (D-069). Inference stays in stethoai.app_core.

Direction: docs/design/TOOL_DIRECTION.md. An anatomy chart, phone first.
Findings are always written in words, never colour alone.
"""
from html import escape
from pathlib import Path
import hmac
import os
import re
import sys
import time

import numpy as np
import soundfile as sf
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from stethoai.app_core import LABELS, MANUAL_LABELS, Store, load_frozen
from stethoai.drive_store import DriveClient, DriveError, DriveState, DriveStore

PRODUCT = 'Lung Sound Record'
NOTICE = 'Research prototype. Not a medical device. It does not diagnose.'
SITE_NAMES = {'ARU': 'Right upper', 'ARL': 'Right lower', 'ALU': 'Left upper', 'ALL': 'Left lower',
              'PRU': 'Right upper', 'PRL': 'Right lower', 'PLU': 'Left upper', 'PLL': 'Left lower'}
# Same order as Section 2 of the student recording guide.
SITE_GROUPS = (('Anterior chest', ('ARU', 'ALU', 'ARL', 'ALL')),
               ('Posterior chest', ('PRU', 'PLU', 'PRL', 'PLL')))
SITE_ORDER = tuple(code for _, codes in SITE_GROUPS for code in codes)
MODE_NAMES = {'BELL': 'Bell', 'DIA': 'Diaphragm'}
FINDINGS = {'N': 'No adventitious sounds', 'C': 'Crackles', 'W': 'Wheezes',
            'B': 'Crackles and wheezes', 'U': 'Unable to determine'}
PATTERN_NAMES = {'C': 'Crackle-like', 'W': 'Wheeze-like', 'B': 'Crackle- and wheeze-like'}
SEGMENT_FILL = {'C': '#a4560c', 'W': '#1b578a', 'B': '#792d60'}

# Analysis display: (css class, headline, guidance).
FINDING_STYLE = {
    'N': ('normal', 'No crackle or wheeze pattern', 'No times were marked.'),
    'C': ('crackle', 'Crackle-like sound', 'Listen again at the marked times before you sign.'),
    'W': ('wheeze', 'Wheeze-like sound', 'Listen again at the marked times before you sign.'),
    'B': ('both', 'Crackle-like and wheeze-like sound', 'Listen again at the marked times before you sign.'),
    'U': ('uncertain', 'No clear pattern', 'The analysis could not settle on a pattern. Go by what you heard.'),
    'Q': ('qcfail', 'Recording not usable', 'The analysis did not run on this recording.'),
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
  --ink:#1a2430; --muted:#4d5a66; --ground:#eef1f3; --sheet:#fff; --rule:#cdd5db; --rule-strong:#84919c;
  --tissue:#f3d9d3; --tissue-line:#8f4a45; --skin:#f7f4f2; --skin-line:#9aa5ae; --rib:#d8d3cf; --heart:#e7b9ae; --airway:#7b8894; --bone:#aab2b9; --bone-fill:#ece8e4; --bone-line:#8d98a2; --action:#23427a; --action-hover:#182f59; --on-action:#fff; --selected:#e1e8f4; --focus:#b3540a;
  --n:#1d6640; --c:#7f4507; --w:#1b578a; --b:#792d60; --u:#3f4a55; --q:#a3231a; --problem:#a3231a; --q-bg:#fbe9e6;
  --hover:#e4e9ed; --disabled:#e3e7ea; --on-disabled:#56636e; --track:#f3f5f7; --on-problem:#7c1c15; --radius:6px;
}
html,body,.stApp{background:var(--ground);color:var(--ink);
  font-family:"Segoe UI",system-ui,-apple-system,Roboto,"Helvetica Neue",Arial,sans-serif;}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="stExpandSidebarButton"],
[data-testid="stHeader"],[data-testid="stDecoration"]{display:none!important;}
.block-container{max-width:1180px;padding:0 1rem 3rem!important;}
h1,h2,h3{color:var(--ink);font-family:inherit;letter-spacing:0;text-wrap:balance;}
h1{font-size:1.5rem!important;font-weight:650!important;padding:1rem 0 .15rem!important;}
h2{font-size:1.15rem!important;font-weight:650!important;padding:.9rem 0 .25rem!important;}
h3{font-size:1rem!important;font-weight:650!important;padding:.6rem 0 .2rem!important;}
p,li,label{font-size:1rem;line-height:1.5;}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:var(--muted)!important;opacity:1!important;font-size:.92rem;}
/* Masthead */
.st-key-masthead{border-bottom:1px solid var(--rule-strong);padding:.75rem 0 .35rem;gap:.35rem;}
.brand{margin:0;padding:0 .65rem 0 0;font-weight:700;font-size:1rem;line-height:1.4;color:var(--ink);}
.st-key-masthead [data-testid="stMarkdownContainer"] p.brand{margin-bottom:0;}
.st-key-masthead [data-testid="stMarkdown"],.st-key-masthead [data-testid="stMarkdownContainer"]{margin-bottom:0!important;height:auto!important;}
.st-key-masthead [data-testid="stElementContainer"]:has(.brand){height:auto!important;min-height:1.5rem;}
.st-key-nav{gap:.1rem!important;flex-wrap:nowrap;overflow-x:auto;scrollbar-width:none;}
.st-key-nav [data-testid="stPageLink"] a{min-height:44px;padding:.25rem .65rem;border-radius:var(--radius);white-space:nowrap;}
.st-key-nav [data-testid="stPageLink"] a p{font-size:.92rem;font-weight:550;color:var(--muted);}
.st-key-nav [data-testid="stPageLink"] a:hover{background:var(--hover);}
.st-key-nav [data-testid="stPageLink"] a[aria-current="page"]{background:var(--selected);}
.st-key-nav [data-testid="stPageLink"] a[aria-current="page"] p{color:var(--action);font-weight:650;}
/* Controls */
.stButton button,.stDownloadButton button,[data-testid="stFormSubmitButton"] button{
  min-height:48px;width:100%;font-size:1rem;font-weight:600;border-radius:var(--radius);border:1px solid var(--rule-strong);
  background:var(--sheet);color:var(--ink);box-shadow:none;transition:background-color .15s ease-out;touch-action:manipulation;}
.stButton button:hover,.stDownloadButton button:hover{background:#eef2ef;border-color:var(--action);color:var(--action);}
.stButton button[kind="primary"],[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"],
[data-testid="stFormSubmitButton"] button{background:var(--action);border-color:var(--action);color:var(--on-action);}
.stButton button[kind="primary"]:hover,[data-testid="stFormSubmitButton"] button:hover{background:var(--action-hover);color:var(--on-action);}
.stButton button:disabled{background:var(--disabled);color:var(--on-disabled);border-color:var(--rule);}
input[type="text"],input[type="password"],input[type="number"],[data-baseweb="select"]>div{min-height:46px;font-size:1rem!important;border-radius:var(--radius)!important;}
[data-testid="stTextInputRootElement"],[data-baseweb="select"]>div,[data-testid="stNumberInputContainer"]{background:var(--sheet)!important;border:1px solid var(--rule-strong)!important;border-radius:var(--radius)!important;}
[data-testid="stTextInputRootElement"] *,[data-baseweb="base-input"],[data-baseweb="base-input"] input,[data-testid="stNumberInputContainer"] input{background:var(--sheet)!important;}
[data-testid="stForm"]{border:0;padding:0;}
[data-testid="stRadio"] [role="radiogroup"]{gap:0;flex-direction:column;align-items:stretch!important;border-top:1px solid var(--rule);}
[data-testid="stRadio"] [role="radiogroup"] label{min-height:48px;margin:0;padding:.45rem .5rem;width:100%;border-bottom:1px solid var(--rule);display:flex;align-items:center;}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked){background:var(--selected);}
[data-testid="stRadio"] [role="radiogroup"] label p{font-weight:550;}
.st-key-mode_row [role="radiogroup"]{flex-direction:row;align-items:center!important;border-top:0;gap:.5rem;}
.st-key-mode_row [role="radiogroup"] label{width:auto!important;flex:1 1 auto;min-width:max-content;border:1px solid var(--rule-strong);border-radius:var(--radius);}
[data-testid="stFileUploaderDropzone"]{min-height:64px;border:1px dashed var(--rule-strong);border-radius:var(--radius);background:var(--sheet);}
:focus-visible,button:focus-visible,a:focus-visible,label:focus-within{outline:3px solid var(--focus)!important;outline-offset:1px;}
[data-testid="stHeaderActionElements"]{display:none!important;}
[data-testid="stElementContainer"]:has([data-testid="stRadio"]),[data-testid="stRadio"],[data-testid="stRadio"]>div,[data-testid="stRadio"] [role="radiogroup"]{width:100%!important;}
[data-testid="stRadio"] [role="radiogroup"]>label,[data-testid="stRadio"] [role="radiogroup"]>div{width:100%!important;max-width:none!important;}
.st-key-mode_row [role="radiogroup"]>label,.st-key-mode_row [role="radiogroup"]>div{width:auto!important;flex:1 1 auto;min-width:max-content;}
.st-key-mode_row [role="radiogroup"] label p{white-space:nowrap;}
[class*="st-key-row_"],[class*="st-key-row_"] [data-testid="stElementContainer"],[class*="st-key-row_"] .stButton,[class*="st-key-row_"] [data-testid="stTooltipHoverTarget"]{width:100%!important;}
[class*="st-key-row_"] .stButton button{width:100%!important;min-height:52px;}
/* Flowsheet */
.st-key-sheet [data-testid="stHorizontalBlock"]{align-items:flex-start;}
[class*="st-key-sign_"],[class*="st-key-save_"],[class*="st-key-again_"],[class*="st-key-next_"],
[class*="st-key-sign_"] .stButton,[class*="st-key-save_"] .stButton,[class*="st-key-again_"] .stButton,[class*="st-key-next_"] .stButton,
[data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]),[data-testid="stFormSubmitButton"]{width:100%!important;}
/* Chest chart */
.st-key-charts{gap:.75rem!important;flex-wrap:nowrap;align-items:flex-start;}
.st-key-chart_front,.st-key-chart_back{position:relative;flex:1 1 0;min-width:0;aspect-ratio:400/470;display:block!important;background:var(--sheet);border:1px solid var(--rule);border-radius:var(--radius);overflow:hidden;}
.st-key-chart_front [data-testid="stElementContainer"]:has(.plate),.st-key-chart_back [data-testid="stElementContainer"]:has(.plate){position:absolute;inset:0;margin:0;}
.st-key-charts>[data-testid="stLayoutWrapper"]{flex:1 1 0!important;min-width:0;width:auto!important;}
.st-key-chart_front,.st-key-chart_back{width:100%!important;height:0!important;padding:0 0 117.5% 0!important;box-sizing:content-box;aspect-ratio:auto;}
.st-key-chart_front [data-testid="stMarkdown"],.st-key-chart_back [data-testid="stMarkdown"],.st-key-chart_front [data-testid="stMarkdownContainer"],.st-key-chart_back [data-testid="stMarkdownContainer"]{height:100%!important;width:100%!important;margin:0!important;}
.plate{display:block;position:absolute;inset:0;width:100%;height:100%;}
.plate .body{fill:var(--skin);stroke:var(--skin-line);stroke-width:1.5;}
.plate .ribs{fill:none;stroke:var(--rib);stroke-width:5;stroke-linecap:round;}
.plate .lung{fill:var(--tissue);fill-opacity:.92;stroke:var(--tissue-line);stroke-width:2;stroke-linejoin:round;}
.plate .fissure{fill:none;stroke:var(--tissue-line);stroke-width:1.4;stroke-dasharray:5 4;}
.plate .heart{fill:var(--heart);fill-opacity:.55;stroke:var(--tissue-line);stroke-width:1;}
.plate .airway{fill:none;stroke:var(--airway);stroke-width:6;stroke-linecap:round;}
.plate .bone{fill:none;stroke:var(--bone);stroke-width:7;stroke-linecap:round;}
.plate .spine{fill:none;stroke:var(--airway);stroke-width:2.5;stroke-linecap:round;}
.plate .sternum,.plate .blade{fill:var(--bone-fill);fill-opacity:.82;stroke:var(--bone-line);stroke-width:1.2;}
.plate text{font-family:inherit;fill:var(--ink);paint-order:stroke;stroke:#fff;stroke-width:4px;stroke-linejoin:round;}
.plate .site{font-size:15.5px;font-weight:600;} .plate .side{font-size:18px;fill:var(--muted);}
[class*="st-key-pick_"]{position:absolute!important;width:48px!important;height:48px!important;transform:translate(-50%,-50%);z-index:2;margin:0!important;}
[class*="st-key-pick_"] .stButton,[class*="st-key-pick_"] [data-testid="stTooltipHoverTarget"]{width:48px!important;height:48px!important;}
[class*="st-key-pick_"] button,[class*="st-key-pick_"] button:hover,[class*="st-key-pick_"] button[kind="primary"]{position:relative;width:48px!important;min-width:48px;min-height:48px!important;height:48px;padding:0!important;border:0!important;border-radius:50%!important;background:transparent!important;cursor:pointer;}
[class*="st-key-pick_"] button>div{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap;}
[class*="st-key-pick_"] button::before{content:"";position:absolute;left:11px;top:11px;width:26px;height:26px;box-sizing:border-box;display:grid;place-items:center;font-size:15px;font-weight:800;line-height:1;border-radius:50%;border:2.5px solid var(--ink);background:#fff;transition:transform .15s ease-out;}
[class*="st-key-pick_"] button:hover::before{transform:scale(1.12);}
[class*="st-key-pick_"] button::after{content:"";position:absolute;}
[class*="st-key-pick_"] button[kind="primary"]::before{box-shadow:0 0 0 3px #fff,0 0 0 6px var(--action);}
[data-testid="stFileUploaderDropzoneInstructions"],[data-testid="stFileUploaderDropzone"] button svg,[data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"],[data-testid="stFileUploaderDropzone"] button span:has(>svg),[data-testid="stFileUploaderDropzone"] button span:has(>[data-testid="stIconMaterial"]){display:none!important;}
.st-key-view_switch{gap:0!important;flex-wrap:nowrap;margin-bottom:.15rem;}
.st-key-view_switch .stButton{flex:1 1 0;} .st-key-view_switch [data-testid="stElementContainer"]{flex:1 1 0;width:auto!important;}
.st-key-view_front button{border-radius:var(--radius) 0 0 var(--radius);} .st-key-view_back button{border-radius:0 var(--radius) var(--radius) 0;border-left:0;}
.legend{display:flex;flex-wrap:wrap;gap:.3rem 1rem;color:var(--muted);font-size:.92rem;}
.legend span{display:inline-flex;align-items:center;gap:.4rem;}
.mk{display:inline-grid;place-items:center;width:18px;height:18px;box-sizing:border-box;border-radius:50%;border:2px solid var(--ink);background:#fff;font-size:11px;font-weight:800;font-style:normal;line-height:1;}
.mk.half{background:linear-gradient(90deg,var(--ink) 50%,#fff 50%);} .mk.done{background:var(--ink);color:#fff;} .mk.again{border-color:var(--problem);color:var(--problem);}
.place{margin:.1rem 0 .7rem;line-height:1.5;}
[class*="st-key-row_"] .stButton button{justify-content:flex-start;text-align:left;border:0;border-bottom:1px solid var(--rule);border-radius:0;background:transparent;font-weight:500;padding:.3rem .4rem;}
[class*="st-key-row_"] .stButton button>div,[class*="st-key-row_"] .stButton button span{justify-content:flex-start!important;}
[class*="st-key-row_"] .stButton button>div{width:100%;}
[class*="st-key-row_"] .stButton button p{text-align:left;}
[class*="st-key-row_"] .stButton button:hover{background:var(--hover);color:var(--ink);}
[class*="st-key-row_"] .stButton button[kind="primary"]{background:var(--selected);color:var(--action);font-weight:650;border-bottom-color:var(--action);}
.st-key-work{background:var(--sheet);border:1px solid var(--rule);border-radius:var(--radius);padding:.9rem 1rem 1.1rem;}
.work-site{margin:0;font-size:1.25rem;font-weight:650;line-height:1.25;}
.work-meta{margin:.1rem 0 .6rem;color:var(--muted);font-size:.92rem;}
.entry{margin:0;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:.15rem 1rem;padding:.5rem 0;border-bottom:1px solid var(--rule);}
.entry dt{color:var(--muted);font-size:.92rem;margin:0;} .entry dd{margin:0;font-weight:600;text-align:right;}
/* Analysis */
.pattern{display:flex;gap:.65rem;align-items:flex-start;margin:.4rem 0 .5rem;}
.pattern b{display:block;font-size:1.15rem;line-height:1.3;} .pattern span.why{display:block;color:var(--muted);font-size:.92rem;}
.timeline{position:relative;width:100%;height:84px;display:flex;align-items:center;gap:1px;padding:0 2px;overflow:hidden;background:var(--track);border:1px solid var(--rule);box-sizing:border-box;}
.timeline .bar{position:relative;flex:1 1 0;min-width:0;background:var(--ink);}
.timeline .band{position:absolute;top:0;bottom:0;border-bottom:5px solid;box-sizing:border-box;}
.timeline-key{display:flex;flex-wrap:wrap;gap:.2rem .9rem;margin:.25rem 0 .5rem;color:var(--muted);font-size:.85rem;}
.timeline-key i{display:inline-block;width:12px;height:12px;margin-right:.3rem;vertical-align:-1px;}
.st-key-listen_times{gap:.4rem;flex-wrap:wrap;} .st-key-listen_times .stButton{width:auto!important;} .st-key-listen_times button{min-height:44px;width:auto;font-variant-numeric:tabular-nums;}
.problem{background:var(--q-bg);color:var(--on-problem);padding:.6rem .75rem;border-radius:var(--radius);margin:.3rem 0 .6rem;}
.empty{padding:1rem 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);}
.empty b{display:block;font-size:1.15rem;margin-bottom:.15rem;}
.foot{margin:1.75rem 0 0;padding-top:.5rem;border-top:1px solid var(--rule);color:var(--muted);font-size:.85rem;}
.prose{max-width:68ch;}
@media (min-width:900px){h1{font-size:1.65rem!important;} .st-key-masthead{flex-direction:row;align-items:center;justify-content:space-between;}
  .st-key-masthead>div{width:auto!important;}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;}}
</style>'''

LOGIN_CSS = '''<style>
.block-container{max-width:400px;padding-top:clamp(3rem,16vh,8rem)!important;}
.login-name{margin:0;color:var(--action);font-weight:700;font-size:1rem;}
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
    st.markdown(f'<p class="login-name">{PRODUCT}</p>', unsafe_allow_html=True)
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
    st.caption(NOTICE)
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


def setup(title):
    """Page chrome shared by every screen."""
    st.set_page_config(page_title=f'{title} | {PRODUCT}', page_icon=str(ROOT / 'app/lung_icon.png'),
                       layout='wide', initial_sidebar_state='collapsed')
    st.markdown(CSS, unsafe_allow_html=True)
    require_password()
    store = open_store()
    with st.container(key='masthead'):
        st.markdown(f'<p class="brand">{PRODUCT}</p>', unsafe_allow_html=True)
        with st.container(key='nav', horizontal=True):
            st.page_link('Home.py', label='Assessments')
            st.page_link('pages/2_Follow_up.py', label='Follow-up')
            st.page_link('pages/3_Study_data.py', label='Study data')
            st.page_link('pages/4_About_analysis.py', label='About')
    return store


def footer():
    st.markdown(f'<p class="foot">{NOTICE} Use study codes only, never names.</p>', unsafe_allow_html=True)


def empty_state(title, body):
    st.markdown(f'<div class="empty" role="status"><b>{escape(title)}</b>{escape(body)}</div>',
                unsafe_allow_html=True)


def clean_code(value):
    value = str(value).strip().upper()
    return value if re.fullmatch(r'[A-Z][A-Z0-9]{0,15}', value) else None


def parse_trial(trial_id):
    rid, _, examiner = trial_id.partition('__')
    case, site, attempt, mode = rid.split('_')
    number = int(attempt[1:])
    return dict(case=case, site=site, round=Store.base_trial(number), number=number, retake=number >= 100,
                mode=mode, examiner=examiner)


def site_rows(store, assessment):
    """The current recording per site: a retake replaces its original on the chart."""
    rows, numbers = {}, {}
    for row in store.table('rater_study'):
        info = parse_trial(row['trial_id'])
        if all(info[k] == assessment[k] for k in ('case', 'examiner', 'round', 'mode')) and \
                info['number'] >= numbers.get(info['site'], 0):
            rows[info['site']], numbers[info['site']] = row, info['number']
    return rows


def next_retake(store, assessment, site):
    """Next stored number for a retake of this site and trial, across all examiners."""
    numbers = [info['number'] for info in (parse_trial(r['trial_id']) for r in store.table('rater_study'))
               if info['site'] == site and all(info[k] == assessment[k] for k in ('case', 'round', 'mode'))]
    return max([assessment['round'] * 100, *numbers]) + 1


def is_replaced(row, all_rows):
    info = parse_trial(row['trial_id'])
    return any(other['number'] > info['number'] and
               all(other[k] == info[k] for k in ('case', 'site', 'round', 'mode', 'examiner'))
               for other in (parse_trial(r['trial_id']) for r in all_rows))


def trial_words(info):
    return f"trial {info['round']}" + (' retake' if info['retake'] else '')


def site_status(row):
    if not row:
        return 'Not started'
    if row['assisted_label']:
        return 'Signed'
    if row['ai_prediction_id'] and row['ai_finding'] == 'Q':
        return 'Record again'
    if row['manual_locked_at']:
        return 'Ready to sign'
    return 'Needs finding'


def is_complete(row):
    return bool(row) and bool(row['assisted_label'])  # An unusable recording still needs a retake.


def open_assessment(case, examiner, round_number, mode, site=None):
    st.session_state['assessment'] = dict(case=case, examiner=examiner, round=round_number, mode=mode)
    st.session_state['site'] = site
    st.session_state.pop('chart_view', None)
    st.switch_page('pages/1_Assessment.py')


def clock(seconds):
    seconds = int(seconds)
    return f'{seconds // 60}:{seconds % 60:02d}'


def qc_words(reason):
    parts = []
    for raw in filter(None, str(reason).split(';')):
        code, _, detail = raw.partition(':')
        words = QC_WORDS.get(code.strip(), raw.strip())
        parts.append(f'{words} ({detail.strip()})' if detail else words)
    parts = parts[:1] + [x[0].lower() + x[1:] for x in parts[1:]]
    return '; '.join(parts) or 'The recording failed the quality check'


def marked_segments(windows):
    """Merge overlapping analysis windows of one pattern into times to hear again."""
    merged = []
    for window in windows:
        index = int(np.argmax(window['probabilities']))
        if index == 0:
            continue
        code = 'NCWB'[index]
        if merged and merged[-1]['code'] == code and window['start_s'] <= merged[-1]['end_s']:
            merged[-1]['end_s'] = window['end_s']
        else:
            merged.append(dict(code=code, start_s=window['start_s'], end_s=window['end_s']))
    return merged


def timeline(path, segments):
    """Inline SVG of the sound envelope with marked times; includes a text alternative."""
    values, rate = sf.read(path, dtype='float32', always_2d=True)
    mono = values.mean(axis=1)
    duration = len(mono) / rate
    bins = np.array_split(np.abs(mono), 96)
    levels = np.array([b.max() if b.size else 0 for b in bins])
    levels = levels / max(float(levels.max()), 1e-6)
    bands = ''.join(
        f'<span class="band" style="left:{s["start_s"] / duration:.2%};'
        f'width:{(min(s["end_s"], duration) - s["start_s"]) / duration:.2%};'
        f'background:{SEGMENT_FILL[s["code"]]}33;border-bottom-color:{SEGMENT_FILL[s["code"]]}"></span>'
        for s in segments)
    bars = ''.join(f'<span class="bar" style="height:{max(v * 82, 2):.0f}%"></span>' for v in levels)
    if segments:
        spoken = ', '.join(f'{PATTERN_NAMES[s["code"]]} from {clock(s["start_s"])} to {clock(min(s["end_s"], duration))}'
                           for s in segments)
    else:
        spoken = 'no marked times'
    key = ''.join(f'<span><i style="background:{SEGMENT_FILL[c]}"></i>{PATTERN_NAMES[c]}</span>'
                  for c in dict.fromkeys(s['code'] for s in segments))
    st.html(f'<div class="timeline" role="img" aria-label="Sound over {clock(duration)}: {escape(spoken)}">'
            f'{bands}{bars}</div><div class="timeline-key"><span>0:00 to {clock(duration)}</span>{key}</div>')


def analysis_panel(result, path, trial_id):
    code = result['finding']
    css, title, guidance = FINDING_STYLE[code]
    segments = [] if code == 'Q' else marked_segments(result['window_outputs'])
    if code == 'Q':
        guidance = f"{qc_words(result['qc_reason'])}."
    elif code == 'N' and segments:
        guidance = 'Most of the recording showed no pattern. Listen again at the marked times before you sign.'
    st.markdown(f'<div class="pattern {css}" role="status">'
                f'<div><b>{escape(title)}</b><span class="why">{escape(guidance)}</span></div></div>',
                unsafe_allow_html=True)
    if code == 'Q':
        return
    windows = result['window_outputs']
    timeline(path, segments)
    start_key = f'start_{trial_id}'
    if segments:
        st.caption('Choose a marked time, then press play.')
        with st.container(key='listen_times', horizontal=True):
            for i, segment in enumerate(segments):
                if st.button(f'{clock(segment["start_s"])} {PATTERN_NAMES[segment["code"]]}', key=f'hear_{trial_id}_{i}'):
                    st.session_state[start_key] = (int(segment['start_s']), time.time_ns())
    seconds, nonce = st.session_state.get(start_key, (0, 0))
    with st.container(key=f'player_{nonce}'):  # A new key moves the player even on a repeat tap.
        st.audio(Path(path).read_bytes(), format='audio/wav', start_time=seconds)
    if code == 'U':
        basis = (f"The strongest pattern covered {result['top_score']:.0%} of the 3-second segments. "
                 f"It needs {result['threshold']:.0%} to name one.")
    else:
        count = round(result['top_score'] * len(windows))
        ending = 'showed no crackle or wheeze' if code == 'N' else 'showed this pattern'
        basis = f'{count} of {len(windows)} overlapping 3-second segments {ending}.'
    st.caption(f'{basis} This describes the sound only. It is not a diagnosis.')
