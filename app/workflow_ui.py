"""Shared UI for the auscultation record (D-069). Inference stays in stethoai.app_core.

Direction: docs/design/TOOL_DIRECTION.md. An anatomy chart, phone first.
Findings are always written in words, never colour alone.
"""
from html import escape
from pathlib import Path
import hmac
import tempfile
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

from stethoai.app_core import LABELS, MANUAL_LABELS, Store, check_audio, decodable, load_model, repeat_sites
from stethoai.drive_store import DriveClient, DriveError, DriveState, DriveStore

PRODUCT = 'Lung Sound Record'
NOTICE = 'This research prototype is not a medical device and does not diagnose.'
SITE_NAMES = {'ARU': 'Right upper', 'ARL': 'Right lower', 'ALU': 'Left upper', 'ALL': 'Left lower',
              'PRU': 'Right upper', 'PRL': 'Right lower', 'PLU': 'Left upper', 'PLL': 'Left lower'}
# Same order as section 4 item 7 of the student recording guide.
SITE_GROUPS = (('Anterior chest', ('ARU', 'ALU', 'ARL', 'ALL')),
               ('Posterior chest', ('PRU', 'PLU', 'PRL', 'PLL')))
SITE_ORDER = tuple(code for _, codes in SITE_GROUPS for code in codes)
MODE_NAMES = {'BELL': 'Bell', 'DIA': 'Diaphragm'}
# Diaphragm is listed first and chosen by default: the protocol records in lung (diaphragm) mode (D-097).
MODE_ORDER = ('DIA', 'BELL')
MODE_HELP = {'DIA': 'Used for lung sounds in this study.', 'BELL': 'Not used for lung sounds in this study.'}
DRAFT_STATUS = 'Attached, not saved yet'
# Public project website (D-087, D-088); the tool links back to it from sign-in and every footer.
WEBSITE_URL = 'https://lung-sound-study.pages.dev'
FINDINGS = {'N': 'No adventitious sounds', 'C': 'Crackles', 'W': 'Wheezes',
            'B': 'Crackles and wheezes', 'U': 'Unable to determine'}
PATTERN_NAMES = {'C': 'Crackle-like', 'W': 'Wheeze-like', 'B': 'Crackle- and wheeze-like'}
SEGMENT_FILL = {'C': '#a4560c', 'W': '#1b578a', 'B': '#792d60'}

# Analysis display: (css class, headline, guidance).
FINDING_STYLE = {
    'N': ('normal', 'No crackle or wheeze pattern', 'No times were marked.'),
    'C': ('crackle', 'Crackle-like sound', 'Play the marked times again before you sign.'),
    'W': ('wheeze', 'Wheeze-like sound', 'Play the marked times again before you sign.'),
    'B': ('both', 'Crackle-like and wheeze-like sound', 'Play the marked times again before you sign.'),
    'U': ('uncertain', 'No clear pattern', 'No pattern was strong enough to name. Rely on what you heard.'),
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
.st-key-nav{gap:0!important;flex-wrap:nowrap;overflow:visible;}
/* All four links fit at 390 px; a hidden sideways scroll had cut off "About" (D-112). */
@media (max-width:899px){div.st-key-nav.stHorizontalBlock{width:100%!important;justify-content:space-between;gap:0!important;column-gap:0!important;}.st-key-nav [data-testid="stElementContainer"]{flex:0 0 auto;width:auto!important;min-width:max-content;}.st-key-nav [data-testid="stPageLink"] a span{overflow:visible;text-overflow:clip;}.st-key-nav [data-testid="stPageLink"] a{padding:.25rem .35rem;justify-content:center;}.st-key-nav [data-testid="stPageLink"] a p{font-size:.875rem;}}
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
[data-testid="stFileUploaderDropzone"]{min-height:64px;border:1px dashed var(--rule-strong);border-radius:var(--radius);background:var(--sheet);}
[data-testid="stFileUploaderDropzone"] button{min-height:44px;min-width:88px;}
[data-testid="stNumberInputStepDown"],[data-testid="stNumberInputStepUp"]{min-width:44px!important;min-height:44px!important;}
[data-testid="stNumberInputContainer"]{min-height:46px;}
:focus-visible,button:focus-visible,a:focus-visible,label:has(:focus-visible){outline:3px solid var(--focus)!important;outline-offset:1px;}
/* Text fields: the ring goes round the whole field. On the inner input it was clipped to an orange sliver (D-112). */
:is([data-testid="stTextInputRootElement"],[data-testid="stNumberInputContainer"],[data-testid="stTextAreaRootElement"],[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"]>div) :is(input,textarea):focus-visible{outline:none!important;}
:is([data-testid="stTextInputRootElement"],[data-testid="stNumberInputContainer"],[data-testid="stTextAreaRootElement"],[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"]>div):focus-within{outline:3px solid var(--focus)!important;outline-offset:-3px;}
:is([data-testid="stTextInputRootElement"],[data-testid="stNumberInputContainer"],[data-testid="stTextAreaRootElement"],[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"]>div) button:focus-visible{outline-offset:-3px;}
/* Scrolling regions Chrome makes focusable: the ring is drawn inside so the window edge cannot cut it. */
[data-testid="stAppViewContainer"] :is([data-testid="stMain"],[data-testid="stMainBlockContainer"],section):focus-visible{outline-offset:-3px;}
.st-key-nav [data-testid="stPageLink"] a:focus-visible{outline-offset:-3px;}
[data-testid="stHeaderActionElements"]{display:none!important;}
[data-testid="stElementContainer"]:has([data-testid="stRadio"]),[data-testid="stRadio"],[data-testid="stRadio"]>div,[data-testid="stRadio"] [role="radiogroup"]{width:100%!important;}
[data-testid="stRadio"] [role="radiogroup"]>label,[data-testid="stRadio"] [role="radiogroup"]>div{width:100%!important;max-width:none!important;}
[class*="st-key-row_"],[class*="st-key-row_"] [data-testid="stElementContainer"],[class*="st-key-row_"] .stButton,[class*="st-key-row_"] [data-testid="stTooltipHoverTarget"]{width:100%!important;}
[class*="st-key-row_"] .stButton button{width:100%!important;min-height:52px;}
/* Flowsheet */
.st-key-sheet [data-testid="stHorizontalBlock"]{align-items:flex-start;}
[class*="st-key-sign_"],[class*="st-key-save_"],[class*="st-key-again_"],[class*="st-key-next_"],[class*="st-key-replace_"],[class*="st-key-replace_"] .stButton,
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
/* Segmented pairs (Front/Back, Trial 1/Trial 2): one joined control, shared edge, outer corners rounded (D-112). */
div.stHorizontalBlock.st-key-view_switch,div.stHorizontalBlock.st-key-trial_switch{gap:0!important;column-gap:0!important;flex-wrap:nowrap;margin-bottom:.15rem;}
.st-key-view_switch [data-testid="stElementContainer"],.st-key-trial_switch [data-testid="stElementContainer"]{flex:1 1 0;width:auto!important;min-width:0;}
.st-key-view_switch .stButton,.st-key-trial_switch .stButton{width:100%;}
.st-key-view_switch button,.st-key-trial_switch button{width:100%;min-height:48px;margin:0;}
.st-key-view_front button,.st-key-trial_pick_1 button{border-radius:var(--radius) 0 0 var(--radius)!important;}
.st-key-view_back button,.st-key-trial_pick_2 button{border-radius:0 var(--radius) var(--radius) 0!important;margin-left:-1px;}
.st-key-trial_switch{max-width:22rem;} .seg-label{margin:0 0 .3rem;font-size:.875rem;color:var(--ink);}
.legend{display:flex;flex-wrap:wrap;gap:.3rem 1rem;color:var(--muted);font-size:.92rem;}
.legend span{display:inline-flex;align-items:center;gap:.4rem;}
.mk{display:inline-grid;place-items:center;width:18px;height:18px;box-sizing:border-box;border-radius:50%;border:2px solid var(--ink);background:#fff;font-size:11px;font-weight:800;font-style:normal;line-height:1;}
.saved{color:var(--n);font-weight:600;}
.unsaved{margin:.6rem 0 .3rem;font-size:.92rem;color:var(--ink);}
.st-key-unsaved_sites{margin-bottom:.75rem;} .st-key-unsaved_links{gap:.4rem;flex-wrap:wrap;}
.st-key-unsaved_links .stButton{width:auto!important;} .st-key-unsaved_links button{min-height:44px;width:auto;}
input::placeholder{color:var(--muted)!important;opacity:1;}
.draft-file dd{overflow-wrap:anywhere;}
/* Options with a description: the row holds label and description together. */
[data-testid="stRadio"] [role="radiogroup"]>div:has([data-testid="stRadioCaption"]){border-bottom:1px solid var(--rule);padding-bottom:.45rem;}
[data-testid="stRadio"] [role="radiogroup"]>div:has([data-testid="stRadioCaption"]) label{border-bottom:0;min-height:40px;padding-bottom:0;}
[data-testid="stRadio"] [role="radiogroup"]>div[data-selected="true"]:has([data-testid="stRadioCaption"]){background:var(--selected);}
[data-testid="stRadio"] [role="radiogroup"]>div[data-selected="true"]:has([data-testid="stRadioCaption"]) label{background:transparent;}
[data-testid="stRadioCaption"]{padding:0 .5rem 0 2.15rem;} [data-testid="stRadioCaption"] p{font-size:.9rem!important;margin:0;}
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

.st-key-examiner_line p.examiner{margin:0!important;font-size:1.15rem;font-weight:650;} .st-key-examiner_line{gap:1rem;flex-wrap:wrap;margin-bottom:.5rem;}
.st-key-examiner_line .stButton{width:auto!important;} .st-key-examiner_line button{min-height:44px;width:auto;}
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
/* Analysis in progress (D-111): the model's real stages, drawn on the recording itself. */
.trace{margin:.4rem 0 .6rem;}
.trace .timeline .veil{position:absolute;top:0;bottom:0;right:0;background:var(--track);opacity:.62;transition:left .16s ease-out;}
.trace .timeline .cursor{position:absolute;top:-1px;bottom:-1px;width:2px;background:var(--action);transition:left .16s ease-out;}
.trace .timeline .cell{position:absolute;top:0;bottom:0;border-bottom:5px solid;box-sizing:border-box;animation:settle .2s ease-out both;}
.trace ul.steps{list-style:none;margin:.6rem 0 0!important;padding:0!important;border-top:1px solid var(--rule);}
.steps li{position:relative;margin:0!important;padding:.45rem 0 .5rem;border-bottom:1px solid var(--rule);color:var(--ink);font-size:.95rem;line-height:1.45;}
.steps li.pending{color:var(--muted);}
.steps li.active{font-weight:600;}
.steps li.active::after{content:"";position:absolute;left:0;bottom:-1px;height:2px;width:30%;background:var(--action);animation:travel 1.3s cubic-bezier(.45,0,.55,1) infinite alternate;}
.steps li.failed{color:var(--on-problem);font-weight:600;}
.steps .count{display:block;color:var(--muted);font-weight:400;font-size:.88rem;font-variant-numeric:tabular-nums;}
@keyframes travel{from{left:0;}to{left:70%;}}
.problem{background:var(--q-bg);color:var(--on-problem);padding:.6rem .75rem;border-radius:var(--radius);margin:.3rem 0 .6rem;}
.empty{padding:1rem 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);}
.empty b{display:block;font-size:1.15rem;margin-bottom:.15rem;}
.foot{margin:1.75rem 0 0;padding-top:.5rem;border-top:1px solid var(--rule);color:var(--muted);font-size:.85rem;}
.prose{max-width:68ch;}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap;margin:0;}
@media (min-width:900px){h1{font-size:1.65rem!important;} .st-key-masthead{flex-direction:row;align-items:center;justify-content:space-between;}
  .st-key-masthead>div{width:auto!important;}}
/* Motion (D-100): feedback and continuity only. */
@keyframes arrive{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:none;}}
@keyframes settle{from{opacity:0;}to{opacity:1;}}
@keyframes stamp{0%{transform:scale(.55);}60%{transform:scale(1.18);}100%{transform:scale(1);}}
[class*="st-key-site_"]{animation:arrive .24s cubic-bezier(.16,1,.3,1) both;}
.saved,.unsaved,.draft-file{animation:settle .2s ease-out both;}
.stButton button:active:not(:disabled),[data-testid="stFormSubmitButton"] button:active{transform:scale(.98);}
.stButton button,[data-testid="stFormSubmitButton"] button{transition:background-color .15s ease-out,color .15s ease-out,transform .1s ease-out;}
[class*="st-key-pick_"] button::before{transition:transform .15s ease-out,box-shadow .2s ease-out;}
[data-testid="stRadio"] [role="radiogroup"] label{transition:background-color .15s ease-out;}
@media (max-width:640px){[data-testid="stHorizontalBlock"]{gap:1.5rem!important;}}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{transition:none!important;animation:none!important;}}
</style>'''

LOGIN_CSS = '''<style>
.block-container{max-width:400px;padding-top:clamp(3rem,16vh,8rem)!important;}
.login-name{margin:0;color:var(--action);font-weight:700;font-size:1rem;}
</style>'''


@st.cache_resource(show_spinner=False)
def model_state():
    """Survives page reruns and module reloads, so the trace says the model is loading only when it is."""
    return {}


@st.cache_resource(show_spinner=False)
def get_model():
    # load_model verifies the frozen v2 bytes and config before torch.load (D-113, D-114). Never read a manifest here.
    model = load_model()
    model_state()['ready'] = True
    return model


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
    st.markdown(f'<p class="foot"><a href="{WEBSITE_URL}">Back to the project website</a></p>', unsafe_allow_html=True)
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


# One entry script draws the masthead and runs the chosen page (D-112). A folder of separate pages made Streamlit
# clear the whole screen on every switch, which read as a blank flash; now the masthead stays and only the body changes.
# The URL paths are the ones the separate pages had, so saved links and reload restore keep working.
HOME = 'pages/0_Assessments.py'
PAGES = ((HOME, 'Assessments', None), ('pages/1_Assessment.py', 'Assessment', 'Assessment'),
         ('pages/2_Follow_up.py', 'Follow-up', 'Follow_up'), ('pages/3_Study_data.py', 'Study data', 'Study_data'),
         ('pages/4_About_analysis.py', 'About the analysis', 'About_analysis'))


def frame(title):
    """Chrome shared by every screen, drawn once per run by the entry script before the page runs."""
    st.set_page_config(page_title=f'{title} | {PRODUCT}', page_icon=str(ROOT / 'app/lung_icon.png'),
                       layout='wide', initial_sidebar_state='collapsed')
    st.markdown(CSS, unsafe_allow_html=True)
    require_password()
    with st.container(key='masthead'):
        st.markdown(f'<p class="brand">{PRODUCT}</p>', unsafe_allow_html=True)
        with st.container(key='nav', horizontal=True):
            for path, label in ((HOME, 'Assessments'), ('pages/2_Follow_up.py', 'Follow-up'),
                                ('pages/3_Study_data.py', 'Study data'), ('pages/4_About_analysis.py', 'About')):
                st.page_link(path, label=label)


HEADING_SLOT = '_heading_slot'


def heading(text):
    """The page heading. The entry script draws it first, outside the page, because a page switch removes every
    element the old page drew at once; an entry-script heading stays until the new one replaces it in place, so the
    body is never empty. A page rewrites it only when its text differs. Without a slot (a page run alone) it is st.title."""
    slot, drawn = st.session_state.pop(HEADING_SLOT, (None, None))
    if text != drawn:
        (slot or st).title(text)


def first_heading(title):
    """The heading the entry script draws before the page runs; for an assessment, its case code (session or URL)."""
    if title != 'Assessment':
        return title
    opened = st.session_state.get('assessment')
    case = opened['case'] if opened else clean_code(st.query_params.get('case', ''))
    return f'Case {case}' if case else title


def setup(title):
    """Per-page start: the busy mark for this run and the record store. The chrome is drawn by frame()."""
    # A tap's busy mark belongs to this run only, so an interrupted save never fires again later.
    st.session_state['busy_now'] = st.session_state.pop('busy', None)
    require_password()  # Also here, so no page can run without sign-in even if started on its own.
    return open_store()


def footer():
    st.markdown(f'<p class="foot">{NOTICE} <a href="{WEBSITE_URL}">Project website</a></p>',
                unsafe_allow_html=True)


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


def recording_words(number):
    """Examiner words for a protocol recording number; the word trial is never shown (D-119)."""
    return {1: 'first recording', 2: 'second recording'}.get(number, f'recording {number}')


def trial_words(info):
    return recording_words(info['round']) + (', recorded again' if info['retake'] else '')


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


# Blinding for rater agreement (D-106): a session works under one examiner code and sees only that code's records.
def session_examiner():
    return st.session_state.get('examiner')


def own_rows(store):
    """Records saved under the examiner code of this session; none before a code is entered."""
    code = session_examiner()
    return [r for r in store.table('rater_study') if code and parse_trial(r['trial_id'])['examiner'] == code]


# Recording order (D-119): the four front sites once, then a second recording at the two front sites drawn for this
# assessment, then the same on the back. Only one turn of the patient. Records made before D-119 keep every
# recording they hold, so an older case with a second recording at all eight sites still shows them.
LISTEN_AGAIN = 'Second recording needed'  # A state, like the other statuses; the step button says Listen again.


def held_numbers(store, assessment):
    """Protocol recording numbers already stored per site for this case, examiner and mode."""
    held = {}
    for row in store.table('rater_study'):
        info = parse_trial(row['trial_id'])
        if all(info[k] == assessment[k] for k in ('case', 'examiner', 'mode')):
            held.setdefault(info['site'], set()).add(info['round'])
    return held


def walk(store, assessment):
    """Every site and recording number in recording order."""
    chosen, held, steps = repeat_sites(assessment['case'], assessment['examiner'], assessment['mode']), held_numbers(store, assessment), []
    for _, codes in SITE_GROUPS:
        steps += [(code, 1) for code in codes]
        for code in codes:
            numbers = held.get(code, set()) | ({2} if code in chosen else set())
            steps += [(code, n) for n in sorted(numbers) if n >= 2]
    return steps


def side_open(site, done):
    """Second recordings on a side open only once its four first recordings are signed (D-120, no back-to-back repeat)."""
    codes = next(codes for _, codes in SITE_GROUPS if site in codes)
    return all((code, 1) in done for code in codes)


def steps_at(store, assessment, site):
    """Recording numbers this site can open now: the first, and a second once its side is ready or one is held."""
    done, held = signed_pairs(store, assessment), held_numbers(store, assessment).get(site, set())
    return [n for code, n in walk(store, assessment) if code == site and (n == 1 or n in held or side_open(site, done))]


def signed_pairs(store, assessment):
    return {(site, n) for site, n in walk(store, assessment)
            if is_complete(site_rows(store, {**assessment, 'round': n}).get(site))}


def next_step(store, assessment, after=None):
    """The earliest site and recording not signed that can open now, in recording order (D-120).

    `after` is the step just signed; it is never offered again. A site skipped earlier is offered before later ones."""
    steps, done = walk(store, assessment), signed_pairs(store, assessment)
    return next((step for step in steps if step not in done and step != after
                 and (step[1] == 1 or side_open(step[0], done))), None)


def open_number(store, assessment, site):
    """The recording a site opens at: its first not signed, else its last (D-119)."""
    numbers, done = steps_at(store, assessment, site), signed_pairs(store, assessment)
    return next((n for n in numbers if (site, n) not in done), numbers[-1])


def open_assessment(case, examiner, round_number, mode, site=None):
    assessment = dict(case=case, examiner=examiner, round=round_number, mode=mode)
    st.session_state['assessment'] = assessment
    st.session_state['site'] = site
    st.session_state.pop('chart_view', None)
    st.switch_page('pages/1_Assessment.py', query_params=url_state(assessment, site))


# A reload or a phone app switch starts a new Streamlit session; the open case and site live in the URL
# so the examiner returns to the same place after sign-in (D-108). Study codes only, never names.
def url_state(assessment, site=None):
    state = {'case': assessment['case'], 'examiner': assessment['examiner'], 'mode': assessment['mode']}
    if assessment['round'] is not None:  # No trial yet means the page opens the first site and trial not signed (D-107).
        state['trial'] = str(assessment['round'])
    return {**state, 'site': site} if site in SITE_ORDER else state


def assessment_from_url(store):
    """The assessment and site named in the URL, or (None, None) when anything is missing or malformed."""
    query = st.query_params
    case, examiner = clean_code(query.get('case', '')), clean_code(query.get('examiner', ''))
    mode, trial = query.get('mode'), query.get('trial', '')
    if not (case and examiner and mode in store.protocol['modes'] and (trial == '' or trial.isdigit() and 1 <= int(trial) <= 99)):
        return None, None
    site = query.get('site')
    return (dict(case=case, examiner=examiner, round=int(trial) if trial else None, mode=mode),
            site if site in SITE_ORDER else None)


def keep_in_url(assessment, site):
    wanted = url_state(assessment, site)
    if st.query_params.to_dict() != wanted:
        st.query_params.from_dict(wanted)


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


def envelope(path):
    """Peak level in 96 equal slices of the recording, scaled to 0..1, and its length in seconds."""
    values, rate = sf.read(path, dtype='float32', always_2d=True)
    mono = values.mean(axis=1)
    bins = np.array_split(np.abs(mono), 96)
    levels = np.array([b.max() if b.size else 0 for b in bins])
    return levels / max(float(levels.max()), 1e-6), len(mono) / rate


def envelope_bars(levels):
    return ''.join(f'<span class="bar" style="height:{max(v * 82, 2):.0f}%"></span>' for v in levels)


def timeline(path, segments):
    """Inline SVG of the sound envelope with marked times; includes a text alternative."""
    levels, duration = envelope(path)
    bands = ''.join(
        f'<span class="band" style="left:{s["start_s"] / duration:.2%};'
        f'width:{(min(s["end_s"], duration) - s["start_s"]) / duration:.2%};'
        f'background:{SEGMENT_FILL[s["code"]]}33;border-bottom-color:{SEGMENT_FILL[s["code"]]}"></span>'
        for s in segments)
    bars = envelope_bars(levels)
    if segments:
        spoken = ', '.join(f'{PATTERN_NAMES[s["code"]]} from {clock(s["start_s"])} to {clock(min(s["end_s"], duration))}'
                           for s in segments)
    else:
        spoken = 'no marked times'
    key = ''.join(f'<span><i style="background:{SEGMENT_FILL[c]}"></i>{PATTERN_NAMES[c]}</span>'
                  for c in dict.fromkeys(s['code'] for s in segments))
    st.html(f'<div class="timeline" role="img" aria-label="Sound over {clock(duration)}: {escape(spoken)}">'
            f'{bands}{bars}</div><div class="timeline-key"><span>0:00 to {clock(duration)}</span>{key}</div>')


SEGMENT_WORDS = {'N': 'showed no crackle or wheeze', 'C': ('was', 'were', 'crackle-like'), 'W': ('was', 'were', 'wheeze-like'),
                 'B': ('was', 'were', 'crackle- and wheeze-like')}


def segment_phrase(code, n):
    words = SEGMENT_WORDS[code]
    if isinstance(words, str):
        return f'{n} {words}'
    single, plural, kind = words
    return f'{n} {single if n == 1 else plural} {kind}'
SWEEP_SECONDS = 1.6  # D-111: the real segment outputs are drawn across at most this long, the same for every trial.


class AnalysisTrace:
    """Shows the analysis stages as they happen, on the recording's own sound envelope (D-111).

    Every line reports what a stage actually produced. Only the drawing of the per-segment outputs
    is paced, so that it can be followed; no stage is invented and no result is held back.
    """

    def __init__(self, path, place='the study record'):
        self.place = place
        try:
            self.levels, self.duration = envelope(path)
        except (OSError, RuntimeError, ValueError):  # An unreadable file still gets its quality-check line.
            self.levels, self.duration = np.zeros(96), 1.
        self.slot = st.empty()
        self.steps = {}
        self.cells, self.frontier = [], 0.
        self.count = ''

    def step(self, key, text, state):
        self.steps[key] = (text, state)
        self.draw()

    def draw(self):
        width = max(self.duration, 1e-6)
        cells = ''.join(
            f'<span class="cell" style="left:{c["start_s"] / width:.2%};width:{(c["end_s"] - c["start_s"]) / width:.2%};'
            f'background:{SEGMENT_FILL[c["code"]]}26;border-bottom-color:{SEGMENT_FILL[c["code"]]}"></span>'
            for c in self.cells if c['code'] != 'N')
        edge = min(self.frontier / width, 1)
        # The cursor marks the segment being drawn; it shows only while segment outputs are being drawn.
        cursor = f'<span class="cursor" style="left:calc({edge:.2%} - 1px)"></span>' if 0 < edge < 1 else ''
        lines = []
        for key, (text, state) in self.steps.items():
            extra = f'<span class="count" aria-hidden="true">{escape(self.count)}</span>' if key == 'segments' and self.count else ''
            lines.append(f'<li class="{state}">{escape(text)}{extra}</li>')
        current = next((t for t, s in reversed(self.steps.values()) if s in ('active', 'failed')), '')
        self.slot.html(
            f'<div class="trace">'
            f'<div class="timeline" aria-hidden="true">{envelope_bars(self.levels)}{cells}'
            f'<span class="veil" style="left:{edge:.2%}"></span>{cursor}</div>'
            f'<ul class="steps">{"".join(lines)}</ul>'
            f'<p class="sr-only" role="status">{escape(current)}</p></div>')

    def load_model(self, loader, ready):
        if not ready:
            self.step('model', 'The analysis model is loading, which takes a few seconds once.', 'active')
        model = loader()
        if not ready:
            self.step('model', 'The analysis model is loaded.', 'done')
        self.step('quality', 'The recording is being checked for quality.', 'active')
        return model

    def update(self, stage, data):
        if stage == 'quality':
            if data['qc_status'] == 'pass':
                self.step('quality', f"The recording passed the quality check ({data['duration_s']:.1f} seconds).", 'done')
            else:
                self.step('quality', f"The recording did not pass the quality check. {qc_words(data['qc_reason'])}.", 'failed')
        elif stage == 'prepared':
            self.step('segments', f"The model is examining {data['segments']} overlapping 3‑second segments.", 'active')
        elif stage == 'analyzed' and data['finding'] != 'Q':
            self.sweep(data['window_outputs'])
            self.close_segments(data)
        elif stage == 'stored':
            self.step('quality', 'The saved analysis was loaded and was not run again.', 'done')
            if data['finding'] != 'Q':
                self.cells = [self.cell(w) for w in data['window_outputs']]
                self.frontier = self.duration
                self.close_segments(data)
        elif stage == 'saving':
            self.step('saving', f'The analysis is being saved to {self.place}.', 'active')

    @staticmethod
    def cell(window):
        return dict(start_s=window['start_s'], end_s=window['end_s'],
                    code='NCWB'[int(np.argmax(window['probabilities']))])

    def sweep(self, windows):
        pause = min(.18, SWEEP_SECONDS / max(len(windows), 1))
        for i, window in enumerate(windows, 1):
            self.cells.append(self.cell(window))
            self.frontier = window['end_s']
            self.count = f"{i} of {len(windows)} {'is' if i == 1 else 'are'} drawn."
            self.draw()
            time.sleep(pause)

    def close_segments(self, result):
        self.count = ''
        windows = result['window_outputs']
        tally = {code: sum(1 for w in windows if self.cell(w)['code'] == code) for code in 'NCWB'}
        parts = [segment_phrase(code, n) for code, n in tally.items() if n]
        self.step('segments', f"Of {len(windows)} segments, {', '.join(parts)}.", 'done')
        top = result['top_score'] or 0
        if result['finding'] == 'U':
            words = f"No pattern reached the {result['threshold']:.0%} needed to be named; the largest was {top:.0%}."
        else:
            words = f"{top:.0%} of segments agreed; {result['threshold']:.0%} is needed to report a result."
        self.step('vote', words, 'done')

    def finish(self):
        self.step('saving', f'The analysis was saved to {self.place}.', 'done')

    def fail(self):
        for key, (text, state) in list(self.steps.items()):
            if state == 'active':
                self.steps[key] = (text, 'failed')
        self.draw()


def analysis_panel(result, path, trial_id, signed=False):
    code = result['finding']
    css, title, guidance = FINDING_STYLE[code]
    segments = [] if code == 'Q' else marked_segments(result['window_outputs'])
    if code == 'Q':
        guidance = f"{qc_words(result['qc_reason'])}."
    elif code == 'N' and segments:
        guidance = 'Most of the recording showed no pattern. Play the marked times again before you sign.'
    if signed:  # After signing, the marked times are a record, not an instruction.
        guidance = guidance.replace('Play the marked times again before you sign.',
                                    'The marked times show where the pattern was found.')
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
        basis = (f"The strongest pattern covered {result['top_score']:.0%} of the segments; "
                 f"{result['threshold']:.0%} is needed to name one.")
    else:
        count = round(result['top_score'] * len(windows))
        ending = 'showed no crackle or wheeze' if code == 'N' else 'showed this pattern'
        basis = f'{count} of {len(windows)} segments {ending}.'
    st.caption(basis)


# Unsaved work and slow saves (D-098, D-099).
# Streamlit forgets a widget's value once it leaves the screen, so attached files and chosen findings are
# copied into plain session keys and shown again when the examiner returns to the site.

def assessment_tag(assessment):
    return f"{assessment['case']}_{assessment['examiner']}_{assessment['mode']}_T{assessment['round']}"


def drafts():
    """Attached files not saved yet, keyed by assessment, site and stored trial number."""
    return st.session_state.setdefault('drafts', {})


def picks():
    """Findings chosen but not saved yet, keyed by their radio key."""
    return st.session_state.setdefault('picks', {})


def draft_key(assessment, site, number):
    return f'{assessment_tag(assessment)}_{site}_{number}'


def keep_pick(key):
    picks()[key] = st.session_state.get(key)


def restore_pick(key):
    """Put a chosen-but-unsaved finding back before its radio is drawn again."""
    if key not in st.session_state and picks().get(key) is not None:
        st.session_state[key] = picks()[key]


def unsaved_sites(assessment, rows):
    """Sites of this assessment holding an attached file or a chosen finding that is not saved."""
    prefix, found = assessment_tag(assessment) + '_', []
    for site in SITE_ORDER:
        row = rows.get(site)
        if any(k.startswith(f'{prefix}{site}_') for k in drafts()):
            found.append(site)
        elif row and ((not row['manual_locked_at'] and picks().get(f"initial_{row['trial_id']}")) or
                      (row['manual_locked_at'] and not row['assisted_label'] and picks().get(f"final_{row['trial_id']}"))):
            found.append(site)
    return found


def _decoded_copy(store, data):
    """Write bytes to a private temp file and return (raw path, decodable path)."""
    with tempfile.NamedTemporaryFile(dir=store.root, suffix='.upload', delete=False) as handle:
        handle.write(data)
        raw = Path(handle.name)
    return raw, decodable(raw)


def attach_file(store, name, data):
    """Check an attached file once. The result stays with the site until it is saved or replaced."""
    raw, readable = _decoded_copy(store, data)
    try:
        return dict(name=name, data=data, quality=check_audio(readable))
    finally:
        readable.unlink(missing_ok=True)
        raw.unlink(missing_ok=True)


def save_draft(store, assessment, site, number, draft):
    """Store an attached file as a trial. Saving the same file twice keeps one record."""
    raw, readable = _decoded_copy(store, draft['data'])
    try:
        return store.create_trial(readable, assessment['case'], site, number, assessment['mode'],
                                  assessment['examiner'], unique=True)
    finally:
        readable.unlink(missing_ok=True)
        raw.unlink(missing_ok=True)


def record_place(store):
    return 'the team folder' if isinstance(store, DriveStore) else 'the study record'


def saving_words(store, what):
    return f'Saving {what} to {record_place(store)}...'


def _mark_busy(key):
    st.session_state['busy'] = key


def action_button(label, key, disabled=False):
    """Primary button that greys out while its save runs, so a second tap cannot repeat it.

    Returns True on the run that should do the work. The busy mark lasts one run only, so an
    interrupted save never leaves the button stuck.
    """
    running = st.session_state.get('busy_now') == key
    st.button(label, type='primary', key=key, disabled=disabled or running, on_click=_mark_busy, args=(key,))
    return running


def fail(site, key, message, kind='retry'):
    """Keep a save error for the next run, where the site's panel shows it beside a ready button."""
    st.session_state['failed'] = dict(site=site, key=key, message=message, kind=kind)
    st.rerun()


def take_failure(site):
    """The last save error for this site, shown once under the status line."""
    failed = st.session_state.get('failed')
    if failed and failed['site'] == site:
        return st.session_state.pop('failed')
    return None


def is_conflict(error):
    """A clash with what is already stored; pressing the button again cannot help."""
    return str(error).startswith(('Final finding is already signed', 'Manual label is locked'))


def prune_saved(assessment, rows):
    """Forget drafts and picks whose save already finished, for example after a tap interrupted the run."""
    prefix = assessment_tag(assessment) + '_'
    for key in [k for k in drafts() if k.startswith(prefix)]:
        code, _, number = key[len(prefix):].partition('_')
        row = rows.get(code)
        if row and str(parse_trial(row['trial_id'])['number']) == number:
            del drafts()[key]
    for row in rows.values():
        if row['manual_locked_at']:
            picks().pop(f"initial_{row['trial_id']}", None)
        if row['assisted_label']:
            picks().pop(f"final_{row['trial_id']}", None)


def held_anywhere():
    """Any attached file or chosen finding not saved, in any assessment of this session."""
    return bool(drafts()) or any(v is not None for v in picks().values())


def flash(message):
    st.session_state['flash'] = message


def flash_text():
    """The last save's confirmation, shown once."""
    return escape(st.session_state.pop('flash', '') or '')


PLAIN_ERRORS = {
    'Recording exists': 'A different file is already stored for this recording at this site.',
    'Final finding is already signed': 'This site was already signed with another finding, perhaps from another screen.',
    'Manual label is locked': 'A different first finding was already saved for this site, perhaps from another screen.',
}


def plain_error(error):
    """Storage messages in examiner words. Unknown messages pass through unchanged."""
    text = str(error)
    if text.startswith('This file is already saved as '):
        rid = text.removeprefix('This file is already saved as ').split('.')[0]
        try:
            _, code, attempt, _ = rid.split('_')
            where = (f"{'front' if code.startswith('A') else 'back'}, {SITE_NAMES[code].lower()}, "
                     f'{recording_words(Store.base_trial(int(attempt[1:])))}')
        except (ValueError, KeyError):
            return text
        return f'This file is already saved for {where}.'
    return next((words for start, words in PLAIN_ERRORS.items() if text.startswith(start)), text)
