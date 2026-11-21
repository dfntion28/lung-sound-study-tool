import fresh_modules

fresh_modules.refresh()

import streamlit as st

from workflow_ui import HEADING_SLOT, PAGES, first_heading, frame

# Entry script (D-112): Streamlit Cloud runs this file. It draws the masthead once and runs the chosen page,
# so switching screens keeps the masthead in place instead of clearing the whole page.
page = st.navigation([st.Page(path, title=title, url_path=url, default=url is None) for path, title, url in PAGES],
                     position='hidden')
frame(page.title)
# Continuity, not decoration: the new page body fades in once when the screen changes, never on an ordinary rerun.
# Content stays visible by default; reduced motion turns it off.
# The style element is written on every run so the page body keeps its position and Streamlit updates it in place.
turn = st.session_state.get('page_turn', 0)
if st.session_state.get('page_shown') != page.title:
    turn += 1
    st.session_state['page_shown'], st.session_state['page_turn'] = page.title, turn
st.html(f'<style>@keyframes arrive{turn}{{from{{opacity:.45;}}to{{opacity:1;}}}}'
        f'.st-key-page_body{{animation:arrive{turn} .2s cubic-bezier(.16,1,.3,1);}}'
        '@media (prefers-reduced-motion:reduce){.st-key-page_body{animation:none!important;}}</style>')
# A page switch removes every element the old page drew at once. The heading is drawn here, by the entry script, so
# it stays until the new heading replaces it in place and the body is never empty (workflow_ui.heading).
with st.container(key='page_body'):
    slot, text = st.empty(), first_heading(page.title)
    slot.title(text)
    st.session_state[HEADING_SLOT] = (slot, text)
    try:
        page.run()
    finally:
        st.session_state.pop(HEADING_SLOT, None)
