"""The tappable chest drawing (D-071). Site positions follow Section 5 of the recording protocol."""
import streamlit as st

WIDTH, HEIGHT = 400, 470
# Marker centre, then label offset and anchor, in drawing units.
SITES = {
    'ARU': (132, 140, 0, 44, 'middle'), 'ALU': (268, 140, 0, 44, 'middle'),
    'ARL': (88, 280, 0, 44, 'middle'), 'ALL': (312, 280, 0, 44, 'middle'),
    'PLU': (166, 150, -22, 5, 'end'), 'PRU': (234, 150, 22, 5, 'start'),
    'PLL': (124, 282, 0, 44, 'middle'), 'PRL': (276, 282, 0, 44, 'middle'),
}
PLACEMENT = {
    'ARU': 'Place it in the 2nd intercostal space, three finger-widths below the mid-collarbone.',
    'ALU': 'Place it in the 2nd intercostal space, three finger-widths below the mid-collarbone.',
    'ARL': 'Place it in line with the front of the armpit, 5th to 6th intercostal space.',
    'ALL': 'Place it in line with the front of the armpit, 5th to 6th intercostal space, away from the heart.',
    'PRU': 'Place it at T3 to T4, between the spine and the shoulder blade.',
    'PLU': 'Place it at T3 to T4, between the spine and the shoulder blade.',
    'PRL': 'Place it at T8 to T9, two finger-widths below the tip of the shoulder blade.',
    'PLL': 'Place it at T8 to T9, two finger-widths below the tip of the shoulder blade.',
}
STATE = {'Not started': 'open', 'Needs finding': 'half', 'Ready to sign': 'half', 'Signed': 'done',
         'Record again': 'again', 'Attached, not saved yet': 'open', 'Second recording needed': 'half'}  # Unsaved work keeps the marker; words say it.

TORSO = ('M166 0C166 30 160 42 132 50C92 60 44 66 26 96C16 116 20 170 30 230C38 300 44 390 52 470L348 470'
         'C356 390 362 300 370 230C380 170 384 116 374 96C356 66 308 60 268 50C240 42 234 30 234 0')
FRONT_RIGHT = ('M146 84C120 88 92 130 78 186C66 236 56 300 54 362C54 384 68 392 88 384C120 372 156 356 180 348'
               'C188 345 190 336 188 326L184 160C182 120 170 86 146 84Z')
FRONT_LEFT = ('M254 84C280 88 308 130 322 186C334 236 344 300 346 362C346 384 332 392 312 384C290 376 268 368 250 358'
              'C262 340 258 300 232 276C222 266 214 256 214 240L216 160C218 120 230 86 254 84Z')
BACK_LEFT = ('M142 76C116 82 88 128 74 186C62 240 52 320 50 392C50 414 66 420 86 410C120 396 158 388 182 386'
             'C190 385 192 376 192 366L188 150C186 112 168 74 142 76Z')
BACK_RIGHT = ('M258 76C286 82 316 128 330 186C342 240 352 316 354 382C354 404 338 410 318 400C282 386 242 378 218 376'
              'C210 375 208 366 208 356L212 150C214 112 232 74 258 76Z')


HIT = {'middle': 'left:-28px;top:44px;width:104px;height:26px;', 'end': 'left:-84px;top:10px;width:92px;height:28px;',
       'start': 'left:40px;top:10px;width:92px;height:28px;'}


def _ribs(front):
    paths = []
    for i in range(8):
        y = 104 + i * 36
        if front:  # Front ribs descend toward the sternum.
            paths.append(f'M186 {y + 30}C150 {y + 22} 96 {y + 8} 52 {y + 16}')
            paths.append(f'M214 {y + 30}C250 {y + 22} 304 {y + 8} 348 {y + 16}')
        else:  # Back ribs leave the spine and slope down and outward.
            paths.append(f'M190 {y}C150 {y - 6} 90 {y + 6} 48 {y + 44}')
            paths.append(f'M210 {y}C250 {y - 6} 310 {y + 6} 352 {y + 44}')
    return f'<path class="ribs" d="{"".join(paths)}"/>'


def _front():
    return (f'<path class="body" d="{TORSO}"/>{_ribs(True)}'
            '<path class="heart" d="M214 262C236 252 268 276 276 306C282 334 262 362 238 372C214 366 188 346 184 318'
            'C182 292 194 270 214 262Z"/>'
            f'<path class="lung" d="{FRONT_RIGHT}"/><path class="lung" d="{FRONT_LEFT}"/>'
            '<path class="fissure" d="M57 290C84 306 122 334 150 361M186 236C150 240 104 256 70 298"/>'
            '<path class="fissure" d="M340 300C312 312 286 332 268 366"/>'
            '<path class="airway" d="M200 36L200 112M200 112C192 124 184 134 174 146M200 112C208 124 216 134 226 146"/>'
            '<path class="bone" d="M188 74C160 66 112 68 66 86M212 74C240 66 288 68 334 86"/>'
            '<path class="sternum" d="M190 84L210 84L213 150L206 248L194 248L187 150Z"/>')


def _back():
    ticks = ''.join(f'M196 {y}L204 {y + 3}' for y in range(30, 456, 19))
    vertebrae = f'<path class="spine" d="M200 18L200 462"/><path class="spine" d="{ticks}"/>'
    return (f'<path class="body" d="{TORSO}"/>{_ribs(False)}'
            f'<path class="lung" d="{BACK_LEFT}"/><path class="lung" d="{BACK_RIGHT}"/>'
            '<path class="fissure" d="M186 150C150 196 104 250 58 318M214 150C250 196 296 250 342 318"/>'
            f'{vertebrae}'
            '<path class="blade" d="M146 94L54 112C72 158 100 204 126 236C134 190 140 140 146 94Z"/>'
            '<path class="blade" d="M254 94L346 112C328 158 300 204 274 236C266 190 260 140 254 94Z"/>')


def _labels(codes, names):
    out = []
    for code in codes:
        x, y, dx, dy, anchor = SITES[code]
        out.append(f'<text class="site" x="{x + dx}" y="{y + dy}" text-anchor="{anchor}">{names[code]}</text>')
    return ''.join(out)


def _drawing(view, codes, names):
    front = view == 'front'
    left, right = ("Patient's right", "Patient's left") if front else ("Patient's left", "Patient's right")
    title = ('Front of the chest. The patient\'s right is on the left of the drawing.' if front else
             'Back of the chest. The patient\'s left is on the left of the drawing.')
    return (f'<svg class="plate" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{title}">'
            f'{_front() if front else _back()}{_labels(codes, names)}'
            f'<text class="side" x="8" y="{HEIGHT - 10}">{left}</text>'
            f'<text class="side" x="{WIDTH - 8}" y="{HEIGHT - 10}" text-anchor="end">{right}</text></svg>')


def _css(active_view, statuses):
    hidden = 'back' if active_view == 'front' else 'front'
    rules = [f'.st-key-pick_{c}{{left:{x / WIDTH:.3%};top:{y / HEIGHT:.3%};}}'
             for c, (x, y, *_) in SITES.items()]
    # The written site name is part of the tap area.
    rules += [f'.st-key-pick_{c} button::after{{{HIT[anchor]}}}' for c, (_, _, _, _, anchor) in SITES.items()]
    for code, status in statuses.items():
        state = STATE[status]
        if state == 'half':
            rules.append(f'.st-key-pick_{code} button::before{{background:linear-gradient(90deg,var(--ink) 50%,#fff 50%)!important;}}')
        elif state == 'done':
            rules.append(f'.st-key-pick_{code} button::before{{content:"\\2713";color:#fff;background:var(--ink)!important;}}')
        elif state == 'again':
            rules.append(f'.st-key-pick_{code} button::before{{content:"!";color:var(--problem);border-color:var(--problem)!important;}}')
    return (f'<style>{"".join(rules)}@media (max-width:899px){{.st-key-charts>[data-testid="stLayoutWrapper"]:has(.st-key-chart_{hidden}){{display:none!important;}}}}'
            '@media (min-width:900px){[data-testid="stLayoutWrapper"]:has(>.st-key-view_switch),'
            '.st-key-view_switch{display:none!important;}}</style>')


def chest_chart(groups, names, statuses, active):
    """Draw both views with one real button per site. Returns the tapped site code or None."""
    view = st.session_state.get('chart_view') or ('front' if active.startswith('A') else 'back')
    with st.container(key='view_switch', horizontal=True):
        for value, word in (('front', 'Front'), ('back', 'Back')):
            if st.button(word, key=f'view_{value}', type='primary' if view == value else 'secondary'):
                st.session_state['chart_view'] = value
                st.rerun()
    st.html(_css(view, statuses))
    picked = None
    with st.container(key='charts', horizontal=True):
        for (_, codes), key, word in zip(groups, ('front', 'back'), ('Front', 'Back')):
            with st.container(key=f'chart_{key}'):
                st.markdown(_drawing(key, codes, names), unsafe_allow_html=True)
                for code in codes:
                    label = f'{word}, {names[code].lower()}: {statuses[code].lower()}' + (', selected' if code == active else '')
                    if st.button(label, key=f'pick_{code}', type='primary' if code == active else 'secondary'):
                        picked = code
    st.html('<div class="legend"><span><i class="mk open"></i>Not started</span>'
            '<span><i class="mk half"></i>Started, not signed</span><span><i class="mk done">&#10003;</i>Signed</span>'
            '<span><i class="mk again">!</i>Record again</span></div>')
    return picked
