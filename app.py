"""
RoadSight AI — a presentation-style Streamlit showcase for a YOLO pothole detector.

Sections (top navigation):
  I   Purpose        – why this exists, told before anything else
  II  Live Demo      – upload a road image, read the detections
  III Method         – pipeline, label format, stack
  IV  Evidence       – real training artifacts, if present beside the app
  V   Responsible    – limits and next steps

Everything shown is either illustration (clearly captioned) or read from your
real files. No metrics are invented.
"""
import base64
import csv
import io
import time
from pathlib import Path
from textwrap import dedent

import streamlit as st
from PIL import Image

# ----------------------------------------------------------------------------
# Personalise here (leave "" to hide)
# ----------------------------------------------------------------------------
PROJECT_AUTHOR = "Saloni Maijorwar"          # e.g. "Your Name"
PROJECT_CONTEXT = "Pothole-Detection Project 2026"         # e.g. "Computer Vision project · 2026"       # e.g. "Computer Vision project · 2026"

ROOT = Path(__file__).resolve().parent
RUN_DIR = Path("results") / "yolov11n_250_epochs_100_patience_reduced_dataset" / "detect" / "train"

st.set_page_config(
    page_title="RoadSight AI — Pothole Detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500;1,600&family=Source+Sans+3:wght@400;500;600;700&display=swap');

:root{
  --paper:#f4efe4; --card:#fbf8f1; --ink:#1a1d21; --muted:#6b6558;
  --brass:#a9761b; --brass-soft:#d9b25f; --rule:#d9d1bf; --deep:#16201f;
}
html, body, [class*="css"], .stApp { font-family:'Source Sans 3',system-ui,sans-serif; color:var(--ink); font-size:18px; }
.stApp p, .stApp li, .stApp label, .stApp span { font-size:1.02em; }
.stApp { background:var(--paper); }
#MainMenu, footer, header, div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display:none !important; }
.block-container { max-width:1560px; padding:1.6rem 3.2rem 3.5rem; }

h1,h2,h3,.serif { font-family:'Cormorant Garamond',Georgia,serif !important; letter-spacing:-.01em; color:var(--ink); }

/* ---------- top bar ---------- */
.brand { font-family:'Cormorant Garamond',Georgia,serif; font-size:2.2rem; font-weight:700; letter-spacing:.01em; line-height:1; padding-top:.35rem; }
.brand i { color:var(--brass); font-style:normal; margin-right:.35rem; }
.brand span { font-weight:500; font-style:italic; color:var(--muted); }
div[data-testid="stRadio"] > label { display:none; }
div[role="radiogroup"] { gap:.2rem 1.6rem; justify-content:flex-end; flex-wrap:wrap; }
div[role="radiogroup"] label { padding:.35rem 0 .45rem !important; margin:0 !important; border-bottom:2px solid transparent; cursor:pointer; background:transparent !important; }
label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) { display:none !important; }
div[role="radiogroup"] label p { font-size:.95rem !important; letter-spacing:.14em; text-transform:uppercase; font-weight:600; color:var(--muted); }
div[role="radiogroup"] label:hover p { color:var(--ink); }
div[role="radiogroup"] label:has(input:checked) { border-bottom-color:var(--brass); }
div[role="radiogroup"] label:has(input:checked) p { color:var(--ink); }
.rule { border:none; border-top:1px solid var(--rule); margin:.7rem 0 2rem; }
.rule-double { border:none; border-top:3px double var(--rule); margin:2.6rem 0; }

/* ---------- type ---------- */
.kicker { font-size:.9rem; font-weight:700; letter-spacing:.2em; text-transform:uppercase; color:var(--brass); margin-bottom:.7rem; }
.h-display { font-family:'Cormorant Garamond',Georgia,serif; font-size:clamp(3.4rem,6.2vw,6.2rem); line-height:.98; font-weight:600; letter-spacing:-.02em; margin:0 0 1.2rem; }
.h-display em { color:var(--brass); font-weight:500; }
.h-section { font-family:'Cormorant Garamond',Georgia,serif; font-size:clamp(2.5rem,4vw,3.7rem); line-height:1.05; font-weight:600; margin:0 0 .8rem; }
.h-section em { color:var(--brass); }
.lede { font-size:1.4rem; line-height:1.65; color:#2e2c27; max-width:680px; }
.body { font-size:1.2rem; line-height:1.7; color:#3d3a33; }
.quote { font-family:'Cormorant Garamond',Georgia,serif; font-size:clamp(2rem,3.4vw,3.2rem); line-height:1.25; font-style:italic; font-weight:500; max-width:1150px; margin:0 auto; text-align:center; }
.quote b { font-style:normal; color:var(--brass); }

/* ---------- plates / cards ---------- */
.plate { background:var(--card); border:1px solid var(--rule); padding:10px; box-shadow:0 18px 40px -22px rgba(26,29,33,.45); }
.plate img { width:100%; display:block; }
.caption { font-size:1rem; color:var(--muted); font-style:italic; margin-top:.55rem; line-height:1.5; }
.caption b { font-style:normal; color:var(--ink); letter-spacing:.08em; text-transform:uppercase; font-size:.82rem; margin-right:.4rem; }
.col-card { border-top:2px solid var(--ink); padding-top:1rem; }
.col-card .num { font-family:'Cormorant Garamond',serif; font-size:3.2rem; color:var(--brass); line-height:1; }
.col-card h3 { font-size:2.1rem; margin:.3rem 0 .4rem; }
.col-card p { color:#3d3a33; line-height:1.7; font-size:1.15rem; margin:0; }
.step { background:var(--card); border:1px solid var(--rule); padding:1.5rem 1.3rem 1.4rem; height:100%; }
.step img { width:96px; height:96px; margin-bottom:.9rem; }
.step .n { font-size:.85rem; letter-spacing:.2em; font-weight:700; color:var(--brass); }
.step h3 { font-size:2rem; margin:.25rem 0 .4rem; }
.step p { font-size:1.1rem; line-height:1.65; color:#3d3a33; margin:0; }
.band { background:var(--deep); color:#f1ece0; padding:4rem 2.4rem; text-align:center; position:relative; overflow:hidden; }
.band h2 { color:#f6f1e4 !important; font-size:clamp(2rem,3.6vw,3rem); margin:0 0 .6rem; font-weight:600; }
.band p { color:#ddd7c6; font-size:1.2rem; max-width:720px; margin:0 auto 0; line-height:1.7; }
.band:before { content:""; position:absolute; inset:10px; border:1px solid rgba(217,178,95,.35); pointer-events:none; }

/* ---------- stat tiles ---------- */
.tile { border:1px solid var(--rule); background:var(--card); padding:1.1rem 1.2rem; }
.tile .v { font-family:'Cormorant Garamond',serif; font-size:3.6rem; font-weight:600; line-height:1; }
.tile .l { font-size:.85rem; letter-spacing:.18em; text-transform:uppercase; color:var(--muted); font-weight:600; margin-top:.4rem; }
.bar-row { display:flex; align-items:center; gap:1rem; margin:.7rem 0; font-size:1.1rem; color:#4a463d; }
.bar-row .lab { width:130px; letter-spacing:.06em; }
.bar-row .track { flex:1; height:12px; background:#e8e1d0; }
.bar-row .fill { height:100%; background:linear-gradient(90deg,var(--brass-soft),var(--brass)); }
.bar-row .pct { width:64px; text-align:right; font-weight:600; color:var(--ink); }

/* ---------- streamlit widgets ---------- */
.stButton > button { border-radius:0; border:1px solid var(--ink); background:var(--ink); color:#f6f1e4; font-weight:600; letter-spacing:.14em; text-transform:uppercase; font-size:.95rem; padding:1rem 2.2rem; transition:all .2s; }
.stButton > button:hover { background:var(--brass); border-color:var(--brass); color:#fff; }
.stButton > button:focus { box-shadow:none; color:#fff; }
.band + div .stButton > button, .on-dark .stButton > button { background:var(--brass-soft); border-color:var(--brass-soft); color:var(--deep); }
.stDownloadButton > button { border-radius:0; border:1px solid var(--ink); background:transparent; color:var(--ink); font-weight:600; letter-spacing:.14em; text-transform:uppercase; font-size:.92rem; padding:.85rem 1.7rem; }
.stDownloadButton > button:hover { background:var(--ink); color:#f6f1e4; border-color:var(--ink); }
div[data-testid="stFileUploader"] section { background:var(--card); border:1px dashed #b9ae94; border-radius:0; padding:1.2rem; }
div[data-testid="stFileUploader"] small { color:var(--muted); }
div[data-testid="stImage"] img { border:1px solid var(--rule); background:var(--card); padding:6px; box-shadow:0 18px 40px -24px rgba(26,29,33,.5); }
div[data-testid="stSlider"] [role="slider"] { background:var(--brass) !important; }
div[data-testid="stExpander"] { border:1px solid var(--rule); border-radius:0; background:var(--card); }
div[data-testid="stSelectbox"] > div > div { border-radius:0; background:var(--card); }
.note { font-size:1rem; color:var(--muted); line-height:1.6; border-left:2px solid var(--brass-soft); padding-left:.9rem; }
.footer { text-align:center; color:var(--muted); font-size:1rem; border-top:1px solid var(--rule); padding-top:1.4rem; margin-top:3.2rem; line-height:1.7; }
.footer .serif { font-size:1.7rem; color:var(--ink); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def html(block: str):
    """Render HTML without Markdown turning indented lines into code blocks."""
    cleaned = "\n".join(l.strip() for l in dedent(block).splitlines() if l.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def show_image(img):
    """Works on both older and newer Streamlit."""
    try:
        st.image(img, use_container_width=True)
    except TypeError:
        st.image(img, use_column_width=True)


def goto(page: str):
    st.session_state.page = page


def heading(kicker: str, title: str, copy: str = ""):
    html(f'<div class="kicker">{kicker}</div><div class="h-section">{title}</div>')
    if copy:
        html(f'<div class="body" style="max-width:900px">{copy}</div>')


# ----------------------------------------------------------------------------
# Illustrations (drawn in code, so they ship with the app — no image files needed)
# ----------------------------------------------------------------------------
_PALETTES = {
    "day":      dict(top="#e9d9b6", bot="#f6ecd6", ground="#b9ad87", hill="#c9b98f", hill2="#b5a47a", road="#3d4248", road2="#50565d", rim="#767d84", hole="#1a1d20", line="#f1d684", tree="#7f8766", sun="#f7dc8f", stars=False),
    "dusk":     dict(top="#5d4f73", bot="#f0b48a", ground="#6e5a52", hill="#7a6070", hill2="#5f4a5e", road="#2f3036", road2="#3d3d45", rim="#5d5a62", hole="#131316", line="#e9c46a", tree="#4a3d4d", sun="#ffd9a0", stars=False),
    "overcast": dict(top="#c9cdcd", bot="#eceeea", ground="#a3a898", hill="#aab1ab", hill2="#98a09b", road="#464b50", road2="#585e63", rim="#80868b", hole="#202326", line="#e7e0c0", tree="#7c857a", sun="#f4f4f0", stars=False),
    "night":    dict(top="#0e1522", bot="#2b3a52", ground="#1a2029", hill="#212c3c", hill2="#19222f", road="#1d2126", road2="#2a3037", rim="#454c55", hole="#07080a", line="#c9b26a", tree="#141a24", sun="#e8e4d0", stars=True),
}
_HOLES = {
    "day":      [(0.86, -0.38, 1.0), (0.52, 0.32, 0.9), (0.30, -0.12, 0.8), (0.15, 0.42, 0.8)],
    "dusk":     [(0.80, 0.30, 1.0), (0.46, -0.40, 0.95), (0.27, 0.15, 0.8)],
    "overcast": [(0.88, 0.05, 1.05), (0.50, -0.45, 0.85), (0.24, 0.35, 0.8)],
    "night":    [(0.78, -0.25, 1.0), (0.44, 0.38, 0.9), (0.22, -0.30, 0.8)],
}


def scene_svg(variant: str = "day", boxes: bool = False) -> str:
    p = _PALETTES[variant]
    hz, vx = 212, 400
    half = lambda t: 20 + 540 * t
    ypos = lambda t: hz + 288 * t
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500" preserveAspectRatio="xMidYMid slice">']
    out.append(f'''<defs>
      <linearGradient id="sky{variant}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{p['top']}"/><stop offset="1" stop-color="{p['bot']}"/></linearGradient>
      <linearGradient id="rd{variant}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{p['road2']}"/><stop offset="1" stop-color="{p['road']}"/></linearGradient>
      <radialGradient id="sun{variant}"><stop offset="0" stop-color="{p['sun']}" stop-opacity=".95"/><stop offset="1" stop-color="{p['sun']}" stop-opacity="0"/></radialGradient>
      <linearGradient id="vg{variant}" x1="0" y1="0" x2="0" y2="1"><stop offset=".6" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".28"/></linearGradient>
    </defs>''')
    out.append(f'<rect width="800" height="500" fill="url(#sky{variant})"/>')
    if p["stars"]:
        for sx, sy, sr in [(90, 40, 1.4), (180, 90, 1), (300, 30, 1.2), (520, 60, 1.3), (640, 25, 1), (720, 95, 1.4), (420, 110, 1)]:
            out.append(f'<circle cx="{sx}" cy="{sy}" r="{sr}" fill="#fff" opacity=".7"/>')
    sx = 575 if variant != "dusk" else 330
    out.append(f'<circle cx="{sx}" cy="150" r="110" fill="url(#sun{variant})"/><circle cx="{sx}" cy="150" r="{18 if variant!="night" else 14}" fill="{p["sun"]}"/>')
    out.append(f'<path d="M0,{hz} C90,150 190,196 300,176 S520,150 640,184 S760,168 800,178 L800,{hz+6} L0,{hz+6}Z" fill="{p["hill2"]}"/>')
    out.append(f'<path d="M0,{hz} C120,176 240,205 360,190 S600,172 800,200 L800,{hz+6} L0,{hz+6}Z" fill="{p["hill"]}"/>')
    out.append(f'<rect y="{hz}" width="800" height="{500-hz}" fill="{p["ground"]}"/>')
    # roadside trees
    for tx, ty, s in [(60, 215, 1.0), (140, 222, 1.3), (230, 214, .8), (300, 212, .6), (500, 212, .6), (570, 214, .8), (660, 222, 1.3), (745, 216, 1.0)]:
        out.append(f'<rect x="{tx-1.5*s}" y="{ty}" width="{3*s}" height="{12*s}" fill="{p["tree"]}"/><ellipse cx="{tx}" cy="{ty-6*s}" rx="{13*s}" ry="{19*s}" fill="{p["tree"]}"/>')
    # road
    out.append(f'<polygon points="{vx-20},{hz} {vx+20},{hz} {vx+half(1)},500 {vx-half(1)},500" fill="url(#rd{variant})"/>')
    for sgn in (-1, 1):
        out.append(f'<line x1="{vx+sgn*21}" y1="{hz}" x2="{vx+sgn*(half(1)-14)}" y2="500" stroke="{p["line"]}" stroke-width="2.2" opacity=".55"/>')
    # lane dashes
    for k in range(9):
        ta, tb = (k / 9) ** 2, ((k + .5) / 9) ** 2
        wa, wb = 1.2 + 9 * ta, 1.2 + 9 * tb
        out.append(f'<polygon points="{vx-wa},{ypos(ta)} {vx+wa},{ypos(ta)} {vx+wb},{ypos(tb)} {vx-wb},{ypos(tb)}" fill="{p["line"]}" opacity=".9"/>')
    # hairline cracks
    for (t, o, dx, dy) in [(0.62, -0.6, 40, 10), (0.4, 0.55, -26, 8), (0.9, 0.62, -60, -12), (0.72, -0.05, 34, -6)]:
        cx, cy = vx + o * half(t), ypos(t)
        out.append(f'<path d="M{cx},{cy} l{dx*.4},{dy*.5} l{dx*.3},{-dy*.3} l{dx*.3},{dy*.8}" stroke="#000" stroke-opacity=".28" fill="none" stroke-width="{.6+1.2*t}"/>')
    # potholes (+ optional detection boxes)
    for i, (t, o, s) in enumerate(_HOLES[variant]):
        cx, cy = vx + o * half(t), ypos(t)
        rx = .13 * half(t) * s
        ry = rx * .3
        out.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx*1.18}" ry="{ry*1.25}" fill="{p["rim"]}" opacity=".85"/>')
        out.append(f'<ellipse cx="{cx}" cy="{cy+ry*.1}" rx="{rx}" ry="{ry}" fill="{p["hole"]}"/>')
        out.append(f'<path d="M{cx-rx*.8},{cy+ry*.35} Q{cx},{cy+ry*1.05} {cx+rx*.8},{cy+ry*.35}" stroke="#fff" stroke-opacity=".12" fill="none" stroke-width="{max(1,rx*.06)}"/>')
        if boxes:
            bw, bh = rx * 2.7, ry * 5.0
            bx, by = cx - bw / 2, cy - bh * .58
            sw = max(1.6, 4.2 * t + 1)
            out.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="none" stroke="#d99a1e" stroke-width="{sw}"/>')
            if t > .4:
                fs = 8 + 10 * t
                tw = fs * 4.6
                out.append(f'<rect x="{bx-sw/2}" y="{by-fs*1.45}" width="{tw}" height="{fs*1.45}" fill="#d99a1e"/><text x="{bx+fs*.4}" y="{by-fs*.42}" font-family="Source Sans 3,Arial,sans-serif" font-size="{fs}" font-weight="700" fill="#16201f" letter-spacing=".5">pothole</text>')
    out.append(f'<rect width="800" height="500" fill="url(#vg{variant})"/></svg>')
    return "".join(out)


def report_svg() -> str:
    rows = [("Detection 1", True), ("Detection 2", True), ("Detection 3", False)]
    r = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500"><rect width="800" height="500" fill="#efe7d4"/>']
    r.append('<rect x="130" y="50" width="540" height="400" fill="#fbf8f1" stroke="#d9d1bf" stroke-width="2"/>')
    r.append('<text x="170" y="112" font-family="Cormorant Garamond,Georgia,serif" font-size="34" font-weight="600" fill="#1a1d21">Review queue</text>')
    r.append('<line x1="170" y1="130" x2="630" y2="130" stroke="#1a1d21" stroke-width="2"/>')
    for i, (lab, ok) in enumerate(rows):
        y = 175 + i * 80
        r.append(f'<rect x="170" y="{y}" width="100" height="56" fill="#cdbf9b"/><rect x="190" y="{y+22}" width="60" height="16" fill="#1a1d20"/><rect x="184" y="{y+14}" width="72" height="30" fill="none" stroke="#d99a1e" stroke-width="2.5"/>')
        r.append(f'<rect x="295" y="{y+8}" width="150" height="9" fill="#8d8571"/><rect x="295" y="{y+30}" width="100" height="9" fill="#c8bfa8"/>')
        if ok:
            r.append(f'<circle cx="590" cy="{y+28}" r="20" fill="none" stroke="#a9761b" stroke-width="3"/><path d="M580,{y+28} l8,9 l15,-18" stroke="#a9761b" stroke-width="4" fill="none"/>')
        else:
            r.append(f'<circle cx="590" cy="{y+28}" r="20" fill="none" stroke="#9a9484" stroke-width="3" stroke-dasharray="5 4"/><text x="582" y="{y+38}" font-family="Georgia,serif" font-size="28" fill="#9a9484">?</text>')
    r.append('<text x="170" y="425" font-family="Source Sans 3,Arial,sans-serif" font-size="15" fill="#6b6558" letter-spacing="1.5">A PERSON CONFIRMS EVERY FLAG</text></svg>')
    return "".join(r)


def icon_svg(kind: str) -> str:
    b, ink = "#a9761b", "#1a1d21"
    g = {
        "data": f'<rect x="18" y="30" width="44" height="34" fill="none" stroke="{ink}" stroke-width="2.5"/><rect x="28" y="22" width="44" height="34" fill="#f4efe4" stroke="{ink}" stroke-width="2.5"/><rect x="38" y="14" width="44" height="34" fill="#f4efe4" stroke="{ink}" stroke-width="2.5"/><rect x="50" y="26" width="20" height="14" fill="none" stroke="{b}" stroke-width="3"/>',
        "train": f'<g stroke="{ink}" stroke-width="1.8" opacity=".6"><line x1="22" y1="30" x2="48" y2="22"/><line x1="22" y1="30" x2="48" y2="48"/><line x1="22" y1="30" x2="48" y2="74"/><line x1="22" y1="62" x2="48" y2="22"/><line x1="22" y1="62" x2="48" y2="48"/><line x1="22" y1="62" x2="48" y2="74"/><line x1="48" y1="22" x2="76" y2="48"/><line x1="48" y1="48" x2="76" y2="48"/><line x1="48" y1="74" x2="76" y2="48"/></g><g fill="#f4efe4" stroke="{ink}" stroke-width="2.5"><circle cx="22" cy="30" r="7"/><circle cx="22" cy="62" r="7"/><circle cx="48" cy="22" r="7"/><circle cx="48" cy="48" r="7"/><circle cx="48" cy="74" r="7"/></g><circle cx="76" cy="48" r="8" fill="{b}" stroke="{b}"/>',
        "eval": f'<path d="M18 18 V76 H80" fill="none" stroke="{ink}" stroke-width="2.5"/><path d="M22 70 C34 68 36 30 50 28 S68 26 78 24" fill="none" stroke="{b}" stroke-width="3.5"/><path d="M22 70 C36 66 44 52 58 48 S72 46 78 44" fill="none" stroke="{ink}" stroke-width="2" stroke-dasharray="4 4" opacity=".6"/>',
        "infer": f'<rect x="14" y="26" width="52" height="40" fill="#f4efe4" stroke="{ink}" stroke-width="2.5"/><ellipse cx="40" cy="52" rx="12" ry="4" fill="{ink}"/><rect x="24" y="38" width="32" height="22" fill="none" stroke="{b}" stroke-width="3"/><circle cx="62" cy="40" r="15" fill="#f4efe480" stroke="{ink}" stroke-width="3"/><line x1="73" y1="51" x2="84" y2="64" stroke="{ink}" stroke-width="5" stroke-linecap="round"/>',
        "video": f'<rect x="14" y="26" width="52" height="44" fill="#f4efe4" stroke="{ink}" stroke-width="2.5"/><path d="M66 40 L84 30 V66 L66 56 Z" fill="none" stroke="{ink}" stroke-width="2.5"/><path d="M33 38 L33 58 L50 48 Z" fill="{b}"/>',
        "pin": f'<path d="M48 80 C30 58 24 48 24 38 a24 24 0 0 1 48 0 C72 48 66 58 48 80Z" fill="#f4efe4" stroke="{ink}" stroke-width="2.5"/><circle cx="48" cy="38" r="9" fill="{b}"/>',
        "check": f'<rect x="22" y="16" width="52" height="66" fill="#f4efe4" stroke="{ink}" stroke-width="2.5"/><rect x="36" y="10" width="24" height="12" fill="{ink}"/><g stroke="{ink}" stroke-width="2.5"><line x1="52" y1="40" x2="66" y2="40"/><line x1="52" y1="56" x2="66" y2="56"/><line x1="52" y1="72" x2="66" y2="72"/></g><g fill="none" stroke="{b}" stroke-width="3"><path d="M29 40 l4 4 l7 -9"/><path d="M29 56 l4 4 l7 -9"/><path d="M29 72 l4 4 l7 -9"/></g>',
    }[kind]
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96"><circle cx="48" cy="48" r="46" fill="#efe7d4" stroke="#d9d1bf" stroke-width="2"/>{g}</svg>'


def plate(svg: str, label: str, caption: str):
    html(f'<div class="plate"><img src="{svg_uri(svg)}"/></div><div class="caption"><b>{label}</b>{caption}</div>')


# ----------------------------------------------------------------------------
# Model + data helpers
# ----------------------------------------------------------------------------
def find_model():
    for c in (ROOT / "best.pt", ROOT / "models" / "best.pt", ROOT / RUN_DIR / "weights" / "best.pt"):
        if c.is_file():
            return c
    try:
        return next(ROOT.rglob("best.pt"))
    except StopIteration:
        return None


@st.cache_resource(show_spinner="Loading the detection model…")
def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


@st.cache_data(show_spinner=False, max_entries=24)
def run_inference(model_path: str, data: bytes, conf: float, labels: bool):
    model = load_model(model_path)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    t0 = time.perf_counter()
    result = model.predict(source=image, conf=conf, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    annotated = Image.fromarray(result.plot(labels=labels)[:, :, ::-1])  # BGR -> RGB
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    scores = sorted((float(c) for c in result.boxes.conf.cpu().tolist()), reverse=True) if result.boxes is not None else []
    return dict(png=buf.getvalue(), scores=scores, ms=ms)


def find_artifact(name: str):
    for d in (ROOT / "assets", ROOT / RUN_DIR, ROOT / "assets" / "results"):
        if (d / name).is_file():
            return d / name
    return None


def read_run_metrics():
    """Real numbers from your own results.csv (best epoch by mAP50-95). Returns None if absent."""
    for d in (ROOT / "assets", ROOT / RUN_DIR, ROOT):
        f = d / "results.csv"
        if f.is_file():
            try:
                with open(f, newline="") as fh:
                    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(fh)]
                key = "metrics/mAP50-95(B)"
                best = max(rows, key=lambda r: float(r[key]))
                return dict(
                    epoch=int(float(best["epoch"])), total=len(rows),
                    precision=float(best["metrics/precision(B)"]), recall=float(best["metrics/recall(B)"]),
                    map50=float(best["metrics/mAP50(B)"]), map5095=float(best[key]),
                )
            except Exception:
                return None
    return None


def list_samples():
    out = []
    for d in (ROOT / "assets" / "samples", ROOT / "samples"):
        if d.is_dir():
            out += sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    return out


# ----------------------------------------------------------------------------
# Top bar / navigation
# ----------------------------------------------------------------------------
PAGES = ["Purpose", "Live Demo", "Method", "Evidence", "Responsible Use"]
ROMAN = dict(zip(PAGES, ["I", "II", "III", "IV", "V"]))
st.session_state.setdefault("page", "Purpose")

c_brand, c_nav = st.columns([1, 2.4], vertical_alignment="center")
with c_brand:
    html('<div class="brand"><i>◈</i>RoadSight <span>AI</span></div>')
with c_nav:
    st.radio("Section", PAGES, key="page", horizontal=True, label_visibility="collapsed",
             format_func=lambda p: f"{ROMAN[p]} · {p}")
html('<hr class="rule">')
page = st.session_state.page


# ----------------------------------------------------------------------------
# I · PURPOSE
# ----------------------------------------------------------------------------
def page_purpose():
    left, right = st.columns([1.02, 1.1], gap="large", vertical_alignment="center")
    with left:
        html('<div class="kicker">A computer-vision study in road safety</div>'
             '<div class="h-display">Finding the pothole <em>before</em> it finds you.</div>'
             '<div class="lede">RoadSight AI is a trained object-detection model that reads a photograph of a road '
             'and marks where it sees potholes. This page tells you why it exists, then lets you try it.</div>')
        st.write("")
        st.button("Open the live detector  →", on_click=goto, args=("Live Demo",), key="cta_top")
    with right:
        plate(scene_svg("day", boxes=True), "Fig. 1", "An illustration of the idea: every flagged region is a candidate for a person to review.")

    html('<hr class="rule-double">')
    html('<div class="quote">Roads are inspected by eye. This project asks whether a model can help <b>point the eye</b> — '
         'quickly, consistently, and at the scale of a whole network.</div>')
    html('<hr class="rule-double">')

    heading("The purpose", "Why this project exists")
    st.write("")
    a, b, c = st.columns(3, gap="large")
    for col, n, t, d in [
        (a, "01", "The problem", "Potholes damage vehicles, disrupt journeys and create safety risks. Finding them still depends largely on people travelling the road and noticing — slow, and uneven in coverage."),
        (b, "02", "The approach", "Train a YOLO11n detector on annotated road images so it learns what a pothole looks like, then run it on new photographs and return boxes with confidence scores."),
        (c, "03", "The intent", "A decision-support tool, not a replacement for inspection: the model proposes, a person disposes. It shortens the search; it does not make the repair decision."),
    ]:
        with col:
            html(f'<div class="col-card"><div class="num">{n}</div><h3>{t}</h3><p>{d}</p></div>')

    html('<hr class="rule-double">')
    heading("How it reads", "From photograph to <em>decision support</em>")
    st.write("")
    f1, f2, f3 = st.columns(3, gap="large")
    with f1:
        plate(scene_svg("overcast"), "Plate I", "A road photograph goes in.")
    with f2:
        plate(scene_svg("overcast", boxes=True), "Plate II", "The model proposes candidate potholes.")
    with f3:
        plate(report_svg(), "Plate III", "A reviewer confirms what matters.")

    html('<hr class="rule-double">')
    heading("The difficulty", "No two roads look alike",
            "Light, shadow, wet surfaces and road texture change how a pothole appears. A useful detector has to cope with that variety — which is why a model trained on many examples is used instead of fixed rules.")
    st.write("")
    g1, g2, g3 = st.columns(3, gap="large")
    with g1:
        plate(scene_svg("day"), "Daylight", "Clear, high contrast.")
    with g2:
        plate(scene_svg("dusk"), "Dusk", "Long shadows, low light.")
    with g3:
        plate(scene_svg("night"), "Night", "Little visible detail.")
    html('<div class="caption" style="margin-top:.9rem">All scenes on this page are illustrations drawn for explanation. They are not outputs of the model; the real detections appear in the Live Demo.</div>')

    st.write("")
    html('<div class="band"><div class="kicker" style="color:#d9b25f">Now see it work</div>'
         '<h2>Give it a road. Read what it finds.</h2>'
         '<p>Upload a photograph in the next section and the trained model will mark potholes, with a confidence score for each.</p></div>')
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        st.write("")
        st.button("Go to the live detector  →", on_click=goto, args=("Live Demo",), key="cta_bottom", use_container_width=True)


# ----------------------------------------------------------------------------
# II · LIVE DEMO
# ----------------------------------------------------------------------------
def page_demo():
    heading("II · The detector", "Upload a road. <em>Read the result.</em>",
            "Choose a photograph, adjust the confidence threshold if you like, and the saved model runs on it immediately.")
    st.write("")

    model_path = find_model()
    samples = list_samples()
    ctl, opt = st.columns([1.35, 1], gap="large")
    with ctl:
        uploaded = st.file_uploader("Road image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    with opt:
        conf = st.slider("Confidence threshold", 0.10, 0.95, 0.35, 0.05,
                         help="Detections below this score are hidden. Lower = more candidates, more false alarms.")
        show_labels = st.checkbox("Show labels on boxes", value=True)
        chosen = None
        if samples:
            pick = st.selectbox("…or use a sample", ["—"] + [s.name for s in samples])
            if pick != "—":
                chosen = next(s for s in samples if s.name == pick)

    data, name = None, None
    if uploaded is not None:
        data, name = uploaded.getvalue(), uploaded.name
    elif chosen is not None:
        data, name = chosen.read_bytes(), chosen.name

    st.write("")
    if data is None:
        a, b = st.columns(2, gap="large")
        with a:
            plate(scene_svg("day"), "Plate A · Input", "Your road image will appear here.")
        with b:
            plate(scene_svg("day", boxes=True), "Plate B · Detections", "Illustration of the expected output — not a real result.")
        if not model_path:
            st.warning("No `best.pt` found. Place your trained checkpoint beside `app.py` to enable detection.")
        return

    if not model_path:
        st.warning("No `best.pt` found. Place your trained checkpoint beside `app.py` to enable detection.")
        show_image(data)
        return

    try:
        with st.spinner("Reading the road…"):
            res = run_inference(str(model_path), data, float(conf), bool(show_labels))
    except Exception as exc:  # keep the demo alive in front of an audience
        st.error(f"Inference failed: {exc}")
        st.caption("Check the image is valid and `best.pt` matches the installed Ultralytics version.")
        return

    a, b = st.columns(2, gap="large")
    with a:
        html('<div class="caption" style="margin:0 0 .5rem"><b>Plate A · Input</b>' + (name or "") + '</div>')
        show_image(data)
    with b:
        html('<div class="caption" style="margin:0 0 .5rem"><b>Plate B · Detections</b>model output</div>')
        show_image(res["png"])

    st.write("")
    n = len(res["scores"])
    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        html(f'<div class="tile"><div class="v">{n}</div><div class="l">Potholes detected</div></div>')
    with m2:
        best = f'{res["scores"][0]:.0%}' if n else "—"
        html(f'<div class="tile"><div class="v">{best}</div><div class="l">Highest confidence</div></div>')
    with m3:
        html(f'<div class="tile"><div class="v">{res["ms"]:.0f}<span style="font-size:1.2rem"> ms</span></div><div class="l">Approx. inference time</div></div>')

    st.write("")
    l, r = st.columns([1.5, 1], gap="large")
    with l:
        if n:
            html('<div class="kicker" style="margin-bottom:.4rem">Confidence by detection</div>')
            rows = "".join(
                f'<div class="bar-row"><div class="lab">Detection {i}</div><div class="track"><div class="fill" style="width:{s*100:.1f}%"></div></div><div class="pct">{s:.0%}</div></div>'
                for i, s in enumerate(res["scores"][:12], 1))
            html(rows)
        else:
            st.info("Nothing found above this threshold. That does not guarantee the road is pothole-free — try lowering the threshold.")
    with r:
        st.download_button("Download annotated image", res["png"], file_name="roadsight_detection.png", mime="image/png")
        html('<div class="note" style="margin-top:1rem">Model output, not a certified safety assessment. Timing varies with hardware and includes first-run warm-up.</div>')


# ----------------------------------------------------------------------------
# III · METHOD
# ----------------------------------------------------------------------------
def page_method():
    heading("III · The method", "A simple pipeline, <em>honestly built</em>",
            "Training and the demo are separate. Training produced the weights; this app only loads them to predict on new images.")
    st.write("")
    cols = st.columns(4, gap="medium")
    for col, (n, t, d, ic) in zip(cols, [
        ("01", "Annotated data", "Road images paired with YOLO-format bounding boxes marking each pothole.", "data"),
        ("02", "Training", "YOLO11n learns the visual features of potholes from the labelled examples.", "train"),
        ("03", "Evaluation", "Validation curves and metrics show how well the detector generalises.", "eval"),
        ("04", "Inference", "The saved checkpoint predicts boxes and confidence scores on new photographs.", "infer"),
    ]):
        with col:
            html(f'<div class="step"><img src="{svg_uri(icon_svg(ic))}"/><div class="n">{n}</div><h3>{t}</h3><p>{d}</p></div>')

    html('<hr class="rule-double">')
    l, r = st.columns([1, 1], gap="large", vertical_alignment="center")
    with l:
        heading("The model", "Why a <em>YOLO</em> detector")
        html('<div class="body">YOLO (“You Only Look Once”) finds and classifies objects in a single pass over the image, '
             'which keeps inference fast. The “n” in YOLO11n is the nano variant — small enough to run quickly on modest hardware, '
             'which suits a prototype that has to respond live in a demo.</div>')
        st.write("")
        with st.expander("What does a YOLO label line mean?"):
            st.markdown("A label line holds five numbers:\n\n`class_id  x_center  y_center  width  height`\n\n"
                        "The four box values are normalised to the image width and height. Here class `0` is `pothole`; each line describes one box.")
    with r:
        plate(scene_svg("overcast", boxes=True), "Fig. 2", "Illustration of a bounding box: a rectangle around a candidate pothole, with a class label and score.")

    html('<hr class="rule-double">')
    heading("The stack", "Three tools, kept small")
    st.write("")
    s1, s2, s3 = st.columns(3, gap="large")
    for col, k, t, d in [
        (s1, "Model", "YOLO11n · Ultralytics", "Object detection and prediction from the saved `best.pt` checkpoint."),
        (s2, "Compute", "PyTorch", "The tensor runtime underneath the detector."),
        (s3, "Interface", "Python · Streamlit", "This page: upload, inference and display, in one Python file."),
    ]:
        with col:
            html(f'<div class="col-card"><div class="kicker" style="margin:0">{k}</div><h3>{t}</h3><p>{d}</p></div>')


# ----------------------------------------------------------------------------
# IV · EVIDENCE
# ----------------------------------------------------------------------------
def page_evidence():
    heading("IV · The evidence", "What the training run <em>actually shows</em>",
            "Figures on this page are read directly from your own training outputs. If they are not beside the app, nothing is shown in their place.")
    st.write("")
    m = read_run_metrics()
    if m:
        c = st.columns(4, gap="medium")
        for col, lab, val in zip(c, ["Precision", "Recall", "mAP@50", "mAP@50–95"], [m["precision"], m["recall"], m["map50"], m["map5095"]]):
            with col:
                html(f'<div class="tile"><div class="v">{val:.3f}</div><div class="l">{lab}</div></div>')
        html(f'<div class="caption" style="margin-top:.7rem">Best epoch by mAP@50–95: epoch {m["epoch"]} of {m["total"]}, read from <i>results.csv</i> (validation set).</div>')
        st.write("")

    arts = [("Training curves", "results.png"), ("Confusion matrix", "confusion_matrix.png"),
            ("Precision–recall curve", "PR_curve.png"), ("F1 curve", "F1_curve.png")]
    found = [(t, find_artifact(f)) for t, f in arts if find_artifact(f)]
    if found:
        cols = st.columns(2, gap="large")
        for i, (t, p) in enumerate(found):
            with cols[i % 2]:
                html(f'<div class="caption" style="margin:1.2rem 0 .5rem"><b>{t}</b></div>')
                show_image(str(p))
    else:
        cols = st.columns(2, gap="large")
        for i, (t, f) in enumerate(arts):
            with cols[i % 2]:
                html(f'<div class="step" style="margin-bottom:1rem"><div class="n">NOT YET ADDED</div><h3>{t}</h3>'
                     f'<p>Copy <code>{f}</code> from your training run into an <code>assets/</code> folder beside <code>app.py</code> and it will appear here.</p></div>')
    if not m:
        html('<div class="note" style="margin-top:1rem">Add <code>results.csv</code> to <code>assets/</code> to show precision, recall and mAP computed from your real run.</div>')


# ----------------------------------------------------------------------------
# V · RESPONSIBLE USE
# ----------------------------------------------------------------------------
def page_responsible():
    heading("V · Responsible use", "A prototype that knows its <em>limits</em>")
    st.write("")
    a, b = st.columns(2, gap="large")
    with a:
        html('<div class="col-card"><div class="kicker" style="margin:0">What it does</div><h3>Flags visible candidates</h3>'
             '<p>Marks regions the model associates with potholes, with a confidence score, for a person to review.</p></div>')
    with b:
        html('<div class="col-card"><div class="kicker" style="margin:0">What it does not do</div><h3>Judge safety or depth</h3>'
             '<p>It does not measure depth, assess structural condition, or decide which road should be repaired. Results depend on image quality, lighting and viewpoint.</p></div>')

    html('<hr class="rule-double">')
    heading("Next", "Where it could go")
    st.write("")
    c = st.columns(3, gap="medium")
    for col, ic, t, d in zip(c, ["video", "pin", "check"], ["Video inference", "Location tagging", "Maintenance workflow"], [
        "Run the detector frame by frame on dashcam-style road video.",
        "Attach GPS where available, with appropriate privacy safeguards.",
        "Let an authorised reviewer verify detections and record follow-up.",
    ]):
        with col:
            html(f'<div class="step"><img src="{svg_uri(icon_svg(ic))}"/><div class="n">FUTURE SCOPE</div><h3>{t}</h3><p>{d}</p></div>')

    html('<div class="footer"><div class="serif">◈ RoadSight AI</div>'
         + (f'Presented by {PROJECT_AUTHOR}' + (f' · {PROJECT_CONTEXT}' if PROJECT_CONTEXT else '') + '<br>' if PROJECT_AUTHOR else '')
         + 'Prototype for demonstrating computer-vision inference. Review detections before any real-world maintenance decision.</div>')


{"Purpose": page_purpose, "Live Demo": page_demo, "Method": page_method,
 "Evidence": page_evidence, "Responsible Use": page_responsible}[page]()

if page != "Responsible Use":
    html('<div class="footer"><div class="serif">◈ RoadSight AI</div>'
         + (f'Presented by {PROJECT_AUTHOR}<br>' if PROJECT_AUTHOR else '')
         + 'Pothole detection showcase · prototype for demonstration</div>')