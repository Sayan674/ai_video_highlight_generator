   
"""
HighlightAI — Streamlit UI
===========================
Visual layer for HighlightAI video highlight generator.
All AI pipeline logic lives in run.py and remains untouched.
"""

import os
import sys
import inspect
import tempfile
import traceback
from pathlib import Path

import streamlit as st
import yaml

try:
    import cv2
except ImportError:
    cv2 = None

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from run import run  # existing pipeline entry point — untouched

PIPELINE_STAGES = [
    ("extract", "Extracting Frames", "Sampling frames from the video with OpenCV."),
    ("models", "Loading AI Models", "Loading ResNet-18, CLIP and Whisper into memory."),
    ("scoring", "Scoring Frames", "Ranking frames by visual and semantic relevance."),
    ("selecting", "Selecting Highlights", "Choosing the strongest moments for the reel."),
    ("whisper", "Whisper Transcription", "Transcribing speech and aligning captions."),
    ("render", "Rendering Final Reel", "Assembling clips and burning in captions."),
]

DIAGRAM_NODES = [
    ("VIDEO", "video"),
    ("VISION AI", "vision"),
    ("SEMANTIC AI", "semantic"),
    ("SPEECH AI", "speech"),
    ("HIGHLIGHT REEL", "reel"),
]

LOGO_SVG = """
<svg width="26" height="26" viewBox="0 0 512 512" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="logoGrad" x1="0" y1="0" x2="512" y2="512" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#6366f1"/>
      <stop offset="1" stop-color="#ec4899"/>
    </linearGradient>
  </defs>
  <path d="M120 110 L340 250 L120 400 Z" stroke="url(#logoGrad)" stroke-width="28" stroke-linejoin="round" stroke-linecap="round" fill="none"/>
  <path d="M366 170 L392 232 L456 250 L392 268 L366 330 L340 268 L276 250 L340 232 Z" fill="url(#logoGrad)"/>
</svg>
"""

NODE_ICONS = {
    "video": """<svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M3 6.5C3 5.67 3.67 5 4.5 5H14.5C15.33 5 16 5.67 16 6.5V17.5C16 18.33 15.33 19 14.5 19H4.5C3.67 19 3 18.33 3 17.5V6.5Z" stroke="white" stroke-width="1.8"/><path d="M16 10L21 7V17L16 14" stroke="white" stroke-width="1.8" stroke-linejoin="round"/></svg>""",
    "vision": """<svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M2 12C2 12 5.5 5.5 12 5.5C18.5 5.5 22 12 22 12C22 12 18.5 18.5 12 18.5C5.5 18.5 2 12 2 12Z" stroke="white" stroke-width="1.8"/><circle cx="12" cy="12" r="3" stroke="white" stroke-width="1.8"/></svg>""",
    "semantic": """<svg width="22" height="22" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="8.5" stroke="white" stroke-width="1.8"/><path d="M8 13.5C8.6 14.6 9.9 15.5 12 15.5C14.5 15.5 16 14 16 12.3C16 10.6 14.7 9.9 12.5 9.4C10.3 8.9 9.3 8.2 9.3 6.9C9.3 5.6 10.6 4.8 12 4.8C13.7 4.8 14.9 5.5 15.5 6.6" stroke="white" stroke-width="1.8" stroke-linecap="round"/></svg>""",
    "speech": """<svg width="22" height="22" viewBox="0 0 24 24" fill="none"><rect x="9" y="3" width="6" height="11" rx="3" stroke="white" stroke-width="1.8"/><path d="M5 11C5 14.87 8.13 18 12 18C15.87 18 19 14.87 19 11" stroke="white" stroke-width="1.8" stroke-linecap="round"/><path d="M12 18V21" stroke="white" stroke-width="1.8" stroke-linecap="round"/></svg>""",
    "reel": """<svg width="22" height="22" viewBox="0 0 24 24" fill="none"><rect x="3" y="5" width="18" height="14" rx="2" stroke="white" stroke-width="1.8"/><path d="M3 9H21" stroke="white" stroke-width="1.8"/><path d="M7 5V9" stroke="white" stroke-width="1.8"/><path d="M13 5V9" stroke="white" stroke-width="1.8"/><path d="M19 5V9" stroke="white" stroke-width="1.8"/></svg>""",
}

STAGE_CHECK = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M5 13L10 18L19 7" stroke="#22c55e" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>"""
STAGE_SPIN = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M21 12a9 9 0 1 1-3-6.7" stroke="#818cf8" stroke-width="2.6" stroke-linecap="round"/></svg>"""


# ---------------------------------------------------------------------------
# Page setup / dark cinematic theme styling
# ---------------------------------------------------------------------------

def setup_page():
    st.set_page_config(
        page_title="HighlightAI — Cinematic AI Video Intelligence",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        #MainMenu, header[data-testid="stHeader"], footer {
            visibility: hidden;
            height: 0;
        }
        section[data-testid="stSidebar"] {
            display: none;
        }

        .stApp {
            background: #07070d;
            color: #e4e4e7;
        }

        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* ---------------- Header ---------------- */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.1rem 1.5rem;
            background: rgba(18, 18, 28, 0.6);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            margin-bottom: 1.8rem;
        }
        .app-header .logo {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-weight: 800;
            font-size: 1.2rem;
            color: #ffffff;
            letter-spacing: -0.02em;
        }
        .app-header .navlinks {
            display: flex;
            gap: 2rem;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .app-header .navlinks a {
            color: #a1a1aa;
            text-decoration: none;
            transition: color 0.2s ease;
        }
        .app-header .navlinks a:hover, .app-header .navlinks a.active {
            color: #ffffff;
        }
        .header-cta {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: #ffffff !important;
            font-weight: 700;
            font-size: 0.85rem;
            padding: 0.6rem 1.2rem;
            border-radius: 10px;
            text-decoration: none !important;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
            transition: all 0.2s ease;
        }
        .header-cta:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        }

        /* ---------------- Hero ---------------- */
        .hero-wrap {
            text-align: center;
            padding: 2.8rem 1rem 2.2rem 1rem;
        }
        .hero-title {
            font-size: 3.4rem;
            font-weight: 800;
            line-height: 1.15;
            color: #ffffff;
            margin: 0;
            letter-spacing: -0.03em;
        }
        .hero-title .grad {
            background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .hero-sub {
            font-size: 1.08rem;
            color: #a1a1aa;
            max-width: 650px;
            margin: 1.2rem auto 1.8rem auto;
            line-height: 1.6;
        }
        .badge-row {
            display: flex;
            gap: 0.6rem;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 2rem;
        }
        .tech-badge {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.25);
            color: #a5b4fc;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.4rem 1rem;
            border-radius: 999px;
        }
        .cta-row {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-bottom: 2.8rem;
        }
        .cta-primary {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: #ffffff !important;
            font-weight: 700;
            font-size: 0.95rem;
            padding: 0.85rem 1.8rem;
            border-radius: 12px;
            text-decoration: none !important;
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
        }
        .cta-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: #e4e4e7 !important;
            font-weight: 700;
            font-size: 0.95rem;
            padding: 0.85rem 1.8rem;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            text-decoration: none !important;
        }

        /* ---------------- Cards & Surfaces ---------------- */
        .surface-card {
            background: rgba(18, 18, 28, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 1.8rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        }
        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 1.2rem;
            letter-spacing: -0.01em;
        }

        /* Upload zone */
        .upload-visual {
            border: 2px dashed rgba(99, 102, 241, 0.35);
            border-radius: 16px;
            background: rgba(99, 102, 241, 0.03);
            text-align: center;
            padding: 2.8rem 1.5rem 2rem 1.5rem;
            transition: border-color 0.2s ease;
        }
        .upload-visual:hover {
            border-color: rgba(99, 102, 241, 0.6);
        }
        .upload-icon-circle {
            width: 58px;
            height: 58px;
            border-radius: 50%;
            background: rgba(99, 102, 241, 0.15);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1.1rem auto;
        }
        .upload-title {
            font-weight: 700;
            font-size: 1.05rem;
            color: #ffffff;
            margin-bottom: 0.3rem;
        }
        .upload-sub {
            font-size: 0.85rem;
            color: #71717a;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }
        [data-testid="stFileUploader"] section {
            padding: 0 !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] {
            display: none;
        }

        .meta-strip {
            display: flex;
            gap: 0.75rem;
            margin-top: 1.2rem;
            flex-wrap: wrap;
        }
        .meta-chip {
            flex: 1;
            min-width: 100px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 0.65rem 0.85rem;
        }
        .meta-chip .lbl {
            font-size: 0.7rem;
            color: #71717a;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .meta-chip .val {
            font-size: 0.9rem;
            color: #ffffff;
            font-weight: 700;
            margin-top: 0.2rem;
            word-break: break-all;
        }

        /* ---------------- Pipeline visual ---------------- */
        .pipeline-card {
            background: rgba(18, 18, 28, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 2.2rem 2rem 1.8rem 2rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        }
        .pipeline-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .pipeline-node {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.7rem;
        }
        .node-circle {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        }
        .node-circle.n0 { background: linear-gradient(135deg, #3b82f6, #6366f1); }
        .node-circle.n1 { background: linear-gradient(135deg, #6366f1, #8b5cf6); }
        .node-circle.n2 { background: linear-gradient(135deg, #8b5cf6, #a855f7); }
        .node-circle.n3 { background: linear-gradient(135deg, #a855f7, #ec4899); }
        .node-circle.n4 {
            background: linear-gradient(135deg, #ec4899, #f43f5e);
            box-shadow: 0 0 25px rgba(244, 63, 94, 0.4);
        }
        .node-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: #a1a1aa;
        }
        .node-connector {
            flex: 1;
            height: 0;
            border-top: 2px dashed rgba(255, 255, 255, 0.15);
            margin: 0 0.8rem 1.8rem 0.8rem;
        }

        /* ---------------- Config panel inputs ---------------- */
        .field-label {
            font-size: 0.85rem;
            font-weight: 700;
            color: #e4e4e7;
            margin: 1rem 0 0.4rem 0;
        }
        .field-row-between {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
        }
        .field-value {
            font-size: 0.85rem;
            font-weight: 700;
            color: #818cf8;
        }
        .helper-caption {
            display: flex;
            justify-content: space-between;
            font-size: 0.72rem;
            color: #71717a;
            margin-top: 0.2rem;
        }

        div[data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 10px !important;
            color: #ffffff !important;
            font-size: 0.88rem !important;
        }

        .stSlider { padding-top: 0.1rem; }
        .stSlider [data-baseweb="slider"] div[role="slider"] {
            background: #6366f1 !important;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2) !important;
        }

        .scoring-bar {
            display: flex;
            height: 10px;
            border-radius: 6px;
            overflow: hidden;
            margin: 0.6rem 0 0.4rem 0;
            background: rgba(255, 255, 255, 0.05);
        }
        .scoring-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            font-weight: 700;
        }
        .scoring-labels .resnet { color: #818cf8; }
        .scoring-labels .clip { color: #c084fc; }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 12px !important;
            background: rgba(255, 255, 255, 0.02) !important;
        }

        /* ---------------- Streamlit Buttons ---------------- */
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: 0.8rem 1.2rem !important;
            border: none !important;
            transition: all 0.2s ease !important;
            width: 100%;
        }
        .stButton > button[kind="primary"], .stDownloadButton > button {
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            color: #ffffff !important;
            box-shadow: 0 8px 22px rgba(99, 102, 241, 0.35) !important;
        }
        .stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
            box-shadow: 0 10px 28px rgba(99, 102, 241, 0.5) !important;
            transform: translateY(-1px);
        }
        .stButton > button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            color: #e4e4e7 !important;
        }

        /* ---------------- Processing ---------------- */
        .proc-title-wrap { text-align: center; padding: 2rem 0 1.2rem 0; }
        .proc-title { font-size: 2.3rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; }
        .proc-sub { color: #a1a1aa; font-size: 1rem; margin-top: 0.4rem; }

        .ring-wrap { display: flex; justify-content: center; margin: 1.8rem 0 2.2rem 0; }

        .stage-status-title {
            font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; color: #71717a; margin-bottom: 1rem;
        }
        .stage-cell {
            display: flex; gap: 0.8rem; padding: 0.9rem 1rem; border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.05); background: rgba(255, 255, 255, 0.02);
            margin-bottom: 0.6rem;
        }
        .stage-cell.processing {
            background: rgba(99, 102, 241, 0.08); border-color: rgba(99, 102, 241, 0.3);
        }
        .stage-cell.completed {
            background: rgba(34, 197, 94, 0.05); border-color: rgba(34, 197, 94, 0.2);
        }
        .stage-cell .icon-dot {
            width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
            background: rgba(255, 255, 255, 0.05); flex-shrink: 0; margin-top: 1px;
        }
        .stage-name-p { font-weight: 700; font-size: 0.92rem; color: #ffffff; }
        .stage-name-p.pending-text { color: #52525b; }
        .stage-desc-p { font-size: 0.8rem; color: #71717a; margin-top: 0.1rem; }
        .stage-progress-track { height: 5px; background: rgba(255, 255, 255, 0.08); border-radius: 4px; margin-top: 0.6rem; overflow: hidden; }
        .stage-progress-fill { height: 100%; background: linear-gradient(90deg, #6366f1, #c084fc); }

        /* ---------------- Results ---------------- */
        .result-hero { text-align: center; margin: 2rem 0 1.8rem 0; }
        .result-hero .rtitle {
            font-size: 2.3rem; font-weight: 800;
            background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
            -webkit-background-clip: text; background-clip: text; color: transparent;
        }
        .result-hero .rsub { color: #a1a1aa; font-size: 1rem; margin-top: 0.4rem; }

        .how-card {
            background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 16px; padding: 1.3rem; height: 100%; transition: border-color 0.2s ease;
        }
        .how-card:hover { border-color: rgba(129, 140, 248, 0.4); }
        .how-num { font-size: 0.8rem; font-weight: 800; color: #c084fc; margin-bottom: 0.4rem; letter-spacing: 0.05em; }
        .how-card h4 { margin: 0 0 0.4rem 0; font-size: 0.98rem; color: #ffffff; font-weight: 700; }
        .how-card p { margin: 0; font-size: 0.83rem; color: #a1a1aa; line-height: 1.5; }

        /* ---------------- Footer ---------------- */
        .app-footer {
            display: flex; justify-content: space-between; flex-wrap: wrap; gap: 2rem;
            padding-top: 2.5rem; margin-top: 3rem; border-top: 1px solid rgba(255, 255, 255, 0.08);
        }
        .app-footer .brand { font-weight: 800; font-size: 1.1rem; color: #ffffff; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem; }
        .app-footer .desc { font-size: 0.82rem; color: #71717a; max-width: 280px; line-height: 1.6; }
        .app-footer .col-title { font-size: 0.8rem; font-weight: 700; color: #a1a1aa; margin-bottom: 0.7rem; letter-spacing: 0.05em; }
        .app-footer .col a { display: block; font-size: 0.85rem; color: #71717a; text-decoration: none; margin-bottom: 0.5rem; transition: color 0.2s; }
        .app-footer .col a:hover { color: #ffffff; }

        @media (max-width: 900px) {
            .hero-title { font-size: 2.3rem; }
            .app-header .navlinks { display: none; }
            .pipeline-row { flex-direction: column; }
            .node-connector { display: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header / Hero / Pipeline visual
# ---------------------------------------------------------------------------

def render_header():
    st.markdown(
        f"""
        <div class="app-header">
            <div class="logo">{LOGO_SVG} HighlightAI</div>
            <div class="navlinks">
                <a href="#top" class="active">Home</a>
                <a href="#how-it-works">How It Works</a>
                <a href="#technology">Technology</a>
            </div>
            <a class="header-cta" href="#workspace">Create Highlight</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-wrap" id="top">
            <div class="hero-title">Turn Every Video Into<br><span class="grad">Its Best Moments.</span></div>
            <div class="hero-sub">
                AI-powered video intelligence that understands your content, finds meaningful moments,
                transcribes speech, and automatically creates polished highlight reels.
            </div>
            <div class="badge-row" id="technology">
                <span class="tech-badge">PyTorch</span>
                <span class="tech-badge">ResNet-18</span>
                <span class="tech-badge">CLIP</span>
                <span class="tech-badge">Whisper</span>
                <span class="tech-badge">OpenCV</span>
            </div>
            <div class="cta-row">
                <a class="cta-primary" href="#workspace">✨ Generate AI Highlight</a>
                <a class="cta-secondary" href="#how-it-works">See How It Works</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_visual():
    nodes_html = ""
    for i, (label, key) in enumerate(DIAGRAM_NODES):
        nodes_html += f"""
        <div class="pipeline-node">
            <div class="node-circle n{i}">{NODE_ICONS[key]}</div>
            <div class="node-label">{label}</div>
        </div>
        """
        if i < len(DIAGRAM_NODES) - 1:
            nodes_html += '<div class="node-connector"></div>'
    st.markdown(f'<div class="pipeline-card" id="how-it-works"><div class="pipeline-row">{nodes_html}</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_bytes(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def format_duration(seconds):
    if seconds is None:
        return "—"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


def get_video_metadata(path):
    metadata = {"duration": None, "resolution": None, "fps": None}
    if cv2 is None:
        return metadata
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return metadata
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if fps > 0 and frame_count > 0:
            metadata["duration"] = frame_count / fps
        metadata["resolution"] = f"{width}x{height}" if width and height else None
        metadata["fps"] = round(fps, 2) if fps else None
    except Exception:
        pass
    return metadata


# ---------------------------------------------------------------------------
# Workspace: upload + preview (left) / configuration (right)
# ---------------------------------------------------------------------------

def render_upload_section():
    st.markdown('<div class="surface-card" id="workspace">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="upload-visual">
            <div class="upload-icon-circle">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
                  <path d="M12 16V4M12 4L7 9M12 4L17 9" stroke="#818cf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                  <path d="M4 16V18C4 19.1 4.9 20 6 20H18C19.1 20 20 19.1 20 18V16" stroke="#818cf8" stroke-width="2" stroke-linecap="round"/>
                </svg>
            </div>
            <div class="upload-title">Drop your video here or browse files</div>
            <div class="upload-sub">Supported formats: MP4, AVI, MOV, MKV</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload video", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded_file


def render_input_video_card(uploaded_file, metadata, file_size):
    st.markdown('<div class="surface-card"><div class="card-title">PREVIEW</div>', unsafe_allow_html=True)
    st.video(uploaded_file)

    chips = [
        ("Filename", uploaded_file.name),
        ("Duration", format_duration(metadata.get("duration"))),
        ("Resolution", metadata.get("resolution") or "—"),
        ("FPS", str(metadata.get("fps")) if metadata.get("fps") else "—"),
        ("Size", format_bytes(file_size)),
    ]
    chip_html = "".join(
        f'<div class="meta-chip"><div class="lbl">{label}</div><div class="val">{value}</div></div>'
        for label, value in chips
    )
    st.markdown(f'<div class="meta-strip">{chip_html}</div></div>', unsafe_allow_html=True)


def render_configuration_panel(uploaded_file):
    """Renders the AI Configuration card and returns the chosen settings dict."""
    st.markdown(
        """
        <div class="surface-card">
        <div class="card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M4 6H14M18 6H20M4 12H8M12 12H20M4 18H16M20 18H20" stroke="#818cf8" stroke-width="2" stroke-linecap="round"/><circle cx="16" cy="6" r="2" stroke="#818cf8" stroke-width="2"/><circle cx="10" cy="12" r="2" stroke="#818cf8" stroke-width="2"/><circle cx="18" cy="18" r="2" stroke="#818cf8" stroke-width="2"/></svg>
            AI Configuration
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="field-label">Whisper Model</div>', unsafe_allow_html=True)
    whisper_model = st.selectbox(
        "Whisper Model", ["tiny", "base", "small", "medium (Accurate)", "large"],
        index=3, label_visibility="collapsed",
    )
    whisper_model = whisper_model.split(" ")[0]

    st.markdown('<div class="field-label">CLIP Model</div>', unsafe_allow_html=True)
    clip_model = st.selectbox(
        "CLIP Model", ["ViT-B/32", "ViT-L/14"], index=0, label_visibility="collapsed",
    )

    if "highlight_fps" not in st.session_state:
        st.session_state.highlight_fps = 2
    st.markdown(
        f'<div class="field-row-between"><div class="field-label" style="margin-top:0.9rem;">Frame Sampling (FPS)</div>'
        f'<div class="field-value">{st.session_state.highlight_fps} FPS</div></div>',
        unsafe_allow_html=True,
    )
    highlight_fps = st.slider(
        "Frame Sampling (FPS)", min_value=1, max_value=5, step=1,
        key="highlight_fps", label_visibility="collapsed",
    )
    st.markdown('<div class="helper-caption"><span>Faster Processing</span><span>More Detail</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="field-label" style="margin-top:1.1rem;">Scoring Balance</div>', unsafe_allow_html=True)
    if "resnet_weight" not in st.session_state:
        st.session_state.resnet_weight = 0.6
    if "clip_weight" not in st.session_state:
        st.session_state.clip_weight = 0.4

    resnet_pct = int(round(st.session_state.resnet_weight * 100))
    clip_pct = 100 - resnet_pct
    st.markdown(
        f"""
        <div class="scoring-bar">
            <div style="width:{resnet_pct}%; background:linear-gradient(90deg, #6366f1, #818cf8);"></div>
            <div style="width:{clip_pct}%; background:linear-gradient(90deg, #c084fc, #f472b6);"></div>
        </div>
        <div class="scoring-labels">
            <span class="resnet">ResNet-18 ({resnet_pct}%)</span>
            <span class="clip">CLIP ({clip_pct}%)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Advanced Settings"):
        max_highlight_duration = st.slider("Max Highlight Duration (sec)", 10, 300, 60, step=10)
        caption_font_size = st.slider("Caption Font Size", 16, 48, 28, step=2)
        resnet_weight_input = st.slider(
            "ResNet Weight (adjusts Scoring Balance above)", 0.0, 1.0, st.session_state.resnet_weight, step=0.1,
        )
        if resnet_weight_input != st.session_state.resnet_weight:
            st.session_state.resnet_weight = resnet_weight_input
            st.session_state.clip_weight = round(1.0 - resnet_weight_input, 2)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    return {
        "whisper_model": whisper_model,
        "clip_model": clip_model,
        "highlight_fps": highlight_fps,
        "resnet_weight": st.session_state.resnet_weight,
        "clip_weight": st.session_state.clip_weight,
        "max_highlight_duration": max_highlight_duration,
        "caption_font_size": caption_font_size,
    }


# ---------------------------------------------------------------------------
# Processing dashboard
# ---------------------------------------------------------------------------

def _ring_svg(pct):
    r = 78
    circumference = 2 * 3.14159265 * r
    offset = circumference * (1 - pct / 100)
    return f"""
    <svg width="190" height="190" viewBox="0 0 190 190">
        <defs>
            <linearGradient id="ringGrad" x1="0" y1="0" x2="190" y2="190" gradientUnits="userSpaceOnUse">
                <stop offset="0" stop-color="#818cf8"/>
                <stop offset="1" stop-color="#c084fc"/>
            </linearGradient>
        </defs>
        <circle cx="95" cy="95" r="{r}" stroke="rgba(255,255,255,0.08)" stroke-width="13" fill="none"/>
        <circle cx="95" cy="95" r="{r}" stroke="url(#ringGrad)" stroke-width="13" fill="none"
            stroke-linecap="round" stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}"
            transform="rotate(-90 95 95)"/>
        <text x="95" y="103" text-anchor="middle" font-size="34" font-weight="800" fill="#ffffff" font-family="Plus Jakarta Sans, sans-serif">{pct}%</text>
    </svg>
    """


def render_stage_grid(active_key, done_keys, scoring_progress=None, error_key=None):
    st.markdown('<div class="stage-status-title">PIPELINE STATUS</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (key, name, desc) in enumerate(PIPELINE_STAGES):
        with cols[i % 2]:
            if key == error_key:
                css, icon_svg, name_css = "error", "!", ""
            elif key in done_keys:
                css, icon_svg, name_css = "completed", STAGE_CHECK, ""
            elif key == active_key:
                css, icon_svg, name_css = "processing", STAGE_SPIN, ""
            else:
                css, icon_svg, name_css = "pending", "", "pending-text"

            status_label = {"completed": "Completed", "processing": "Processing", "error": "Error", "pending": "Pending"}[css]
            detail = desc if css != "pending" else "Pending"
            if key == "scoring" and key == active_key and scoring_progress:
                current, total = scoring_progress
                detail = f"Processing • {current}/{total} frames analyzed"

            progress_html = ""
            if key == "scoring" and key == active_key and scoring_progress:
                current, total = scoring_progress
                pct = int((current / total) * 100) if total else 0
                progress_html = f'<div class="stage-progress-track"><div class="stage-progress-fill" style="width:{pct}%;"></div></div>'

            st.markdown(
                f"""
                <div class="stage-cell {css}">
                    <div class="icon-dot">{icon_svg}</div>
                    <div style="flex:1;">
                        <div class="stage-name-p {name_css}">{name}</div>
                        <div class="stage-desc-p">{detail if css != 'pending' else status_label}</div>
                        {progress_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def run_pipeline_with_dashboard(cfg):
    st.markdown(
        """
        <div class="proc-title-wrap">
            <div class="proc-title">Creating Your Highlight Reel</div>
            <div class="proc-sub">AI is analyzing your video to find the moments that matter.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ring_placeholder = st.empty()
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    stage_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    done_keys = set()
    state = {"scoring_progress": None}
    total_stages = len(PIPELINE_STAGES)

    def refresh(active_key, error_key=None):
        pct = int((len(done_keys) / total_stages) * 100)
        if active_key and state["scoring_progress"] and active_key == "scoring":
            current, total = state["scoring_progress"]
            frac = current / total if total else 0
            pct = int(((len(done_keys) + frac) / total_stages) * 100)
        elif active_key:
            pct = int(((len(done_keys) + 0.5) / total_stages) * 100)
        pct = min(max(pct, 0), 100)

        with ring_placeholder.container():
            st.markdown(f'<div class="ring-wrap">{_ring_svg(pct)}</div>', unsafe_allow_html=True)
        with stage_placeholder.container():
            render_stage_grid(active_key, done_keys, state["scoring_progress"], error_key)

    def progress_callback(stage, current=None, total=None):
        order = [k for k, *_ in PIPELINE_STAGES]
        if stage in order:
            for k in order[: order.index(stage)]:
                done_keys.add(k)
        state["scoring_progress"] = (current, total) if (stage == "scoring" and current is not None and total) else None
        refresh(stage)

    refresh("extract")

    sig = inspect.signature(run)
    supports_callback = "progress_callback" in sig.parameters

    if supports_callback:
        run(cfg, progress_callback=progress_callback)
    else:
        with st.spinner("Running AI pipeline..."):
            run(cfg)

    for k, *_ in PIPELINE_STAGES:
        done_keys.add(k)
    refresh(None)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def render_results(final_output, settings, input_metadata, pipeline_stats):
    st.markdown(
        """
        <div class="result-hero">
            <div class="rtitle">Your Highlight Is Ready</div>
            <div class="rsub">Your best moments have been intelligently selected and rendered.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.video(str(final_output))
    with open(final_output, "rb") as video_file:
        st.download_button(
            label="⬇️ Download Highlight Reel",
            data=video_file,
            file_name="ai_highlight_reel.mp4",
            mime="video/mp4",
            type="primary",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="surface-card"><div class="card-title">Summary Metrics</div>', unsafe_allow_html=True)
    metrics = [("Original Duration", format_duration(input_metadata.get("duration")))]
    if pipeline_stats.get("frames_analyzed") is not None:
        metrics.append(("Frames Analyzed", str(pipeline_stats["frames_analyzed"])))
    if pipeline_stats.get("frames_selected") is not None:
        metrics.append(("Highlight Frames", str(pipeline_stats["frames_selected"])))
    metrics.append(("Whisper Model", settings.get("whisper_model", "base")))
    metrics.append(("CLIP Model", settings.get("clip_model", "ViT-B/32")))
    if pipeline_stats.get("device"):
        metrics.append(("Device", pipeline_stats["device"]))

    chip_html = "".join(
        f'<div class="meta-chip"><div class="lbl">{label}</div><div class="val">{value}</div></div>'
        for label, value in metrics
    )
    st.markdown(f'<div class="meta-strip">{chip_html}</div></div>', unsafe_allow_html=True)

    if st.session_state.get("original_video_path"):
        st.markdown('<div class="surface-card"><div class="card-title">Original vs. AI Highlight</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("ORIGINAL VIDEO")
            st.video(st.session_state["original_video_path"])
        with c2:
            st.caption("AI HIGHLIGHT")
            st.video(str(final_output))
        st.markdown("</div>", unsafe_allow_html=True)

    render_ai_summary()

    if st.button("🔄 Create Another Highlight", use_container_width=True):
        for key in ["processed_file_id", "current_file_id", "result_path", "pipeline_stats", "original_video_path"]:
            st.session_state.pop(key, None)
        st.rerun()


def render_ai_summary():
    st.markdown('<div class="surface-card"><div class="card-title">AI Intelligence Stack</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    items = [
        ("01", "Visual Intelligence", "ResNet-18 identifies visually informative frames."),
        ("02", "Semantic Understanding", "CLIP evaluates semantic relevance against your content."),
        ("03", "Speech Intelligence", "Whisper transcribes speech and aligns captions."),
        ("04", "Smart Rendering", "The strongest moments are assembled into the final reel."),
    ]
    for col, (num, title, desc) in zip(cols, items):
        with col:
            st.markdown(
                f"<div class='how-card'><div class='how-num'>{num}</div>"
                f"<h4>{title}</h4><p>{desc}</p></div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_footer():
    st.markdown(
        f"""
        <div class="app-footer">
            <div>
                <div class="brand">{LOGO_SVG.replace('width="26" height="26"', 'width="20" height="20"')} HighlightAI</div>
                <div class="desc">AI-powered video intelligence for automatically discovering meaningful moments. Powered by PyTorch, ResNet-18, CLIP, Whisper, and OpenCV.</div>
            </div>
            <div class="col">
                <div class="col-title">NAVIGATION</div>
                <a href="#top">Home</a>
                <a href="#how-it-works">How It Works</a>
                <a href="#technology">Technology</a>
            </div>
            <div class="col">
                <div class="col-title">RESOURCES</div>
                <a href="#">Documentation</a>
                <a href="#">GitHub Repository</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Application Entrypoint
# ---------------------------------------------------------------------------

def main():
    setup_page()
    inject_custom_css()
    render_header()

    for key, default in [
        ("current_file_id", None),
        ("result_path", None),
        ("pipeline_stats", {}),
        ("original_video_path", None),
        ("processing", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if st.session_state["processing"]:
        cfg = st.session_state["_active_cfg"]
        try:
            run_pipeline_with_dashboard(cfg)
            final_output = Path(cfg["final_video_path"])
            if final_output.exists():
                st.session_state["result_path"] = str(final_output)
                st.session_state["pipeline_stats"] = {
                    "frames_analyzed": cfg.get("frames_analyzed"),
                    "frames_selected": cfg.get("frames_selected"),
                    "device": cfg.get("device"),
                }
            else:
                st.error("❌ Something went wrong while generating your highlight.")
        except Exception as e:
            st.error("❌ Something went wrong while generating your highlight.")
            with st.expander("Technical Details"):
                st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        finally:
            st.session_state["processing"] = False
            st.rerun()
        return

    if st.session_state["result_path"] and Path(st.session_state["result_path"]).exists():
        render_results(
            Path(st.session_state["result_path"]),
            st.session_state.get("_active_settings", {}),
            st.session_state.get("input_metadata", {}),
            st.session_state.get("pipeline_stats", {}),
        )
        render_footer()
        return

    render_hero()
    render_pipeline_visual()

    col_left, col_right = st.columns([1.25, 1])
    with col_left:
        uploaded_file = render_upload_section()
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_id = f"{uploaded_file.name}-{len(file_bytes)}"
            if st.session_state.get("current_file_id") != file_id:
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file_bytes)
                    st.session_state["current_input_path"] = tmp.name
                st.session_state["current_file_id"] = file_id
                st.session_state["original_video_path"] = st.session_state["current_input_path"]

            input_path = st.session_state["current_input_path"]
            metadata = get_video_metadata(input_path)
            st.session_state["input_metadata"] = metadata
            render_input_video_card(uploaded_file, metadata, len(file_bytes))

    with col_right:
        settings = render_configuration_panel(uploaded_file)

    if uploaded_file is None:
        render_ai_summary()
        render_footer()
        return

    generate_clicked = st.button(
        "✨ Generate AI Highlight", type="primary", use_container_width=True,
    )

    if generate_clicked:
        output_dir = ROOT_DIR / "output"
        output_dir.mkdir(exist_ok=True)
        final_output = output_dir / "streamlit_highlight.mp4"

        config_path = ROOT_DIR / "config.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            st.error("❌ config.yaml was not found in the project root.")
            st.stop()

        cfg["input_video"] = st.session_state["current_input_path"]
        cfg["output_dir"] = str(output_dir)
        cfg["frames_dir"] = str(output_dir / "frames")
        cfg["selected_frames_dir"] = str(output_dir / "selected_frames")
        cfg["temp_video_path"] = str(output_dir / "streamlit_temp_highlight.mp4")
        cfg["final_video_path"] = str(final_output)
        cfg["audio_path"] = str(output_dir / "streamlit_audio.wav")

        cfg["whisper_model_size"] = settings["whisper_model"]
        cfg["clip_model_name"] = settings["clip_model"]
        cfg["highlight_fps"] = settings["highlight_fps"]
        cfg["resnet_score_weight"] = settings["resnet_weight"]
        cfg["clip_score_weight"] = settings["clip_weight"]
        cfg["max_highlight_duration"] = settings["max_highlight_duration"]
        cfg["caption_font_size"] = settings["caption_font_size"]

        st.session_state["_active_cfg"] = cfg
        st.session_state["_active_settings"] = settings
        st.session_state["processing"] = True
        st.rerun()

    render_ai_summary()
    render_footer()


if __name__ == "__main__":
    main()