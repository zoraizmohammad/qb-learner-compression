#!/usr/bin/env python3
"""Build a 36x48 in portrait Duke-branded research poster with python-pptx."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from PIL import Image

BASE = "/sessions/amazing-blissful-knuth/mnt/outputs/poster_work"
OUT  = os.path.join(BASE, "pptx", "poster_pptx.pptx")

# ---------- palette ----------
NAVY  = RGBColor(0x01, 0x21, 0x69)
ROYAL = RGBColor(0x00, 0x53, 0x9B)
AMBER = RGBColor(0xE8, 0x99, 0x23)
TINT  = RGBColor(0xEE, 0xF2, 0xF8)
INK   = RGBColor(0x1A, 0x1A, 0x1A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY  = RGBColor(0x5A, 0x66, 0x78)
AMBER_TINT = RGBColor(0xFD, 0xF3, 0xE3)

FONT = "Arial"

# ---------- geometry (inches) ----------
PW, PH = 36.0, 48.0
MARGIN = 0.6
GUTTER = 0.5
HEADER_H = 5.0
COL_W = (PW - 2*MARGIN - 2*GUTTER) / 3.0   # ~11.27
CONTENT_TOP = HEADER_H + 0.45
CONTENT_BOT = 47.4
COL_X = [MARGIN + i*(COL_W + GUTTER) for i in range(3)]

prs = Presentation()
prs.slide_width  = Inches(PW)
prs.slide_height = Inches(PH)
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

# ---------- helpers ----------
def set_fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color

def no_line(shape):
    shape.line.fill.background()

def set_line(shape, color, w=1.0):
    shape.line.color.rgb = color
    shape.line.width = Pt(w)

def rect(x, y, w, h, fill=None, line=None, lw=1.0, rounded=False, radius=0.06):
    shp = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(shp, Inches(x), Inches(y), Inches(w), Inches(h))
    s.shadow.inherit = False
    if rounded:
        try:
            s.adjustments[0] = radius
        except Exception:
            pass
    if fill is None:
        s.fill.background()
    else:
        set_fill(s, fill)
    if line is None:
        no_line(s)
    else:
        set_line(s, line, lw)
    return s

def _set_run(r, text, size, color, bold=False, italic=False, font=FONT):
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.name = font
    r.font.color.rgb = color

def textbox(x, y, w, h, anchor=MSO_ANCHOR.TOP, wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    return tb, tf

def add_para(tf, first=False):
    if first and not tf.paragraphs[0].runs and tf.paragraphs[0].text == "":
        return tf.paragraphs[0]
    return tf.add_paragraph()

# segments: list of (text, opts) where opts dict size/color/bold/italic
def rich_para(p, segments, align=PP_ALIGN.LEFT, space_after=6, space_before=0,
              line=1.02, bullet=False, level=0):
    p.alignment = align
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    try:
        p.line_spacing = line
    except Exception:
        pass
    p.level = level
    for (txt, o) in segments:
        r = p.add_run()
        _set_run(r, txt, o.get("size", 22), o.get("color", INK),
                 o.get("bold", False), o.get("italic", False), o.get("font", FONT))
    if bullet:
        _add_bullet(p)
    else:
        _no_bullet(p)
    return p

def _pPr(p):
    pPr = p._pPr
    if pPr is None:
        pPr = p._p.get_or_add_pPr()
    return pPr

def _no_bullet(p):
    pPr = _pPr(p)
    for tag in ('a:buChar', 'a:buAutoNum', 'a:buNone'):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    pPr.append(pPr.makeelement(qn('a:buNone'), {}))

def _add_bullet(p, char="▪", color=ROYAL):
    pPr = _pPr(p)
    pPr.set('indent', str(Inches(-0.28)))
    pPr.set('marL', str(Inches(0.28)))
    for tag in ('a:buChar', 'a:buAutoNum', 'a:buNone'):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    buFont = pPr.makeelement(qn('a:buFont'), {'typeface': 'Arial'})
    pPr.append(buFont)
    buClr = pPr.makeelement(qn('a:buClr'), {})
    srgb = pPr.makeelement(qn('a:srgbClr'), {'val': '%02X%02X%02X' % (color[0], color[1], color[2])})
    buClr.append(srgb)
    pPr.insert(list(pPr).index(buFont), buClr)
    buChar = pPr.makeelement(qn('a:buChar'), {'char': char})
    pPr.append(buChar)

def pic_fit(path, x, y, w):
    im = Image.open(path)
    ar = im.size[1] / im.size[0]
    h = w * ar
    slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    return h

# A content box with navy numbered header bar; returns inner text region top.
def section_box(x, y, w, h, num, title, fill=TINT, border=None, bw=1.0,
                bar_color=NAVY, bar_h=0.92, accent=None):
    # outer box
    rect(x, y, w, h, fill=fill, line=border, lw=bw, rounded=True, radius=0.025)
    # header bar
    rect(x, y, w, bar_h, fill=bar_color, rounded=True, radius=0.10)
    # cover bottom rounded corners of bar so it sits flush
    rect(x, y + bar_h*0.5, w, bar_h*0.5, fill=bar_color)
    # number chip
    chip_w = 0.78
    if num is not None:
        rect(x + 0.22, y + (bar_h-0.62)/2, chip_w, 0.62, fill=WHITE if bar_color==NAVY else NAVY,
             rounded=True, radius=0.18)
        ctb, ctf = textbox(x + 0.22, y + (bar_h-0.62)/2 - 0.02, chip_w, 0.62, anchor=MSO_ANCHOR.MIDDLE)
        rich_para(add_para(ctf, True), [(str(num), {"size": 28, "color": bar_color if bar_color==NAVY else WHITE, "bold": True})],
                  align=PP_ALIGN.CENTER, space_after=0)
    tx = x + (0.22 + chip_w + 0.28 if num is not None else 0.3)
    htb, htf = textbox(tx, y, w - (tx - x) - 0.2, bar_h, anchor=MSO_ANCHOR.MIDDLE)
    rich_para(add_para(htf, True), [(title, {"size": 27, "color": WHITE, "bold": True})],
              align=PP_ALIGN.LEFT, space_after=0, line=0.98)
    if accent is not None:
        # accent stripe on left edge of body
        rect(x, y, 0.16, h, fill=accent, rounded=True, radius=0.5)
    return y + bar_h + 0.18  # inner content top

# ===================================================================
# HEADER BAND
# ===================================================================
rect(0, 0, PW, HEADER_H, fill=NAVY)
# thin royal + amber accent rule at bottom of header
rect(0, HEADER_H - 0.10, PW, 0.07, fill=ROYAL)
rect(0, HEADER_H - 0.03, PW, 0.03, fill=AMBER)

# Logo on right of band (white knockout). native 2048x271 -> ratio ~0.1323
logo_w = 6.6
logo_im = Image.open(os.path.join(BASE, "assets/duke_ece_logo.png"))
logo_h = logo_w * logo_im.size[1]/logo_im.size[0]
logo_x = PW - MARGIN - logo_w
logo_y = 0.55
slide.shapes.add_picture(os.path.join(BASE, "assets/duke_ece_logo.png"),
                         Inches(logo_x), Inches(logo_y), Inches(logo_w), Inches(logo_h))

# Venue tag pill (top-left of header)
tagw = 8.6
rect(MARGIN, 0.5, tagw, 0.66, fill=ROYAL, rounded=True, radius=0.5)
ttb, ttf = textbox(MARGIN, 0.5, tagw, 0.66, anchor=MSO_ANCHOR.MIDDLE)
rich_para(add_para(ttf, True),
          [("IEEE Quantum Week  —  QCE26 Poster", {"size": 22, "color": WHITE, "bold": True})],
          align=PP_ALIGN.CENTER, space_after=0)

# Title
title_w = PW - 2*MARGIN
ttb2, ttf2 = textbox(MARGIN, 1.32, title_w - 0.2, 2.3, anchor=MSO_ANCHOR.TOP)
rich_para(add_para(ttf2, True),
          [("Hardware-Aware Quantum Bayesian Learner Compression", {"size": 56, "color": WHITE, "bold": True})],
          align=PP_ALIGN.LEFT, space_after=2, line=0.98)
rich_para(ttf2.add_paragraph(),
          [("Linking Human Belief Updating to NISQ Circuit Complexity", {"size": 38, "color": RGBColor(0xCF,0xDD,0xF0), "bold": False, "italic": True})],
          align=PP_ALIGN.LEFT, space_after=0, line=0.98)

# Author + affiliation + contact (bottom strip of header)
atb, atf = textbox(MARGIN, 3.78, title_w - logo_w - 0.4, 1.1, anchor=MSO_ANCHOR.TOP)
rich_para(add_para(atf, True),
          [("Mohammad Zoraiz", {"size": 30, "color": WHITE, "bold": True}),
           ("      ✉ mz248@duke.edu", {"size": 26, "color": RGBColor(0xCF,0xDD,0xF0), "bold": False})],
          align=PP_ALIGN.LEFT, space_after=3, line=1.0)
rich_para(atf.add_paragraph(),
          [("Duke University  ·  Pratt School of Engineering  ·  Electrical Engineering, Physics & Computer Science  ·  Durham, NC",
            {"size": 22, "color": RGBColor(0xCF,0xDD,0xF0)})],
          align=PP_ALIGN.LEFT, space_after=0, line=1.0)

# ===================================================================
# COLUMN 1
# ===================================================================
def body_box(tf, lines):
    """lines: list of dicts: {'segs':[...], 'bullet':True/False, 'sa':6, 'sb':0, 'align':...}"""
    first = True
    for ln in lines:
        p = add_para(tf, first)
        rich_para(p, ln["segs"], align=ln.get("align", PP_ALIGN.LEFT),
                  space_after=ln.get("sa", 6), space_before=ln.get("sb", 0),
                  line=ln.get("line", 1.02), bullet=ln.get("bullet", False))
        first = False

S = 22  # base body size
BS = 22

x1 = COL_X[0]
y = CONTENT_TOP

# [1] Motivation
h1 = 6.55
top = section_box(x1, y, COL_W, h1, 1, "Motivation — Two fields, one question")
tb, tf = textbox(x1+0.35, top, COL_W-0.7, h1-(top-y)-0.2)
body_box(tf, [
    {"segs":[("Bayesian models cast human learning as ", {"size":BS}),
             ("belief updating", {"size":BS,"bold":True}),
             (": a prior over hypotheses, revised by evidence. ", {"size":BS}),
             ("Quantum cognition", {"size":BS,"bold":True}),
             (" represents beliefs as density operators in Hilbert space, with evidence acting through ", {"size":BS}),
             ("quantum channels", {"size":BS,"bold":True}),(".", {"size":BS})],
     "bullet":True, "sa":7},
    {"segs":[("On ", {"size":BS}),("NISQ", {"size":BS,"bold":True}),
             (" hardware, ", {"size":BS}),
             ("two-qubit gates dominate the error budget", {"size":BS,"bold":True}),
             (", so removing them without breaking the task is a central engineering goal.", {"size":BS})],
     "bullet":True, "sa":7},
    {"segs":[("Central question:  ", {"size":BS,"bold":True,"color":NAVY}),
             ("How much of a quantum Bayesian learner’s entangling structure can we remove before it can no longer represent the task — and what happens on real hardware?",
              {"size":BS,"italic":True,"color":NAVY})],
     "bullet":True, "sa":0},
])
y += h1 + 0.45

# [2] The Quantum Bayesian Learner (schematic)
h2 = 12.2
top = section_box(x1, y, COL_W, h2, 2, "The Quantum Bayesian Learner")
# detailed circuit pipeline (2.49:1) fit to inner width
sch_w = COL_W - 0.7
sch_h = pic_fit(os.path.join(BASE,"assets/pipeline_detailed.png"), x1+0.35, top+0.02, sch_w)
tb, tf = textbox(x1+0.35, top+sch_h+0.18, COL_W-0.7, h2-(top-y)-sch_h-0.4)
body_box(tf, [
    {"segs":[("Belief = state ρ on a ", {"size":BS}),("2-qubit", {"size":BS,"bold":True}),
             (" register; prior ρ₀ = “no evidence yet.” A stimulus x updates the belief via a ", {"size":BS}),
             ("CPTP evidence channel", {"size":BS,"bold":True}),
             ("  \U0001D4D4ₓ(ρ) = Σₖ Eₓₖ ρ Eₓₖ†.", {"size":BS})],
     "bullet":True, "sa":7},
    {"segs":[("Update circuit = ", {"size":BS}),("hardware-efficient ansatz", {"size":BS,"bold":True}),
             (": data re-uploading + per-layer Rₓ, R_z + ", {"size":BS}),
             ("Heisenberg entanglers", {"size":BS,"bold":True}),
             ("  U_ij = exp[−i(α XᵢXⱼ + β YᵢYⱼ + γ ZᵢZⱼ)]  as native R_XX, R_YY, R_ZZ. Depth ", {"size":BS}),
             ("D = 4", {"size":BS,"bold":True}),(", 2 re-uploads.", {"size":BS})],
     "bullet":True, "sa":7},
    {"segs":[("Readout: single-qubit Pauli expectations", {"size":BS,"bold":True}),
             ("  v = (⟨Z₀⟩,⟨Z₁⟩,⟨X₀⟩,⟨X₁⟩) → trainable head ŷ = σ(w·v + b). Single-qubit-only readout means ", {"size":BS}),
             ("entanglement is the ONLY way to carry feature interaction", {"size":BS,"bold":True}),
             (" — the two-qubit budget literally bounds the learner’s capacity.", {"size":BS})],
     "bullet":True, "sa":7},
    {"segs":[("Two realizations, same conclusions: a fast ", {"size":BS}),
             ("pure-state", {"size":BS,"bold":True}),(" model and a literal ", {"size":BS}),
             ("mixed-state density matrix", {"size":BS,"bold":True}),
             (" (maximally mixed prior I/2ⁿ, non-unital Lüders/POVM evidence filter + amplitude damping, Bayesian trace renormalization).", {"size":BS})],
     "bullet":True, "sa":0},
])
y += h2 + 0.45

# [3] Method
h3 = 11.05
top = section_box(x1, y, COL_W, h3, 3, "Method — mask, real cost, greedy pruning")
tb, tf = textbox(x1+0.35, top, COL_W-0.7, h3-(top-y)-0.2)
body_box(tf, [
    {"segs":[("Per-term binary mask", {"size":BS,"bold":True}),
             ("  mₗ,ₚ ∈ {0,1} over p ∈ {XX,YY,ZZ} in each of D=4 layers  ⇒  ", {"size":BS}),
             ("12 maskable terms", {"size":BS,"bold":True}),
             ("  (R_PP(0)=I when off).", {"size":BS})], "bullet":True, "sa":7},
    {"segs":[("Hardware-aware cost N₂q", {"size":BS,"bold":True}),
             (" = real post-transpile two-qubit gate count on IBM ", {"size":BS}),
             ("FakeManila", {"size":BS,"bold":True}),
             (" (5-qubit, includes SWAPs), via Qiskit’s transpiler. Full circuit ⇒ ", {"size":BS}),
             ("N₂q = 24", {"size":BS,"bold":True}),
             ("  (each active Heisenberg term = 2 CNOTs).", {"size":BS})], "bullet":True, "sa":7},
    {"segs":[("Objective:", {"size":BS,"bold":True}),
             ("  L(θ,m) = (1/N) Σ BCE(yₖ, ŷ) + λ·N₂q(m).", {"size":BS})], "bullet":True, "sa":7},
    {"segs":[("Greedy structured pruning:", {"size":BS,"bold":True}),
             ("  from the full circuit, repeatedly drop the term whose removal least increases training cross-entropy, ", {"size":BS}),
             ("warm-start retrain", {"size":BS,"bold":True}),
             (" → traces the accuracy-vs-N₂q ", {"size":BS}),
             ("frontier", {"size":BS,"bold":True}),(" from 24 down to 0.", {"size":BS})], "bullet":True, "sa":7},
    {"segs":[("Exact ", {"size":BS}),("JAX autodiff", {"size":BS,"bold":True}),
             (" (no parameter-shift); statevector verified ", {"size":BS}),
             ("bit-for-bit vs Qiskit", {"size":BS,"bold":True}),
             ("  (|1 − fidelity| ≈ 4×10⁻¹⁶).", {"size":BS})], "bullet":True, "sa":0},
])
y += h3 + 0.45

# Task box
h4 = 5.4
top = section_box(x1, y, COL_W, h4, None, "Task — controllable difficulty knob", bar_color=ROYAL)
tb, tf = textbox(x1+0.35, top, COL_W-0.7, h4-(top-y)-0.2)
body_box(tf, [
    {"segs":[("Checkerboard", {"size":BS,"bold":True}),
             (" categorization:  y = (⌊f·x₀⌋ + ⌊f·x₁⌋) mod 2,  stimuli in [0,1]².", {"size":BS})],
     "bullet":True, "sa":7},
    {"segs":[("Difficulty f:  ", {"size":BS}),
             ("f=2 easy / f=3 medium / f=4 hard", {"size":BS,"bold":True}),
             (".  200 stimuli, 70/30 split, balanced ⇒ ", {"size":BS}),
             ("chance = 0.5", {"size":BS,"bold":True}),(".", {"size":BS})],
     "bullet":True, "sa":7},
    {"segs":[("Local similarity, but the rule depends on the ", {"size":BS}),
             ("joint", {"size":BS,"bold":True}),
             (" features ⇒ needs the interaction only entanglement can supply.", {"size":BS})],
     "bullet":True, "sa":0},
])

# ===================================================================
# COLUMN 2  (RESULTS)
# ===================================================================
x2 = COL_X[1]
y = CONTENT_TOP

# [4] Result 1
h = 13.5
top = section_box(x2, y, COL_W, h, 4, "Result 1 — The capacity boundary")
fig_w = COL_W - 1.4
fig_x = x2 + (COL_W - fig_w)/2.0
fig_h = pic_fit(os.path.join(BASE,"figures/fig_frontier_family.png"), fig_x, top+0.02, fig_w)
tb, tf = textbox(x2+0.35, top+fig_h+0.16, COL_W-0.7, h-(top-y)-fig_h-0.35)
body_box(tf, [
    {"segs":[("Above chance at all nonzero budgets (easy/medium ≈ ", {"size":BS}),
             ("0.95–0.97", {"size":BS,"bold":True}),(", hard ≈ ", {"size":BS}),
             ("0.87–0.93", {"size":BS,"bold":True}),
             ("); frontier ", {"size":BS}),("nearly flat", {"size":BS,"bold":True}),
             (" → most entangling structure is ", {"size":BS}),
             ("redundant", {"size":BS,"bold":True}),(".", {"size":BS})], "bullet":True, "sa":7},
    {"segs":[("At ", {"size":BS}),("N₂q = 0", {"size":BS,"bold":True}),
             (" all difficulties ", {"size":BS}),("collapse toward chance", {"size":BS,"bold":True}),
             ("  (0.52 easy, 0.64 medium, 0.49 hard) → some entanglement is ", {"size":BS}),
             ("necessary", {"size":BS,"bold":True}),(", but far less than the full circuit.", {"size":BS})], "bullet":True, "sa":7},
    {"segs":[("Hard task peaks at ", {"size":BS}),
             ("intermediate cost (0.927 @ N₂q=6)", {"size":BS,"bold":True}),
             (" and is lower for the full circuit ", {"size":BS}),
             ("(0.873 @ N₂q=24)", {"size":BS,"bold":True}),
             (" → moderate compression acts as a capacity-control ", {"size":BS}),
             ("regularizer", {"size":BS,"bold":True}),(".", {"size":BS})], "bullet":True, "sa":0},
])
y += h + 0.45

# [5] Result 2 + table
h = 19.4
top = section_box(x2, y, COL_W, h, 5, "Result 2 — Structure beats count")
fig_w = COL_W - 1.4
fig_x = x2 + (COL_W - fig_w)/2.0
fig_h = pic_fit(os.path.join(BASE,"figures/fig_mask_ablation.png"), fig_x, top+0.02, fig_w)
txt_top = top + fig_h + 0.16
tb, tf = textbox(x2+0.35, txt_top, COL_W-0.7, 3.0)
body_box(tf, [
    {"segs":[("Learned (greedy) masks vs ", {"size":BS}),
             ("random masks at matched N₂q", {"size":BS,"bold":True}),(".", {"size":BS})], "bullet":True, "sa":6},
    {"segs":[("Learned wins at ", {"size":BS}),("every", {"size":BS,"bold":True}),
             (" budget by ", {"size":BS}),("~6–12 points", {"size":BS,"bold":True}),
             (".  Per seed: learned beats random in ", {"size":BS}),
             ("49 / 55", {"size":BS,"bold":True}),
             (" paired comparisons, mean advantage ", {"size":BS}),
             ("+0.083", {"size":BS,"bold":True}),
             (", one-sided Wilcoxon ", {"size":BS}),("p < 10⁻⁷", {"size":BS,"bold":True}),(".", {"size":BS})], "bullet":True, "sa":6},
    {"segs":[("⇒  ", {"size":BS,"bold":True,"color":NAVY}),
             ("Which", {"size":BS,"bold":True,"color":NAVY,"italic":True}),
             (" entanglers you keep matters, not just how many.", {"size":BS,"bold":True,"color":NAVY})], "bullet":False, "sa":0},
])

# ---- table (hard task: learned vs random) ----
cap_top = txt_top + 3.55   # below the 3 text lines (clear bullet 3)
tcap, tcf = textbox(x2+0.35, cap_top, COL_W-0.7, 0.5)
rich_para(add_para(tcf, True), [("Hard task — learned vs random  (mean acc, 5 seeds)", {"size":20,"bold":True,"color":NAVY})],
          align=PP_ALIGN.LEFT, space_after=0)
tbl_top = cap_top + 0.55
rows_data = [
    ("N₂q","Learned","Random","Δ"),
    ("2","0.847","0.742","+0.104"),
    ("6","0.927","0.814","+0.112"),
    ("10","0.913","0.794","+0.119"),
    ("14","0.917","0.826","+0.091"),
    ("18","0.900","0.809","+0.091"),
    ("22","0.897","0.836","+0.061"),
]
nrows, ncols = len(rows_data), 4
tbl_w = COL_W - 0.7
tbl_h = 5.55
gframe = slide.shapes.add_table(nrows, ncols, Inches(x2+0.35), Inches(tbl_top), Inches(tbl_w), Inches(tbl_h))
table = gframe.table
table.first_row = False
table.horz_banding = False
# disable default style banding by setting explicit fills
col_fracs = [0.18, 0.30, 0.30, 0.22]
for j, fr in enumerate(col_fracs):
    table.columns[j].width = Inches(tbl_w*fr)
for i, row in enumerate(rows_data):
    table.rows[i].height = Inches(tbl_h/nrows)
    for j, val in enumerate(row):
        cell = table.cell(i, j)
        cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.08)
        cell.margin_top = Inches(0.02); cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf2 = cell.text_frame
        tf2.word_wrap = False
        p = tf2.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        if i == 0:
            _set_run(r, val, 21, WHITE, bold=True)
            cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        else:
            isdelta = (j == 3)
            _set_run(r, val, 20, (ROYAL if isdelta else INK), bold=isdelta)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if (i % 2 == 1) else TINT
y += h + 0.45

# [5b] Structure visualized -- learned vs random mask grid
h = 7.2
top = section_box(x2, y, COL_W, h, None, "Structure, visualized — learned vs. random", bar_color=ROYAL)
gw = COL_W - 1.2
gx = x2 + (COL_W - gw)/2.0
g_h = pic_fit(os.path.join(BASE,"assets/lvr_grid.png"), gx, top+0.05, gw)
tb, tf = textbox(x2+0.35, top+g_h+0.14, COL_W-0.7, h-(top-y)-g_h-0.3)
body_box(tf, [
    {"segs":[("At the ", {"size":BS}),("same N₂q", {"size":BS,"bold":True}),
             (", greedy pruning keeps task-relevant interactions while random masks discard them — the ", {"size":BS}),
             ("pattern", {"size":BS,"bold":True}),(" of retained entanglers, not the count, drives accuracy.", {"size":BS})],
     "bullet":False, "sa":0},
])
y += h + 0.45

# ===================================================================
# COLUMN 3
# ===================================================================
x3 = COL_X[2]
y = CONTENT_TOP

# [6] Result 3 -- CENTERPIECE, amber accent
h = 16.6
top = section_box(x3, y, COL_W, h, 6, "Result 3 — On REAL hardware, compression becomes NECESSARY",
                  fill=AMBER_TINT, border=AMBER, bw=3.0, bar_color=NAVY, accent=None)
# amber outline emphasised: redraw a thin amber inner frame
fig_w = COL_W - 0.7
fig_h = pic_fit(os.path.join(BASE,"figures/fig_hardware_frontier.png"), x3+0.35, top+0.02, fig_w)
tb, tf = textbox(x3+0.35, top+fig_h+0.12, COL_W-0.7, 4.4)
body_box(tf, [
    {"segs":[("Trained circuits run on physical IBM ", {"size":21}),
             ("ibm_fez", {"size":21,"bold":True}),
             (" (156-qubit Heron), hard task, 5 device budgets, ", {"size":21}),
             ("16 test stimuli @ 2048 shots", {"size":21,"bold":True}),
             ("  (endpoints: 40 stimuli @ 4096 shots).", {"size":21})], "bullet":True, "sa":6},
    {"segs":[("In simulation the frontier is flat; ", {"size":21}),
             ("on hardware it falls monotonically", {"size":21,"bold":True}),
             (" with N₂q:  0.81 @ N₂q=2,6 → 0.75 @ 12 → ", {"size":21}),
             ("collapses to 0.56", {"size":21,"bold":True}),
             (" (near chance) @ N₂q=18,24.", {"size":21})], "bullet":True, "sa":6},
    {"segs":[("Endpoint confirmation:  ", {"size":21}),
             ("full circuit 0.60", {"size":21,"bold":True}),
             (" (sim 0.95) vs ", {"size":21}),
             ("compressed 0.83", {"size":21,"bold":True}),(" (sim 0.93).", {"size":21})], "bullet":True, "sa":0},
])
# amber callout box at bottom of section
call_h = 3.05
call_y = y + h - call_h - 0.3
rect(x3+0.35, call_y, COL_W-0.7, call_h, fill=AMBER, rounded=True, radius=0.05)
ctb, ctf = textbox(x3+0.55, call_y+0.12, COL_W-1.1, call_h-0.24, anchor=MSO_ANCHOR.MIDDLE)
rich_para(add_para(ctf, True),
          [("TAKEAWAY  ", {"size":22,"bold":True,"color":NAVY})],
          align=PP_ALIGN.LEFT, space_after=2)
rich_para(ctf.add_paragraph(),
          [("On real NISQ hardware, hardware-aware compression is not merely economical but ", {"size":21,"color":NAVY}),
           ("necessary", {"size":21,"bold":True,"color":NAVY}),
           (" — surplus entangling capacity is actively harmful, and task-aware pruning is what makes the learner usable at all.", {"size":21,"color":NAVY})],
          align=PP_ALIGN.LEFT, space_after=0, line=1.02)
y += h + 0.45

# [7] Robustness
h = 9.2
top = section_box(x3, y, COL_W, h, 7, "Robustness — not an artifact")
tb, tf = textbox(x3+0.35, top, COL_W-0.7, h-(top-y)-0.2)
body_box(tf, [
    {"segs":[("Device noise model", {"size":BS,"bold":True}),
             ("  (FakeManila, 4096 shots, 5 seeds): noisy accuracy tracks the noiseless frontier within a few % at every budget → savings genuine net of noise.", {"size":BS})], "bullet":True, "sa":7},
    {"segs":[("Density-matrix model", {"size":BS,"bold":True}),
             ("  (genuine mixed state, non-unital channel): reproduces every finding — flat frontier, collapse at N₂q=0, hard peak ", {"size":BS}),
             ("0.88 @ 6 vs 0.87 @ 24", {"size":BS,"bold":True}),
             (", learned > random (mean ", {"size":BS}),("+0.043", {"size":BS,"bold":True}),(").", {"size":BS})], "bullet":True, "sa":7},
    {"segs":[("Classical baselines", {"size":BS,"bold":True}),
             ("  (identical splits, 5 seeds): logistic regression at chance on every level; RBF-SVM & MLP fall monotonically easy→hard. Best compressed QBL: ", {"size":BS}),
             ("easy 0.970, medium 0.973, hard 0.927", {"size":BS,"bold":True}),(".", {"size":BS})], "bullet":True, "sa":0},
])
y += h + 0.45

# [8] Key takeaways
h = 7.0
top = section_box(x3, y, COL_W, h, 8, "Key takeaways", bar_color=ROYAL)
tb, tf = textbox(x3+0.35, top, COL_W-0.7, h-(top-y)-0.2)
body_box(tf, [
    {"segs":[("Most entanglement is ", {"size":BS}),("redundant", {"size":BS,"bold":True}),
             (" — sheds ~", {"size":BS}),("2/3", {"size":BS,"bold":True}),
             (" of two-qubit gates for ", {"size":BS}),("free", {"size":BS,"bold":True}),
             (", ", {"size":BS}),(">90%", {"size":BS,"bold":True}),(" at modest cost.", {"size":BS})], "bullet":True, "sa":6},
    {"segs":[("Entanglement is ", {"size":BS}),("necessary", {"size":BS,"bold":True}),
             (" — removing all of it collapses the learner to chance: a concrete ", {"size":BS}),
             ("capacity boundary", {"size":BS,"bold":True}),(".", {"size":BS})], "bullet":True, "sa":6},
    {"segs":[("Structure beats count", {"size":BS,"bold":True}),
             (" — learned masks beat random at every matched budget.", {"size":BS})], "bullet":True, "sa":6},
    {"segs":[("On real hardware, compression is what makes the learner work at all.", {"size":BS,"bold":True,"color":NAVY})], "bullet":True, "sa":0},
])
y += h + 0.45

# Footer: refs + future work + ack + QR  (two sub-areas)
foot_h = CONTENT_BOT - y
top = section_box(x3, y, COL_W, foot_h, None, "References  ·  Acknowledgments  ·  Links", bar_color=NAVY)
inner_top = top
# Left text region (refs + future + ack); right region QR
qr_w = 2.55
text_w = COL_W - 0.7 - qr_w - 0.35
tb, tf = textbox(x3+0.35, inner_top, text_w, foot_h-(inner_top-y)-0.2)
RS = 15.5
body_box(tf, [
    {"segs":[("Future work:  ", {"size":17,"bold":True,"color":NAVY}),
             ("Multi-qubit belief registers & higher-dim stimuli; richer non-unital evidence channels; hardware-aware objectives weighting depth, coherence & native fidelity; measured device error rates fed back into the compression objective.", {"size":16})],
     "bullet":False, "sa":7, "line":1.0},
    {"segs":[("[1] Busemeyer & Bruza, Quantum Models of Cognition and Decision, 2012.", {"size":RS})], "bullet":False, "sa":2, "line":0.98},
    {"segs":[("[2] Pothos & Chater, “A simplicity principle in unsupervised human categorization,” Cogn. Psych., 2002.", {"size":RS})], "bullet":False, "sa":2, "line":0.98},
    {"segs":[("[3] Kandala et al., “Hardware-efficient VQE,” Nature, 2017.", {"size":RS})], "bullet":False, "sa":2, "line":0.98},
    {"segs":[("[4] Preskill, “Quantum computing in the NISQ era and beyond,” Quantum, 2018.", {"size":RS})], "bullet":False, "sa":2, "line":0.98},
    {"segs":[("[5] Cerezo et al., “Variational quantum algorithms,” Nat. Rev. Phys., 2021.", {"size":RS})], "bullet":False, "sa":2, "line":0.98},
    {"segs":[("[6] Sim et al., “Adaptive pruning-based optimization of PQCs,” Quantum Sci. Technol., 2021.", {"size":RS})], "bullet":False, "sa":7, "line":0.98},
    {"segs":[("Acknowledgments:  ", {"size":16,"bold":True,"color":NAVY}),
             ("Device runs used IBM Quantum services. The author thanks Duke University’s Pratt School of Engineering.", {"size":16})], "bullet":False, "sa":0, "line":1.0},
])
# QR on the right
qr_x = x3 + COL_W - 0.35 - qr_w
qr_y = inner_top + 0.1
slide.shapes.add_picture(os.path.join(BASE,"assets/repo_qr.png"), Inches(qr_x), Inches(qr_y), Inches(qr_w), Inches(qr_w))
qtb, qtf = textbox(qr_x-0.25, qr_y+qr_w+0.04, qr_w+0.5, 1.2)
rich_para(add_para(qtf, True), [("Scan for code & data", {"size":14.5,"bold":True,"color":NAVY})],
          align=PP_ALIGN.CENTER, space_after=1, line=0.98)
rich_para(qtf.add_paragraph(), [("github.com/zoraizmohammad/\nqb-learner-compression", {"size":13,"color":ROYAL})],
          align=PP_ALIGN.CENTER, space_after=1, line=0.98)
rich_para(qtf.add_paragraph(), [("Mohammad Zoraiz · mz248@duke.edu", {"size":13,"color":INK})],
          align=PP_ALIGN.CENTER, space_after=0, line=0.98)

prs.save(OUT)
print("Saved", OUT)
print("Slide size EMU:", prs.slide_width, prs.slide_height,
      "=>", prs.slide_width/914400, "x", prs.slide_height/914400, "in")
