# Create a branded + PDF-export version of the Streamlit app.
# This version adds:
# - Branding controls (logo upload, brand colors, footer text)
# - Optional case studies section (CSV upload)
# - PDF export using ReportLab (if available), with graceful fallback
# - Simple HTML download as an alternative
#
# It reuses the same calculations as the base app by inlining the prior logic (trimmed for clarity).

app_code = r'''
# meta_ads_revenue_leak_report_app_branded.py
# -------------------------------------------
# Streamlit app to generate a branded client-facing "Meta Ads Revenue Leak Report"
# with PDF export (via ReportLab if installed) and optional case studies section.
#
# How to run locally:
#   1) pip install streamlit pandas numpy python-dateutil reportlab
#   2) streamlit run meta_ads_revenue_leak_report_app_branded.py
#
# Inputs mirror the non-branded app:
#  - meta_ads.csv, backend_orders.csv, web_analytics.csv (required/optional as described)
#  - pixel_events.csv, creative_stats.csv (optional)
#  - NEW: case_studies.csv (optional): columns: title, problem, approach, result
#
# Author: Your Agency Name
# License: MIT

import io
import base64
import textwrap
from datetime import datetime
from dateutil import parser
import numpy as np
import pandas as pd
import streamlit as st

# Try to import reportlab for PDF export; if not available, we fall back.
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

st.set_page_config(page_title="Meta Ads Revenue Leak Report (Branded)", layout="wide")

# ------------------------
# Utility Functions
# ------------------------
def _to_date(x):
    try:
        return parser.parse(str(x)).date()
    except Exception:
        return pd.NaT

def load_csv(uploaded_file, parse_dates=None):
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = df[col].apply(_to_date)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def pct(n):
    try:
        return f"{n:.1%}"
    except Exception:
        return "n/a"

def money(n):
    try:
        return f"${n:,.0f}"
    except Exception:
        return "$0"

def try_col(df, col, default=0.0):
    return df[col].apply(safe_float) if col in df.columns else pd.Series([default]*len(df))

def infer_period(meta_df, backend_df, web_df):
    dates = []
    for df in [meta_df, backend_df, web_df]:
        if df is not None and "date" in df.columns:
            dd = df["date"].dropna()
            if len(dd) > 0:
                dates.append((dd.min(), dd.max()))
    if not dates:
        return None, None
    start = min([d[0] for d in dates])
    end = max([d[1] for d in dates])
    return start, end

# ------------------------
# Sidebar: Branding
# ------------------------
st.sidebar.title("Branding")
agency_name = st.sidebar.text_input("Agency Name", "Your Agency Name")
primary_hex = st.sidebar.color_picker("Primary Color", "#1F6FEB")
secondary_hex = st.sidebar.color_picker("Secondary Color", "#111827")
accent_hex = st.sidebar.color_picker("Accent Color", "#06B6D4")
footer_text = st.sidebar.text_input("Footer Text", "© Your Agency Name — Confidential")
logo_file = st.sidebar.file_uploader("Logo (PNG/JPG)", type=["png","jpg","jpeg"])

st.sidebar.markdown("---")
st.sidebar.title("Upload Data")
meta_file = st.sidebar.file_uploader("Meta Ads (CSV/XLSX)", type=["csv","xlsx"])
backend_file = st.sidebar.file_uploader("Backend Orders (CSV/XLSX)", type=["csv","xlsx"])
web_file = st.sidebar.file_uploader("Web Analytics (CSV/XLSX)", type=["csv","xlsx"])
pixel_file = st.sidebar.file_uploader("Pixel Events (optional)", type=["csv","xlsx"])
creative_file = st.sidebar.file_uploader("Creative Stats (optional)", type=["csv","xlsx"])
case_file = st.sidebar.file_uploader("Case Studies (optional CSV)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.title("Report Settings")
client_name = st.sidebar.text_input("Client/Brand Name", "Acme Co.")
report_date = st.sidebar.date_input("Report Date", value=datetime.today())
prepared_by = st.sidebar.text_input("Prepared By", agency_name)

st.sidebar.markdown("---")
st.sidebar.subheader("Assumptions (Adjust)")
assumed_aov = st.sidebar.number_input("Average Order Value (AOV)", value=75.0, min_value=0.0, step=1.0)
uplift_obj_misalignment = st.sidebar.slider("Uplift if Objectives Fixed", 0.0, 0.5, 0.1, 0.01)
uplift_creative_fatigue = st.sidebar.slider("Uplift if Fatigue Fixed", 0.0, 0.5, 0.08, 0.01)
uplift_overlap = st.sidebar.slider("Spend Efficiency Gain (Overlap Fix)", 0.0, 0.4, 0.06, 0.01)
mobile_cr_target_factor = st.sidebar.slider("Mobile CR target vs Desktop CR", 0.1, 1.0, 0.6, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Thresholds")
ctr_low = st.sidebar.number_input("Low CTR threshold", value=0.007, format="%.4f")
freq_high = st.sidebar.number_input("High Frequency threshold", value=5.0, step=0.5)
min_spend_considered = st.sidebar.number_input("Min Spend to Consider (per ad)", value=50.0, step=10.0)

# ------------------------
# Load data
# ------------------------
meta_df = load_csv(meta_file, parse_dates=["date"])
backend_df = load_csv(backend_file, parse_dates=["date"])
web_df = load_csv(web_file, parse_dates=["date"])
pixel_df = load_csv(pixel_file, parse_dates=["date"])
creative_df = load_csv(creative_file, parse_dates=["first_seen_date","last_seen_date"])
case_df = load_csv(case_file)

# ------------------------
# Guard
# ------------------------
st.title("Meta Ads Revenue Leak Report — Branded")
st.caption("Upload client data and configure branding for a polished, client-ready report.")

if meta_df is None:
    st.info("Upload at least Meta Ads data to begin.")
    st.stop()

# Ensure key columns exist in Meta
for col in ["date","campaign_name","adset_name","ad_name","objective","spend","impressions","clicks","ctr","frequency","purchases","revenue"]:
    if col not in meta_df.columns:
        meta_df[col] = np.nan

meta_df["date"] = meta_df["date"].apply(_to_date)
meta_df["spend"] = try_col(meta_df, "spend", 0.0)
meta_df["revenue"] = try_col(meta_df, "revenue", 0.0)
meta_df["purchases"] = try_col(meta_df, "purchases", 0.0)
meta_df["ctr"] = try_col(meta_df, "ctr", 0.0)
meta_df["frequency"] = try_col(meta_df, "frequency", 0.0)
meta_df["objective"] = meta_df["objective"].fillna("")
meta_df["campaign_name"] = meta_df["campaign_name"].fillna("")
meta_df["adset_name"] = meta_df["adset_name"].fillna("")
meta_df["ad_name"] = meta_df["ad_name"].fillna("")
meta_df["audience_stage"] = meta_df["audience_stage"].fillna("") if "audience_stage" in meta_df.columns else ""

# Backend defaults
if backend_df is not None:
    for col in ["date","orders","backend_revenue","refunds","returns_cost","shipping_cost","product_cost","payment_fees"]:
        if col not in backend_df.columns:
            backend_df[col] = np.nan
    backend_df["date"] = backend_df["date"].apply(_to_date)
    backend_df["orders"] = try_col(backend_df, "orders", 0.0)
    backend_df["backend_revenue"] = try_col(backend_df, "backend_revenue", 0.0)
else:
    tmp = meta_df.groupby("date", dropna=True).agg({"purchases":"sum"}).reset_index()
    if len(tmp) == 0:
        from datetime import date
        tmp = pd.DataFrame({"date":[date.today()], "purchases":[0.0]})
    tmp["orders"] = tmp["purchases"]
    tmp["backend_revenue"] = tmp["orders"] * assumed_aov
    backend_df = tmp[["date","orders","backend_revenue"]].copy()

# Web analytics defaults
if web_df is not None:
    for col in ["date","device","sessions","conversion_rate","add_to_cart_rate","checkout_rate","purchase_rate","avg_page_load_time","bounce_rate"]:
        if col not in web_df.columns:
            web_df[col] = np.nan
    web_df["date"] = web_df["date"].apply(_to_date)
else:
    web_df = pd.DataFrame(columns=["date","device","sessions","conversion_rate","add_to_cart_rate","checkout_rate","purchase_rate","avg_page_load_time","bounce_rate"])

# Pixel defaults
if pixel_df is not None:
    for col in ["date","event_name","event_id","device","url"]:
        if col not in pixel_df.columns:
            pixel_df[col] = np.nan
    pixel_df["date"] = pixel_df["date"].apply(_to_date)

# Creative defaults
if creative_df is not None:
    for col in ["ad_name","first_seen_date","last_seen_date","spend","impressions","clicks","ctr","purchases","revenue"]:
        if col not in creative_df.columns:
            creative_df[col] = np.nan

# ------------------------
# Period
# ------------------------
def infer_period(meta_df, backend_df, web_df):
    dates = []
    for df in [meta_df, backend_df, web_df]:
        if df is not None and "date" in df.columns:
            dd = df["date"].dropna()
            if len(dd) > 0:
                dates.append((dd.min(), dd.max()))
    if not dates:
        return None, None
    start = min([d[0] for d in dates])
    end = max([d[1] for d in dates])
    return start, end

period_start, period_end = infer_period(meta_df, backend_df, web_df)
period_str = f"{period_start} to {period_end}" if period_start and period_end else "Selected Period"

# ------------------------
# 1) Tracking & Attribution Errors
# ------------------------
meta_rev = meta_df["revenue"].sum()
backend_rev = backend_df["backend_revenue"].sum()
tracking_diff = meta_rev - backend_rev
tracking_diff_pct = (tracking_diff/backend_rev) if backend_rev > 0 else 0.0

dup_ratio = None
if pixel_df is not None and len(pixel_df) > 0:
    purchases_events = pixel_df[pixel_df["event_name"].fillna("").str.lower().str.contains("purchase")]
    if "event_id" in purchases_events.columns and purchases_events["event_id"].notna().any():
        total_purchase_events = len(purchases_events)
        unique_purchase_events = purchases_events["event_id"].nunique()
        if unique_purchase_events > 0:
            dup_ratio = (total_purchase_events - unique_purchase_events) / unique_purchase_events

# ------------------------
# 2) Objective Misalignment
# ------------------------
meta_df["is_bofu_like"] = False
if "audience_stage" in meta_df.columns and isinstance(meta_df["audience_stage"], pd.Series):
    meta_df["is_bofu_like"] = meta_df["audience_stage"].str.upper().isin(["MOFU","BOFU"])
mask_rt = meta_df["campaign_name"].str.lower().str.contains("retarget|remarket|cart|viewcontent|view content|dpa|catalog", regex=True)
meta_df.loc[mask_rt, "is_bofu_like"] = True

meta_df["objective_clean"] = meta_df["objective"].str.lower()
is_purchase_objective = meta_df["objective_clean"].str.contains("purchase|sales|conversions", regex=True)

bofu_spend = meta_df.loc[meta_df["is_bofu_like"], "spend"].sum()
bofu_spend_wrong_obj = meta_df.loc[meta_df["is_bofu_like"] & (~is_purchase_objective), "spend"].sum()
obj_misalignment_spend_pct = (bofu_spend_wrong_obj / bofu_spend) if bofu_spend > 0 else 0.0

# ------------------------
# 3) Creative Fatigue & Underperforming Ads
# ------------------------
ad_perf = meta_df.groupby("ad_name", dropna=True).agg(
    spend=("spend","sum"),
    ctr=("ctr","mean"),
    freq=("frequency","mean"),
    purchases=("purchases","sum"),
    revenue=("revenue","sum"),
    impressions=("impressions","sum"),
    clicks=("clicks","sum")
).reset_index()

ctr_low = float(ctr_low)
freq_high = float(freq_high)
min_spend_considered = float(min_spend_considered)

ad_perf["is_underperformer"] = (ad_perf["ctr"] < ctr_low) & (ad_perf["freq"] > freq_high) & (ad_perf["spend"] >= min_spend_considered)
underperf_spend = ad_perf.loc[ad_perf["is_underperformer"], "spend"].sum()
total_spend = ad_perf["spend"].sum() if len(ad_perf) else 0.0
underperf_spend_pct = (underperf_spend / total_spend) if total_spend > 0 else 0.0

# ------------------------
# 4) Audience Overlap (Heuristic)
# ------------------------
if "audience_name" in meta_df.columns:
    aud_counts = meta_df.groupby(["campaign_name","audience_name"]).size().reset_index(name="count")
    duplicated_audiences = aud_counts.groupby("audience_name")["campaign_name"].nunique()
    overlap_risky_aud = duplicated_audiences[duplicated_audiences > 1].index.tolist()
    overlap_spend = meta_df[meta_df["audience_name"].isin(overlap_risky_aud)]["spend"].sum()
else:
    overlap_spend = 0.0
overlap_spend_pct = (overlap_spend / total_spend) if total_spend > 0 else 0.0

# ------------------------
# 5) Landing Page / Checkout Drop-Off
# ------------------------
mobile_cr = desktop_cr = None
mobile_sessions = desktop_sessions = 0.0
if len(web_df) > 0 and "device" in web_df.columns:
    if "conversion_rate" in web_df.columns:
        mobile_cr = web_df.loc[web_df["device"].str.lower()=="mobile", "conversion_rate"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().mean()
        desktop_cr = web_df.loc[web_df["device"].str.lower()=="desktop", "conversion_rate"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().mean()
    if "sessions" in web_df.columns:
        mobile_sessions = web_df.loc[web_df["device"].str.lower()=="mobile", "sessions"].astype(float).sum()
        desktop_sessions = web_df.loc[web_df["device"].str.lower()=="desktop", "sessions"].astype(float).sum()

avg_page_load_mobile = None
if "avg_page_load_time" in web_df.columns and "device" in web_df.columns:
    avg_page_load_mobile = web_df.loc[web_df["device"].str.lower()=="mobile", "avg_page_load_time"].astype(float).dropna().mean()

mobile_loss = 0.0
if (mobile_cr is not None) and (desktop_cr is not None) and (mobile_sessions > 0):
    target_mobile_cr = desktop_cr * float(mobile_cr_target_factor)
    if target_mobile_cr > mobile_cr:
        delta_cr = max(0.0, target_mobile_cr - mobile_cr)
        extra_orders = delta_cr * mobile_sessions
        mobile_loss = extra_orders * float(assumed_aov)
else:
    if backend_rev > 0 and len(web_df) > 0 and "sessions" in web_df.columns:
        total_sessions = web_df["sessions"].astype(float).sum()
        if total_sessions > 0:
            mobile_share = mobile_sessions / total_sessions
            mobile_loss = backend_rev * mobile_share * 0.1

# ------------------------
# Recovery Estimates
# ------------------------
recover_obj = max(0.0, backend_rev * obj_misalignment_spend_pct * float(uplift_obj_misalignment))
recover_creative = max(0.0, backend_rev * underperf_spend_pct * float(uplift_creative_fatigue))
recover_overlap = max(0.0, total_spend * float(uplift_overlap))
recover_checkout = max(0.0, mobile_loss)
total_recoverable = recover_obj + recover_creative + recover_overlap + recover_checkout

# ------------------------
# Build Markdown Report
# ------------------------
period_str = f"{period_start} to {period_end}" if (period_start and period_end) else "Selected Period"

exec_summary = f"""
# 🚨 Meta Ads Revenue Leak Report  
**Client:** {client_name}  
**Date:** {report_date.strftime('%Y-%m-%d')}  
**Prepared By:** {prepared_by}

## Executive Summary
Our audit of your Meta Ads account and e-commerce funnel (period **{period_str}**) uncovered **5 major revenue leaks** causing wasted spend and lost conversions.

If resolved, these leaks could recover an estimated **{money(total_recoverable)} per month** (conservative assumptions; see details).

This report details:
1. The top 5 leaks
2. Supporting evidence (data & diagnostics)
3. Actionable recommendations prioritized by impact vs effort
"""

leak1 = f"""
## 📊 Leak #1: Tracking & Attribution Errors
**Issue:** Meta-reported revenue = **{money(meta_rev)}** vs backend revenue = **{money(backend_rev)}** → difference **{money(tracking_diff)}** ({pct(tracking_diff_pct)}).
{"**Possible duplicate purchase events detected in pixel data.**" if (dup_ratio is not None and dup_ratio > 0.05) else ""}

**Impact:** Risk of over/under-spend and misallocation driven by inflated/deflated ROAS.

**Recommendations:**
- Implement **server-side Conversions API** and standardize attribution window (e.g., 7-day click, 1-day view).
- Audit and de-duplicate event triggers (ensure one Purchase event per order).
- Align source of truth (backend) with ad optimization metrics.
"""

leak2 = f"""
## 📊 Leak #2: Campaign Structure & Objective Misalignment
**Finding:** {pct(obj_misalignment_spend_pct)} of BOFU/retargeting spend is not optimized for **Purchase** (or Sales/Conversions).

**Impact:** Meta optimizes toward cheaper non-buyers, inflating clicks and add-to-carts but reducing purchases.

**Recommendations:**
- Switch all BOFU campaigns to **Purchase** optimization.
- Consolidate overlapping ad sets; clarify TOFU/MOFU/BOFU funnel separation.
- Apply budget to proven audiences and objectives.
**Estimated Recoverable Revenue:** {money(recover_obj)}
"""

leak3 = f"""
## 📊 Leak #3: Creative Fatigue & Underperforming Ads
**Finding:** {pct(underperf_spend_pct)} of spend went to underperforming ads (CTR < {float(ctr_low):.2%}, Frequency > {float(freq_high)}).

**Impact:** Audience fatigue → rising CPMs and falling ROAS.

**Recommendations:**
- Pause underperformers and rotate **new creative** (UGC, product demo, offer stack) every 14 days.
- Track CTR/frequency degradation and set auto-pause thresholds.
**Estimated Recoverable Revenue:** {money(recover_creative)}
"""

leak4 = f"""
## 📊 Leak #4: Audience Overlap & Wasted Spend (Heuristic)
**Finding:** Approx. {pct(overlap_spend_pct)} of spend is at risk due to audience duplication across campaigns/ad sets.

**Impact:** Paying multiple times for the same people → inefficient reach and cannibalized performance.

**Recommendations:**
- Implement **mutual exclusions** across funnel stages (TOFU excludes MOFU/BOFU, etc.).
- Seed lookalikes with **high-value buyers** (e.g., AOV > {money(assumed_aov)}). Test Advantage+ Shopping.
**Estimated Recoverable Revenue:** {money(recover_overlap)}
"""

lp_speed_txt = f", avg mobile page load ≈ {round(avg_page_load_mobile,2)}s" if avg_page_load_mobile is not None else ""
cr_txt = ""
if (mobile_cr is not None) and (desktop_cr is not None):
    cr_txt = f"Mobile CR {pct(mobile_cr)} vs Desktop CR {pct(desktop_cr)}"

leak5 = f"""
## 📊 Leak #5: Landing Page & Checkout Drop-Off
**Finding:** {cr_txt}{lp_speed_txt if lp_speed_txt else ""}

**Impact:** Lost mobile buyers (typically majority of traffic).

**Recommendations:**
- Compress media, implement lazy loading; aim for mobile load < 2.5s.
- Simplify checkout (≤ 2 steps) and ensure offer parity with ad messaging.
- Deep-link ads to the most relevant PDP/collection.

**Estimated Recoverable Revenue:** {money(recover_checkout)}
"""

recovery_table = f"""
## 📈 Potential Revenue Recovery
| Leak | Monthly Recovery (Est.) | Priority |
|------|-------------------------|----------|
| Tracking & Attribution | *n/a (risk mitigation)* | High |
| Campaign Objectives | {money(recover_obj)} | High |
| Creative Fatigue | {money(recover_creative)} | High |
| Audience Overlap | {money(recover_overlap)} | Medium |
| Checkout Drop-off | {money(recover_checkout)} | High |

**Total Recoverable Revenue:** **{money(total_recoverable)}** per month
"""

plan = f"""
## 🔑 Next Steps (30/60/90)
**Next 30 Days**
- Fix pixel + Conversions API + attribution windows
- Switch BOFU campaigns to Purchase optimization
- Launch new creatives and set rotation cadence

**Next 60 Days**
- Redesign retargeting with exclusions
- Optimize mobile checkout (UX + speed)

**Next 90 Days**
- Test scaling (Advantage+ Shopping, budget re-allocation)
- Implement monitoring dashboard for leak prevention
"""

bonus = f"""
## 📌 Bonus Insights
- **Benchmarks:** Compare current ROAS and funnel metrics to your niche averages to set targets.
- **Quick Win:** Fixing duplicate purchase events stabilizes reporting immediately.
- **Ongoing Option:** Monthly Leak Monitoring & Optimization for continuous gains.
"""

case_section = ""
if case_df is not None and len(case_df) > 0:
    rows = []
    for _, r in case_df.iterrows():
        t = str(r.get("title","Case Study"))
        p = str(r.get("problem",""))
        a = str(r.get("approach",""))
        res = str(r.get("result",""))
        rows.append(f"### {t}\n**Problem:** {p}\n\n**Approach:** {a}\n\n**Result:** {res}\n")
    case_section = "## 🧪 Case Studies\n" + "\n".join(rows)

report_md = exec_summary + "\n---\n" + leak1 + "\n---\n" + leak2 + "\n---\n" + leak3 + "\n---\n" + leak4 + "\n---\n" + leak5 + "\n---\n" + recovery_table + "\n---\n" + plan + "\n---\n" + bonus + ("\n---\n" + case_section if case_section else "")

# ------------------------
# Header with Brand
# ------------------------
left, mid, right = st.columns([1,3,1])
with left:
    if logo_file is not None:
        st.image(logo_file, caption=agency_name, use_column_width=True)
with mid:
    st.markdown(f"<h2 style='color:{primary_hex};margin-bottom:0;'>Meta Ads Revenue Leak Report</h2>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{secondary_hex};'><b>Client:</b> {client_name} &nbsp;&nbsp; <b>Date:</b> {report_date.strftime('%Y-%m-%d')}</div>", unsafe_allow_html=True)
with right:
    st.markdown(f"<div style='text-align:right;color:{accent_hex};'><b>Prepared by:</b><br>{prepared_by}</div>", unsafe_allow_html=True)

st.markdown("---")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Meta Revenue (period)", money(meta_rev))
col2.metric("Backend Revenue (period)", money(backend_rev))
col3.metric("Tracking Diff", money(tracking_diff))
col4.metric("Total Recoverable (est.)", money(total_recoverable))

# Report body
st.subheader("Generated Report (Markdown)")
st.markdown(report_md)

# -------------
# Downloads
# -------------
# Markdown download
md_bytes = report_md.encode("utf-8")
st.download_button(
    label="⬇️ Download Report (Markdown)",
    data=md_bytes,
    file_name=f"{client_name.replace(' ','_').lower()}_meta_ads_report_{report_date.strftime('%Y%m%d')}.md",
    mime="text/markdown",
)

# Simple HTML download (with branding colors)
html_report = f"""
<html>
<head>
<meta charset="utf-8">
<title>Meta Ads Revenue Leak Report</title>
</head>
<body style="font-family:Arial,Helvetica,sans-serif;max-width:900px;margin:40px auto;line-height:1.5;">
<h1 style="color:{primary_hex};">Meta Ads Revenue Leak Report</h1>
<p><b>Client:</b> {client_name} &nbsp; <b>Date:</b> {report_date.strftime('%Y-%m-%d')} &nbsp; <b>Prepared by:</b> {prepared_by}</p>
<hr/>
{report_md.replace("\n","<br/>\n")}
<hr/>
<div style="color:#666;">{footer_text}</div>
</body>
</html>
""".encode("utf-8")

st.download_button(
    label="⬇️ Download Report (HTML)",
    data=html_report,
    file_name=f"{client_name.replace(' ','_').lower()}_meta_ads_report_{report_date.strftime('%Y%m%d')}.html",
    mime="text/html",
)

# PDF download via ReportLab if available
def build_pdf(buffer, logo_bytes=None):
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", fontSize=18, leading=22, spaceAfter=8, textColor=colors.HexColor(primary_hex)))
    styles.add(ParagraphStyle(name="H2", fontSize=14, leading=18, spaceBefore=8, spaceAfter=6, textColor=colors.HexColor(secondary_hex)))
    styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="Small", fontSize=8, leading=10, textColor=colors.grey))

    story = []

    # Header
    if logo_bytes:
        try:
            img = RLImage(io.BytesIO(logo_bytes), width=60*mm, height=20*mm)
            story.append(img)
            story.append(Spacer(1, 5*mm))
        except Exception:
            pass

    story.append(Paragraph("Meta Ads Revenue Leak Report", styles["H1"]))
    story.append(Paragraph(f"<b>Client:</b> {client_name} &nbsp;&nbsp; <b>Date:</b> {report_date.strftime('%Y-%m-%d')} &nbsp;&nbsp; <b>Prepared by:</b> {prepared_by}", styles["Body"]))
    story.append(Spacer(1, 5*mm))

    # Split markdown sections naively by '##' to create paragraphs
    sections = report_md.split("## ")
    # First chunk may include the main title; add as body
    if sections:
        story.append(Paragraph(sections[0].replace("\n","<br/>"), styles["Body"]))
        story.append(Spacer(1, 4*mm))

    for sec in sections[1:]:
        lines = sec.splitlines()
        header = lines[0].strip()
        body = "\n".join(lines[1:]).replace("\n","<br/>")
        story.append(Paragraph(header, styles["H2"]))
        story.append(Paragraph(body, styles["Body"]))
        story.append(Spacer(1, 3*mm))

    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(footer_text, styles["Small"]))

    doc.build(story)

if REPORTLAB_AVAILABLE:
    logo_bytes = None
    if logo_file is not None:
        logo_bytes = logo_file.read()
    pdf_buffer = io.BytesIO()
    build_pdf(pdf_buffer, logo_bytes=logo_bytes)
    st.download_button(
        label="⬇️ Download Report (PDF)",
        data=pdf_buffer.getvalue(),
        file_name=f"{client_name.replace(' ','_').lower()}_meta_ads_report_{report_date.strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
    )
else:
    st.warning("ReportLab not installed. To enable PDF export, run: pip install reportlab")

# Diagnostics Expanders
with st.expander("Diagnostics: Objective Alignment & Creative Performance"):
    st.write("**BOFU Spend (est.)**:", money(bofu_spend))
    st.write("**BOFU Spend on Wrong Objective**:", money(bofu_spend_wrong_obj))
    st.write("**Underperforming Creative Spend %**:", pct(underperf_spend_pct))
    st.dataframe(ad_perf.sort_values("spend", ascending=False))

with st.expander("Diagnostics: Audience Overlap Heuristic"):
    if "audience_name" in meta_df.columns:
        st.write("Approx. overlap spend:", money(overlap_spend))
        sample = meta_df[["campaign_name","adset_name","audience_name","spend"]].dropna().head(200)
        st.dataframe(sample)
    else:
        st.info("No 'audience_name' column found to assess overlap.")

with st.expander("Diagnostics: Web & Checkout"):
    if len(web_df) > 0:
        if (mobile_cr is not None) and (desktop_cr is not None):
            st.write("Mobile CR:", pct(mobile_cr), "Desktop CR:", pct(desktop_cr))
        if avg_page_load_mobile is not None:
            st.write("Avg Mobile Page Load (s):", round(avg_page_load_mobile,2))
        st.dataframe(web_df.tail(200))
    else:
        st.info("Upload web analytics CSV for deeper funnel insights.")

with st.expander("Notes & Assumptions"):
    st.markdown(f"""
    - **AOV** used for calculations: {money(assumed_aov)}  
    - **Objective Uplift** assumption: {pct(uplift_obj_misalignment)} applied to BOFU spend share  
    - **Creative Uplift** assumption: {pct(uplift_creative_fatigue)} applied to underperformer spend share  
    - **Overlap Efficiency** assumption: {pct(uplift_overlap)} applied to total spend  
    - **Mobile CR Target**: {pct(mobile_cr_target_factor)} × Desktop CR  
    - Tracking fixes are treated as **risk mitigation** (do not count as revenue directly).
    - Colors: Primary {primary_hex}, Secondary {secondary_hex}, Accent {accent_hex}.
    """)
'''

with open('/mnt/data/meta_ads_revenue_leak_report_app_branded.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

'/mnt/data/meta_ads_revenue_leak_report_app_branded.py'
