# streamlit_app.py
# ----------------
# Meta/Google Ads Revenue Leak Finder — Plain Language, FOMO Edition
#
# How to run:
#   pip install streamlit pandas numpy python-dateutil reportlab
#   streamlit run streamlit_app.py

import io
from datetime import date
from dateutil import parser
import numpy as np
import pandas as pd
import streamlit as st

# Optional PDF export
REPORTLAB_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

st.set_page_config(page_title="Revenue Leak Finder (Plain, FOMO)", layout="wide")

# ------------------------
# Helpers
# ------------------------
def _to_date(x):
    try:
        return parser.parse(str(x)).date()
    except Exception:
        return pd.NaT

def load_csv(uploaded_file, parse_dates=None):
    if uploaded_file is None:
        return None
    # Read as CSV first; fall back to Excel if needed
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        try:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
        except Exception:
            return None
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = df[col].apply(_to_date)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def money(n):
    try:
        return f"${n:,.0f}"
    except Exception:
        return "$0"

def pct(n):
    try:
        return f"{n:.1%}"
    except Exception:
        return "n/a"

def get_first(df, cols, default=None):
    for c in cols:
        if c and c in df.columns:
            return c
    return None

def series_or_default(df, cols, default=0.0):
    for c in cols:
        if c and c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return pd.Series([default]*len(df))

# ------------------------
# Sidebar — Inputs
# ------------------------
st.sidebar.title("Inputs")
platform = st.sidebar.radio("Ad Platform", ["Meta Ads", "Google Ads"], index=0)
ads_file = st.sidebar.file_uploader("Ads CSV (Meta OR Google)", type=["csv","txt"], help="Upload ONE CSV for ads data.")
backend_file = st.sidebar.file_uploader("Backend Orders CSV (optional)", type=["csv","txt"])
web_file = st.sidebar.file_uploader("Web Analytics CSV (optional)", type=["csv","txt"])

st.sidebar.markdown("---")
st.sidebar.title("Assumptions (tweak if needed)")
assumed_aov = st.sidebar.number_input("Average Order Value", value=75.0, step=1.0)
uplift_creative = st.sidebar.slider("Gain from refreshing tired ads", 0.0, 0.5, 0.10, 0.01)
uplift_overlap = st.sidebar.slider("Gain from fixing audience overlap", 0.0, 0.4, 0.06, 0.01)
mobile_cr_target_factor = st.sidebar.slider("Target Mobile CR vs Desktop", 0.1, 1.0, 0.6, 0.05)

st.sidebar.markdown("---")
st.sidebar.title("Thresholds")
ctr_low = st.sidebar.number_input("Low CTR threshold (tired ads)", value=0.008, format="%.4f")
freq_high = st.sidebar.number_input("High Frequency threshold", value=5.0, step=0.5)
min_spend_considered = st.sidebar.number_input("Min Spend per Ad to count", value=50.0, step=10.0)

# ------------------------
# Header
# ------------------------
st.title("Revenue Leak Finder — Simple, Actionable, FOMO Ready")
st.caption("Focus: where money is leaking, how much, and what to do right now. No jargon, no graphs.")

if ads_file is None:
    st.info("Upload your **Ads CSV** (Meta OR Google) to get started. Optional: Backend Orders CSV, Web Analytics CSV.")
    st.stop()

# ------------------------
# Load data
# ------------------------
ads_df = load_csv(ads_file, parse_dates=["date"])
if ads_df is None or ads_df.empty:
    st.error("Could not read your Ads CSV. Please upload a valid CSV/TXT.")
    st.stop()

backend_df = load_csv(backend_file, parse_dates=["date"]) if backend_file else None
web_df = load_csv(web_file, parse_dates=["date"]) if web_file else None

# ------------------------
# Normalize columns
# ------------------------
spend_col = get_first(ads_df, ["spend","cost","amount_spent","ad_cost"])
imp_col = get_first(ads_df, ["impressions","impr"])
clk_col = get_first(ads_df, ["clicks","click"])
ctr_col = get_first(ads_df, ["ctr","click_through_rate"])
freq_col = get_first(ads_df, ["frequency","avg_frequency"])
purch_col = get_first(ads_df, ["purchases","conversions","orders","transactions"])
rev_col = get_first(ads_df, ["revenue","conversion_value","value","purchase_value"])
campaign_col = get_first(ads_df, ["campaign_name","campaign"])
adset_col = get_first(ads_df, ["adset_name","ad_group","adgroup","ad_group_name"])
ad_col = get_first(ads_df, ["ad_name","ad","creative"])
objective_col = get_first(ads_df, ["objective","campaign_objective"])
aud_stage_col = get_first(ads_df, ["audience_stage","funnel_stage"])
aud_name_col = get_first(ads_df, ["audience_name","audience","segment"])

U = pd.DataFrame()
date_col = get_first(ads_df, ["date","day","day_date"])
U["date"] = ads_df[date_col] if date_col else pd.NaT
U["campaign_name"] = ads_df[campaign_col] if campaign_col else "Unknown"
U["adset_name"] = ads_df[adset_col] if adset_col else "Unknown"
U["ad_name"] = ads_df[ad_col] if ad_col else "Unknown"
U["objective"] = ads_df[objective_col] if objective_col else ""
U["audience_stage"] = ads_df[aud_stage_col] if aud_stage_col else ""
U["audience_name"] = ads_df[aud_name_col] if aud_name_col else ""

U["spend"] = series_or_default(ads_df, [spend_col], 0.0)
U["impressions"] = series_or_default(ads_df, [imp_col], 0.0)
U["clicks"] = series_or_default(ads_df, [clk_col], 0.0)
if ctr_col:
    U["ctr"] = pd.to_numeric(ads_df[ctr_col], errors="coerce").fillna(0.0)
else:
    U["ctr"] = (U["clicks"] / U["impressions"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)
U["frequency"] = series_or_default(ads_df, [freq_col], 0.0)
U["purchases"] = series_or_default(ads_df, [purch_col], 0.0)
U["revenue_ads"] = series_or_default(ads_df, [rev_col], 0.0)

is_meta = (platform == "Meta Ads")
is_google = (platform == "Google Ads")

# ------------------------
# Period & aggregates
# ------------------------
period_start = pd.to_datetime(U["date"]).dropna().min() if "date" in U.columns else None
period_end = pd.to_datetime(U["date"]).dropna().max() if "date" in U.columns else None
period_str = f"{period_start.date()} to {period_end.date()}" if pd.notna(period_start) and pd.notna(period_end) else "Selected Period"

total_spend = float(U["spend"].sum())
ads_reported_rev = float(U["revenue_ads"].sum())
if ads_reported_rev == 0 and U["purchases"].sum() > 0:
    ads_reported_rev = float(U["purchases"].sum() * assumed_aov)

backend_rev = None
if backend_df is not None and not backend_df.empty:
    b_rev_col = get_first(backend_df, ["backend_revenue","revenue","net_revenue"])
    if b_rev_col:
        backend_rev = float(pd.to_numeric(backend_df[b_rev_col], errors="coerce").fillna(0.0).sum())

current_revenue = float(backend_rev if backend_rev and backend_rev > 0 else ads_reported_rev)

# ------------------------
# Leak calculations
# ------------------------
leaks = []

# 1) Tracking mismatch (if backend present)
if backend_rev and backend_rev > 0:
    tracking_diff = ads_reported_rev - backend_rev
    tracking_diff_pct = tracking_diff / backend_rev if backend_rev > 0 else 0.0
    leaks.append({
        "name":"Tracking/Reporting Mismatch",
        "category":"Tracking Setup",
        "where":"Pixel & Conversions / Measurement",
        "root":"Ads platform reports more/less sales than your backend (duplicate/missing events or settings).",
        "impact":0.0,  # treat as risk, not added to $ recovery
        "impact_note":f"Risk: {money(tracking_diff)} difference ({pct(tracking_diff_pct)}). Fix to avoid bad decisions.",
        "actions":[
            "Turn on server-side Conversions API (Meta) or Enhanced Conversions (Google).",
            "Ensure only one Purchase/Conversion event fires per order.",
            "Use a consistent attribution window (e.g., 7-day click, 1-day view).",
            "Judge real sales by backend, not ad platform alone."
        ]
    })

# 2) Wrong objective (Meta retargeting not set to Purchase)
if is_meta:
    U["is_bofu_like"] = False
    if "audience_stage" in U.columns and U["audience_stage"].dtype == object:
        U["is_bofu_like"] = U["audience_stage"].str.upper().isin(["MOFU","BOFU"])
    mask_rt = U["campaign_name"].str.lower().str.contains("retarget|remarket|cart|viewcontent|view content|dpa|catalog", regex=True)
    U.loc[mask_rt, "is_bofu_like"] = True

    obj = U["objective"].astype(str).str.lower()
    is_purchase_objective = obj.str.contains("purchase|sales|conversions")
    bofu_spend = float(U.loc[U["is_bofu_like"], "spend"].sum())
    bofu_wrong = float(U.loc[U["is_bofu_like"] & (~is_purchase_objective), "spend"].sum())
    misalign_share = (bofu_wrong / bofu_spend) if bofu_spend > 0 else 0.0
    impact_obj = current_revenue * misalign_share * 0.10  # 10% conservative
    leaks.append({
        "name":"Wrong Goal for Retargeting",
        "category":"Ad Account Setup",
        "where":"Retargeting campaigns (bottom of funnel)",
        "root":"Campaigns are set to Clicks/Traffic instead of Purchase, so the system finds clickers not buyers.",
        "impact":max(0.0, impact_obj),
        "impact_note":f"~{pct(misalign_share)} of retargeting spend is on the wrong goal.",
        "actions":[
            "Switch all retargeting campaigns to **Purchase/Sales/Conversions** objective.",
            "Separate prospecting (TOFU) from retargeting (BOFU).",
            "Move budget toward campaigns with actual purchases."
        ]
    })

# 3) Tired creatives (low CTR + high frequency with enough spend)
ad_perf = U.groupby("ad_name", dropna=True).agg(spend=("spend","sum"),
                                                ctr=("ctr","mean"),
                                                freq=("frequency","mean")).reset_index()
ad_perf["is_underperformer"] = (ad_perf["ctr"] < float(ctr_low)) & \
                               (ad_perf["freq"] > float(freq_high)) & \
                               (ad_perf["spend"] >= float(min_spend_considered))
under_spend = float(ad_perf.loc[ad_perf["is_underperformer"], "spend"].sum())
under_share = (under_spend / total_spend) if total_spend > 0 else 0.0
impact_creative = current_revenue * under_share * float(uplift_creative)
leaks.append({
    "name":"Tired/Weak Ads",
    "category":"Creative",
    "where":"Ad level",
    "root":"Same ads shown too often or not compelling → low click-through and fewer buyers.",
    "impact":max(0.0, impact_creative),
    "impact_note":f"{pct(under_share)} of spend went to tired ads (CTR < {ctr_low:.2%} & Frequency > {freq_high}).",
    "actions":[
        "Pause low-CTR/high-frequency ads immediately.",
        "Launch 3 new variations (customer-style video, quick product demo, clear offer).",
        "Rotate creatives every 14 days or when frequency passes limit."
    ]
})

# 4) Audience overlap (if audience names present)
overlap_impact = 0.0
overlap_note = "No audience names found; overlap risk estimated as low."
if "audience_name" in U.columns and U["audience_name"].astype(str).str.len().max() > 0:
    aud_counts = U.groupby(["campaign_name","audience_name"]).size().reset_index(name="count")
    dupe = aud_counts.groupby("audience_name")["campaign_name"].nunique()
    risky_aud = dupe[dupe > 1].index.tolist()
    overlap_spend = float(U[U["audience_name"].isin(risky_aud)]["spend"].sum())
    overlap_share = (overlap_spend / total_spend) if total_spend > 0 else 0.0
    overlap_impact = current_revenue * overlap_share * float(uplift_overlap)
    overlap_note = f"Approx. {pct(overlap_share)} of spend hits the same audiences across multiple campaigns."
leaks.append({
    "name":"Paying Twice for the Same People",
    "category":"Audience",
    "where":"Audience targeting across campaigns",
    "root":"Different campaigns chase the same people at the same time → wasted spend.",
    "impact":max(0.0, overlap_impact),
    "impact_note":overlap_note,
    "actions":[
        "Add exclusions between stages: TOFU excludes MOFU/BOFU audiences.",
        "Use lookalikes from your best buyers (high AOV).",
        "Use simplified structures (e.g., Advantage+ or consolidated campaigns) and monitor overlap."
    ]
})

# 5) Mobile drop-offs (prefer web analytics; else estimate)
mobile_impact = 0.0
mobile_note = "Estimated potential based on typical mobile issues."
if web_df is not None and not web_df.empty:
    dev_col = get_first(web_df, ["device"])
    ses_col = get_first(web_df, ["sessions"])
    cr_col = get_first(web_df, ["conversion_rate","purchase_rate"])
    load_col = get_first(web_df, ["avg_page_load_time"])
    if dev_col and ses_col and cr_col:
        w = web_df.copy()
        w[dev_col] = w[dev_col].astype(str).str.lower()
        mobile_cr = pd.to_numeric(w.loc[w[dev_col]=="mobile", cr_col], errors="coerce").dropna().mean()
        desktop_cr = pd.to_numeric(w.loc[w[dev_col]=="desktop", cr_col], errors="coerce").dropna().mean()
        mobile_sessions = pd.to_numeric(w.loc[w[dev_col]=="mobile", ses_col], errors="coerce").dropna().sum()
        target_mobile = desktop_cr * float(mobile_cr_target_factor) if desktop_cr and not np.isnan(desktop_cr) else None
        if target_mobile and mobile_cr and mobile_sessions and (target_mobile > mobile_cr):
            delta = max(0.0, target_mobile - mobile_cr)
            extra_orders = delta * mobile_sessions
            mobile_impact = extra_orders * float(assumed_aov)
        if load_col and load_col in w.columns:
            avg_mobile_load = pd.to_numeric(w.loc[w[dev_col]=="mobile", load_col], errors="coerce").dropna().mean()
            if avg_mobile_load and avg_mobile_load > 3.0:
                mobile_note = f"Mobile pages ~{avg_mobile_load:.1f}s. Faster pages convert better."
        else:
            mobile_note = "Mobile conversion lags desktop. Speed/checkout fixes can lift results."
else:
    mobile_impact = current_revenue * 0.05
    mobile_note = "No web CSV provided. Estimated 5% recovery by improving mobile speed and checkout."

leaks.append({
    "name":"Mobile Drop-offs",
    "category":"Website/Checkout",
    "where":"Mobile product pages & checkout",
    "root":"Slow pages or clunky checkout on phones make people quit.",
    "impact":max(0.0, mobile_impact),
    "impact_note":mobile_note,
    "actions":[
        "Compress images; enable lazy loading; aim for < 2.5s on mobile.",
        "Keep checkout to 2 steps or fewer; remove distractions.",
        "Make sure the landing page matches the ad exactly (product, price, promo)."
    ]
})

# ------------------------
# Rank & totals
# ------------------------
money_leaks = [l for l in leaks if l["impact"] > 0]
risk_leaks = [l for l in leaks if l["impact"] == 0]

money_leaks.sort(key=lambda x: x["impact"], reverse=True)

total_recoverable = float(sum(l["impact"] for l in money_leaks))
annual_leak = total_recoverable * 12.0

# ------------------------
# FOMO banner
# ------------------------
st.markdown(f"### You’re likely leaking **{money(total_recoverable)} / month** right now.")
st.markdown(f"That’s **{money(annual_leak)} / year** if nothing changes. Fix the top items this week.")

# ------------------------
# Before vs After (table)
# ------------------------
after_revenue = current_revenue + total_recoverable
delta = after_revenue - current_revenue

bt = pd.DataFrame({
    "Scenario":["Current (est.)","After Fixes (est.)","Gain"],
    "Monthly Revenue":[current_revenue, after_revenue, delta],
    "Annual Revenue":[current_revenue*12, after_revenue*12, delta*12],
}).round(2)

st.subheader("Revenue: Before vs After (Estimates)")
st.table(bt.style.format({"Monthly Revenue":"${:,.0f}","Annual Revenue":"${:,.0f}"}))

# ------------------------
# Leak Cards
# ------------------------
def leak_card(L):
    st.markdown(f"#### {L['name']}")
    st.markdown(f"- **Type:** {L['category']}")
    st.markdown(f"- **Where:** {L['where']}")
    st.markdown(f"- **Root cause:** {L['root']}")
    if L["impact"] > 0:
        st.markdown(f"- **Estimated revenue impact:** {money(L['impact'])} / month")
    if L.get("impact_note"):
        st.markdown(f"- **Note:** {L['impact_note']}")
    st.markdown("**What to do next:**")
    st.markdown("\n".join([f"  1. {a}" if i==0 else f"  {i+1}. {a}" for i,a in enumerate(L["actions"])]))
    st.markdown("---")

st.subheader("Your Biggest Leaks (Money First)")
for L in money_leaks:
    leak_card(L)

if risk_leaks:
    st.subheader("Fix These to Avoid Bad Decisions (No direct $ but critical)")
    for L in risk_leaks:
        leak_card(L)

# ------------------------
# Downloads (in-memory)
# ------------------------
report_md_lines = [
    f"# Revenue Leak Report — {platform}",
    f"**Period:** {period_str}",
    f"**Estimated monthly leak:** {money(total_recoverable)}  |  **Annualized:** {money(annual_leak)}",
    "",
    "## Revenue: Before vs After (Est.)",
    f"- Current monthly revenue: {money(current_revenue)}",
    f"- After fixes (monthly): {money(after_revenue)}",
    f"- Gain (monthly): {money(delta)}",
    "",
    "## Top Leaks",
]
for L in money_leaks + risk_leaks:
    report_md_lines += [
        f"### {L['name']}",
        f"- Type: {L['category']}",
        f"- Where: {L['where']}",
        f"- Root: {L['root']}",
        (f"- Estimated monthly impact: {money(L['impact'])}" if L['impact']>0 else "- Impact: risk / accuracy"),
        (f"- Note: {L['impact_note']}" if L.get('impact_note') else ""),
        "- What to do next:",
    ] + [f"  - {a}" for a in L["actions"]] + [""]

report_md = "\n".join(report_md_lines).encode("utf-8")
st.download_button("⬇️ Download Report (.md)", data=report_md, file_name="revenue_leak_report.md", mime="text/markdown")

html_bytes = f"""
<html><head><meta charset="utf-8"><title>Revenue Leak Report</title></head>
<body style="font-family:Arial,Helvetica,sans-serif;max-width:900px;margin:40px auto;line-height:1.5;">
{report_md.decode('utf-8').replace('\\n','<br/>\\n')}
</body></html>
""".encode("utf-8")
st.download_button("⬇️ Download Report (.html)", data=html_bytes, file_name="revenue_leak_report.html", mime="text/html")

def pdf_from_md(md_text):
    if not REPORTLAB_AVAILABLE:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", fontSize=18, leading=22, spaceAfter=8))
    styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14))
    story = [Paragraph("Revenue Leak Report", styles["H1"]), Spacer(1, 6)]
    for block in md_text.split("\n\n"):
        story.append(Paragraph(block.replace("\n","<br/>"), styles["Body"]))
        story.append(Spacer(1, 3))
    doc.build(story)
    return buf.getvalue()

if REPORTLAB_AVAILABLE:
    pdf_bytes = pdf_from_md(report_md.decode("utf-8"))
    if pdf_bytes:
        st.download_button("⬇️ Download Report (.pdf)", data=pdf_bytes, file_name="revenue_leak_report.pdf", mime="application/pdf")
else:
    st.info("PDF export: install ReportLab with `pip install reportlab` to enable the PDF download.")
