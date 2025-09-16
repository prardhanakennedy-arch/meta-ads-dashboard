# Create a simplified, plain-language, no-graphs Streamlit app focused on clear leak detection.
# It removes visual charts and conveys insights with short explanations, definitions, and step-by-step guidance.
# File: /mnt/data/meta_ads_revenue_leak_report_plain.py

code = r'''
# meta_ads_revenue_leak_report_plain.py
# ------------------------------------
# A neat, tidy, plain-language dashboard to spot Meta Ads revenue leaks.
# - No graphs. Clean sections with concise explanations.
# - Defines technical terms in simple language.
# - Mirrors the "Leak #1" clarity for all features: Issue, What this means, Why it matters, How we estimated, What to do next.
#
# How to run:
#   pip install streamlit pandas numpy python-dateutil
#   streamlit run meta_ads_revenue_leak_report_plain.py

import io
from datetime import datetime, date
from dateutil import parser
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Meta Ads Revenue Leak Finder (Plain Language)", layout="wide")

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
# Sidebar
# ------------------------
st.sidebar.title("Inputs")
meta_file = st.sidebar.file_uploader("Meta Ads (CSV/XLSX)", type=["csv","xlsx"])
backend_file = st.sidebar.file_uploader("Backend Orders (CSV/XLSX)", type=["csv","xlsx"])
web_file = st.sidebar.file_uploader("Web Analytics (CSV/XLSX)", type=["csv","xlsx"])
pixel_file = st.sidebar.file_uploader("Pixel Events (optional)", type=["csv","xlsx"])

st.sidebar.markdown("---")
st.sidebar.title("Report Details")
client_name = st.sidebar.text_input("Client/Brand Name", "Acme Co.")
prepared_by = st.sidebar.text_input("Prepared By", "Your Agency Name")
report_date = st.sidebar.date_input("Report Date", value=date.today())

st.sidebar.markdown("---")
st.sidebar.title("Assumptions")
assumed_aov = st.sidebar.number_input("Average Order Value (AOV)", value=75.0, step=1.0)
mobile_cr_target_factor = st.sidebar.slider("Mobile CR Target vs Desktop", 0.1, 1.0, 0.6, 0.05)

st.sidebar.markdown("---")
st.sidebar.title("Thresholds")
ctr_low = st.sidebar.number_input("Low CTR threshold", value=0.007, format="%.4f")
freq_high = st.sidebar.number_input("High Frequency threshold", value=5.0, step=0.5)
min_spend_considered = st.sidebar.number_input("Min Spend to Consider (per ad)", value=50.0, step=10.0)

# ------------------------
# Load & sanitize
# ------------------------
if meta_file is None:
    st.title("Meta Ads Revenue Leak Finder — Plain, No Graphs")
    st.write("Upload your CSVs in the left sidebar. This dashboard speaks plain language and focuses on the 5 biggest leak areas.")
    st.stop()

meta_df = load_csv(meta_file, parse_dates=["date"])
backend_df = load_csv(backend_file, parse_dates=["date"])
web_df = load_csv(web_file, parse_dates=["date"])
pixel_df = load_csv(pixel_file, parse_dates=["date"])

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

if backend_df is not None:
    for col in ["date","orders","backend_revenue"]:
        if col not in backend_df.columns:
            backend_df[col] = np.nan
    backend_df["date"] = backend_df["date"].apply(_to_date)
    backend_df["orders"] = try_col(backend_df, "orders", 0.0)
    backend_df["backend_revenue"] = try_col(backend_df, "backend_revenue", 0.0)
else:
    tmp = meta_df.groupby("date", dropna=True).agg({"purchases":"sum"}).reset_index()
    if len(tmp) == 0:
        tmp = pd.DataFrame({"date":[date.today()], "purchases":[0.0]})
    tmp["orders"] = tmp["purchases"]
    tmp["backend_revenue"] = tmp["orders"] * float(assumed_aov)
    backend_df = tmp[["date","orders","backend_revenue"]].copy()

if web_df is None:
    web_df = pd.DataFrame(columns=["date","device","sessions","conversion_rate","avg_page_load_time"])

# ------------------------
# Period & core aggregates
# ------------------------
period_start, period_end = infer_period(meta_df, backend_df, web_df)
period_str = f"{period_start} to {period_end}" if (period_start and period_end) else "Selected Period"

meta_rev = float(meta_df["revenue"].sum())
backend_rev = float(backend_df["backend_revenue"].sum())
tracking_diff = meta_rev - backend_rev
tracking_diff_pct = (tracking_diff/backend_rev) if backend_rev > 0 else 0.0

# Duplicate purchase detection (heuristic)
dup_ratio_text = "not enough data"
if pixel_df is not None and len(pixel_df) > 0:
    purchases_events = pixel_df[pixel_df["event_name"].fillna("").str.lower().str.contains("purchase")]
    if "event_id" in purchases_events.columns and purchases_events["event_id"].notna().any():
        total_purchase_events = len(purchases_events)
        unique_purchase_events = purchases_events["event_id"].nunique()
        if unique_purchase_events > 0:
            dup_ratio = (total_purchase_events - unique_purchase_events) / unique_purchase_events
            dup_ratio_text = f"{pct(dup_ratio)} estimated duplicate purchase events"
        else:
            dup_ratio_text = "no unique purchase IDs to compare"

# Objective misalignment
meta_df["is_bofu_like"] = False
mask_rt = meta_df["campaign_name"].str.lower().str.contains("retarget|remarket|cart|viewcontent|view content|dpa|catalog", regex=True)
meta_df.loc[mask_rt, "is_bofu_like"] = True
if "audience_stage" in meta_df.columns:
    meta_df.loc[meta_df["audience_stage"].str.upper().isin(["MOFU","BOFU"]), "is_bofu_like"] = True

meta_df["objective_clean"] = meta_df["objective"].str.lower()
is_purchase_objective = meta_df["objective_clean"].str.contains("purchase|sales|conversions", regex=True)

bofu_spend = float(meta_df.loc[meta_df["is_bofu_like"], "spend"].sum())
bofu_spend_wrong_obj = float(meta_df.loc[meta_df["is_bofu_like"] & (~is_purchase_objective), "spend"].sum())
obj_misalignment_spend_pct = (bofu_spend_wrong_obj / bofu_spend) if bofu_spend > 0 else 0.0

# Creative fatigue
ad_perf = meta_df.groupby("ad_name", dropna=True).agg(
    spend=("spend","sum"),
    ctr=("ctr","mean"),
    freq=("frequency","mean")
).reset_index()
ad_perf["is_underperformer"] = (ad_perf["ctr"] < float(ctr_low)) & (ad_perf["freq"] > float(freq_high)) & (ad_perf["spend"] >= float(min_spend_considered))
underperf_spend = float(ad_perf.loc[ad_perf["is_underperformer"], "spend"].sum())
total_spend = float(ad_perf["spend"].sum()) if len(ad_perf) else 0.0
underperf_spend_pct = (underperf_spend / total_spend) if total_spend > 0 else 0.0

# Audience overlap (heuristic)
if "audience_name" in meta_df.columns:
    aud_counts = meta_df.groupby(["campaign_name","audience_name"]).size().reset_index(name="count")
    duplicated_audiences = aud_counts.groupby("audience_name")["campaign_name"].nunique()
    overlap_risky_aud = duplicated_audiences[duplicated_audiences > 1].index.tolist()
    overlap_spend = float(meta_df[meta_df["audience_name"].isin(overlap_risky_aud)]["spend"].sum())
else:
    overlap_spend = 0.0
overlap_spend_pct = (overlap_spend / total_spend) if total_spend > 0 else 0.0

# Landing page & checkout drop-off
mobile_cr = desktop_cr = None
mobile_sessions = desktop_sessions = 0.0
if len(web_df) > 0 and "device" in web_df.columns:
    if "conversion_rate" in web_df.columns:
        mobile_cr = web_df.loc[web_df["device"].str.lower()=="mobile", "conversion_rate"].astype(float).dropna().mean()
        desktop_cr = web_df.loc[web_df["device"].str.lower()=="desktop", "conversion_rate"].astype(float).dropna().mean()
    if "sessions" in web_df.columns:
        mobile_sessions = float(web_df.loc[web_df["device"].str.lower()=="mobile", "sessions"].astype(float).sum())
        desktop_sessions = float(web_df.loc[web_df["device"].str.lower()=="desktop", "sessions"].astype(float).sum())
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
        total_sessions = float(web_df["sessions"].astype(float).sum())
        if total_sessions > 0:
            mobile_share = mobile_sessions / total_sessions if total_sessions else 0.0
            mobile_loss = backend_rev * mobile_share * 0.1

# ------------------------
# UI: Header
# ------------------------
st.title("Meta Ads Revenue Leak Finder — Plain Language")
st.caption(f"Client: {client_name}  |  Period: {period_str}  |  Prepared by: {prepared_by}  |  Date: {report_date}")

# KPIs (plain numbers; no charts)
c1, c2, c3 = st.columns(3)
c1.metric("Meta-reported Revenue", money(meta_rev))
c2.metric("Backend Revenue (actual)", money(backend_rev))
c3.metric("Difference (Meta - Backend)", money(tracking_diff))

st.markdown("---")

# ------------------------
# Glossary (simple language)
# ------------------------
with st.expander("Simple Definitions (click to open)"):
    st.markdown("""
- **Conversion Rate (CR):** Out of 100 visitors, how many buy.
- **CTR (Click-Through Rate):** Out of 100 people who saw the ad, how many clicked.
- **Frequency:** On average, how many times each person saw your ad.
- **BOFU / Retargeting:** Ads shown to people who already visited or added to cart — they're close to buying.
- **Objective (Purchase vs others):** What Meta is trying to get for you. If you choose **Purchase**, Meta finds buyers. If you choose **Clicks**, Meta finds clickers (not always buyers).
- **Audience Overlap:** When multiple ad sets target the same people; you pay to show ads to the same person more than once across different ad sets.
- **AOV (Average Order Value):** Average money made per order.
    """)

# ------------------------
# Leak Sections (no graphs, plain language)
# ------------------------

def leak_section(title, issue, means, matters, estimate, actions, note=None):
    st.subheader(title)
    st.write(f"**Issue (what we see):** {issue}")
    st.write(f"**What this means (simple):** {means}")
    st.write(f"**Why this matters (money/accuracy):** {matters}")
    if estimate:
        st.write(f"**Estimated impact:** {estimate}")
    if actions:
        st.write("**What to do next (step-by-step):**")
        st.write(actions)
    if note:
        with st.expander("How we estimated this (details)"):
            st.write(note)
    st.markdown("---")

# Leak 1: Tracking
dup_line = f" Also, {dup_ratio_text}." if isinstance(dup_ratio_text, str) else ""
leak_section(
    "Leak #1 — Tracking & Attribution",
    issue=f"Meta says {money(meta_rev)}; your backend says {money(backend_rev)}. The difference is {money(tracking_diff)} ({pct(tracking_diff_pct)}).{dup_line}",
    means="Your reports may be inflating or missing sales. That makes decisions (like where to spend more) unreliable.",
    matters="If the numbers are off, you might scale the wrong campaigns or cut the right ones.",
    estimate="We treat this as a **risk** (not direct revenue). Fixing it prevents costly mistakes.",
    actions="""
1. Turn on **server-side Conversions API** (CAPI).
2. Make sure **only one** Purchase event fires per order.
3. Use a consistent attribution window (e.g., **7-day click, 1-day view**).
4. Pick a **single source of truth** (usually backend) to judge real sales.
""",
    note="Difference = Meta revenue minus backend revenue. Duplicate purchase events estimated when the same purchase ID appears multiple times."
)

# Leak 2: Objectives
leak_section(
    "Leak #2 — Wrong Objectives in Retargeting",
    issue=f"About {pct(obj_misalignment_spend_pct)} of retargeting/BOFU spend is not set to **Purchase** optimization.",
    means="Meta may be finding people who click or add-to-cart instead of actual buyers.",
    matters="Money goes to actions that don't reliably produce sales.",
    estimate=f"Recoverable revenue depends on your mix, but we expect a lift by shifting this spend to **Purchase**. A simple target is to move {pct(obj_misalignment_spend_pct)} of BOFU spend into Purchase optimization.",
    actions="""
1. Change all BOFU/retargeting campaigns to **Purchase** (or Sales/Conversions).
2. Keep TOFU (prospecting) separate from MOFU/BOFU in your account.
3. Move budget toward campaigns with proven purchases.
""",
    note="We flagged BOFU via audience_stage where available and retargeting keywords (retarget, DPA, catalog, etc.). We then checked if objective contained 'purchase/sales/conversions'."
)

# Leak 3: Creative fatigue
leak_section(
    "Leak #3 — Tired Creatives (Low CTR + High Frequency)",
    issue=f"{pct(underperf_spend_pct)} of spend went to ads with low CTR (below {ctr_low:.2%}) and high Frequency (above {freq_high}), with enough spend to matter.",
    means="People have seen the same ad too often and stopped reacting, or the ad isn't compelling.",
    matters="You pay for impressions that don't move people to buy, dragging down results.",
    estimate="A practical goal is to pause these ads and replace them. Expect CTR to lift and costs to stabilize when you refresh regularly.",
    actions=f"""
1. **Pause** identified underperformers (look for CTR < {ctr_low:.2%} and Frequency > {freq_high}).
2. Launch **3 fresh formats**: a customer-style video (UGC), a quick product demo, and a clear offer.
3. Set a simple rule: rotate creatives every **14 days** or when Frequency passes **{freq_high}**.
"""
)

# Leak 4: Audience overlap
leak_section(
    "Leak #4 — Audience Overlap (Paying Twice for the Same People)",
    issue=f"About {pct(overlap_spend_pct)} of spend is at risk from duplicated audiences across campaigns/ad sets.",
    means="Different ad sets are targeting the same people at the same time.",
    matters="You pay more to talk to the same person, instead of reaching new potential buyers.",
    estimate="By adding exclusions, you reduce waste and redirect that budget to fresh audiences or best performers.",
    actions="""
1. Add **mutual exclusions** between TOFU, MOFU, BOFU.
2. Build **lookalikes** from your best buyers (high AOV) for prospecting.
3. Consider **Advantage+ Shopping** for a simplified structure (then monitor overlap).
""",
    note="We looked for the same audience names used in multiple campaigns, which signals possible overlap."
)

# Leak 5: Mobile checkout
lp_note = ""
if (mobile_cr is not None) and (desktop_cr is not None):
    lp_note = f"Mobile CR is {pct(mobile_cr)} vs Desktop CR {pct(desktop_cr)}."
if avg_page_load_mobile is not None:
    lp_note += f" Average mobile page load around {avg_page_load_mobile:.2f}s."

leak_section(
    "Leak #5 — Mobile Drop-offs (Slow Pages or Clunky Checkout)",
    issue=lp_note if lp_note else "Mobile conversion appears lower than desktop and/or mobile pages may be slow.",
    means="Even with good ads, people leave if pages load slowly or the checkout is hard to finish on a phone.",
    matters="You lose buyers where most traffic usually is: on mobile.",
    estimate=f"Aim for mobile CR to be at least **{int(mobile_cr_target_factor*100)}%** of desktop CR. Closing that gap can add {money(mobile_loss)} per period (rough estimate).",
    actions="""
1. Compress images and use **lazy loading** to speed up pages (target under 2.5s).
2. Keep checkout to **2 steps or fewer** on mobile.
3. Match ad message and landing page exactly (price, promo, product).
"""
)

# ------------------------
# Final call-outs & download
# ------------------------
st.subheader("Next Steps (Short and Clear)")
st.markdown(f"""
1. Fix tracking first so decisions are based on reality.  
2. Switch retargeting objectives to **Purchase**.  
3. Pause tired creatives; publish 3 fresh variations; rotate every 14 days.  
4. Add audience **exclusions** between funnel stages.  
5. Speed up mobile pages and simplify checkout.
""")

# Build a simple text report for download
report_text = f"""
Meta Ads Revenue Leak Report — Plain Language
Client: {client_name} | Period: {period_str} | Prepared by: {prepared_by} | Date: {report_date}

KPIs:
- Meta revenue: {money(meta_rev)}
- Backend revenue: {money(backend_rev)}
- Difference: {money(tracking_diff)}

Leak 1 — Tracking & Attribution
- Issue: Meta {money(meta_rev)} vs Backend {money(backend_rev)} (Diff {money(tracking_diff)} / {pct(tracking_diff_pct)})
- Meaning: Reporting may be wrong
- Why it matters: Bad decisions
- Action: Enable CAPI, dedupe purchase events, standardize attribution

Leak 2 — Wrong Objectives in Retargeting
- Issue: {pct(obj_misalignment_spend_pct)} of BOFU spend not optimized for Purchase
- Meaning: Optimizing for actions that don't equal sales
- Action: Switch BOFU to Purchase, separate funnel stages, move budget to proven

Leak 3 — Tired Creatives
- Issue: {pct(underperf_spend_pct)} of spend on low CTR/high Frequency ads
- Action: Pause, launch 3 new formats, rotate every 14 days

Leak 4 — Audience Overlap
- Issue: {pct(overlap_spend_pct)} spend at risk from duplicated audiences
- Action: Add exclusions, seed lookalikes from best buyers, consider Advantage+

Leak 5 — Mobile Drop-offs
- Issue: {('Mobile vs Desktop CR gap; ' + lp_note) if lp_note else 'Mobile CR lower and/or slow pages'}
- Estimate: Potential gain ~ {money(mobile_loss)}
- Action: Speed up pages (<2.5s), shorten checkout, match ad to landing page

Next Steps:
1) Fix tracking  2) Switch objectives  3) Refresh creatives  4) Add exclusions  5) Mobile UX
"""

st.download_button(
    "⬇️ Download Plain Report (.txt)",
    data=report_text.encode("utf-8"),
    file_name=f"{client_name.replace(' ','_').lower()}_meta_ads_leak_report_plain.txt",
    mime="text/plain"
)
'''
with open('/mnt/data/meta_ads_revenue_leak_report_plain.py', 'w', encoding='utf-8') as f:
    f.write(code)

'/mnt/data/meta_ads_revenue_leak_report_plain.py'
