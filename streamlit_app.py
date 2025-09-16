# streamlit_app.py
# ----------------
# Revenue Leak Finder — same content, refreshed UI (Poppins + cards + charts)

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
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

st.set_page_config(page_title="Revenue Leak Finder (Plain, FOMO)", layout="wide")

# ------------------------
# Global styles (Poppins + cards)
# ------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root { --card-bg:#ffffff; --card-border:#e5e7eb; --muted:#6b7280; }
  html, body, [class^="css"]  { font-family: 'Poppins', sans-serif !important; }
  .card { background:var(--card-bg); border:1px solid var(--card-border); border-radius:14px; padding:16px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
  .kpi-title { font-size:12px; color:var(--muted); margin:0; }
  .kpi-value { font-size:28px; font-weight:600; margin:2px 0 0; }
  .kpi-sub { font-size:12px; color:var(--muted); margin-top:2px; }
  .section-title { margin: 8px 0 6px; }
  .table-card .row_heading, .table-card .blank {display:none;}
</style>
""", unsafe_allow_html=True)

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
# Sidebar — Inputs (unchanged)
# ------------------------
st.sidebar.title("Inputs")
platform = st.sidebar.radio("Ad Platform", ["Meta Ads", "Google Ads"], index=0)
ads_file = st.sidebar.file_uploader("Ads CSV (Meta OR Google)", type=["csv","txt"])
backend_file = st.sidebar.file_uploader("Backend Orders CSV (optional)", type=["csv","txt"])
web_file = st.sidebar.file_uploader("Web Analytics CSV (optional)", type=["csv","txt"])

st.sidebar.markdown("---")
st.sidebar.title("Assumptions")
assumed_aov = st.sidebar.number_input("Average Order Value", value=75.0, step=1.0)
uplift_creative = st.sidebar.slider("Gain from refreshing tired ads", 0.0, 0.5, 0.10, 0.01)
uplift_overlap  = st.sidebar.slider("Gain from fixing audience overlap", 0.0, 0.4, 0.06, 0.01)
mobile_cr_target_factor = st.sidebar.slider("Target Mobile CR vs Desktop", 0.1, 1.0, 0.6, 0.05)

st.sidebar.markdown("---")
st.sidebar.title("Thresholds")
ctr_low = st.sidebar.number_input("Low CTR (tired ads) threshold", value=0.008, format="%.4f")
freq_high = st.sidebar.number_input("High Frequency threshold", value=5.0, step=0.5)
min_spend_considered = st.sidebar.number_input("Min Spend per Ad to count", value=50.0, step=10.0)

# ------------------------
# Header (unchanged copy)
# ------------------------
st.title("Wittelsbach AI Ad Revenue Leaks Report")
st.caption("Focus: where money is leaking, how much, and what to do right now.")

if ads_file is None:
    st.info("Upload your **Ads CSV** (Meta OR Google) to get started. Optional: Backend Orders CSV, Web Analytics CSV.")
    st.stop()

# ------------------------
# Load + normalize (unchanged logic)
# ------------------------
ads_df = load_csv(ads_file, parse_dates=["date"])
if ads_df is None or ads_df.empty:
    st.error("Could not read your Ads CSV. Please upload a valid CSV/TXT.")
    st.stop()

backend_df = load_csv(backend_file, parse_dates=["date"]) if backend_file else None
web_df = load_csv(web_file, parse_dates=["date"]) if web_file else None

spend_col = get_first(ads_df, ["spend","cost","amount_spent","ad_cost"])
imp_col   = get_first(ads_df, ["impressions","impr"])
clk_col   = get_first(ads_df, ["clicks","click"])
ctr_c     = get_first(ads_df, ["ctr","click_through_rate"])
freq_c    = get_first(ads_df, ["frequency","avg_frequency"])
purch_c   = get_first(ads_df, ["purchases","conversions","orders","transactions"])
rev_c     = get_first(ads_df, ["revenue","conversion_value","value","purchase_value"])
camp_c    = get_first(ads_df, ["campaign_name","campaign"])
adset_c   = get_first(ads_df, ["adset_name","ad_group","adgroup","ad_group_name"])
ad_c      = get_first(ads_df, ["ad_name","ad","creative"])
obj_c     = get_first(ads_df, ["objective","campaign_objective"])
stage_c   = get_first(ads_df, ["audience_stage","funnel_stage"])
aud_c     = get_first(ads_df, ["audience_name","audience","segment"])

U = pd.DataFrame()
date_c    = get_first(ads_df, ["date","day","day_date"])
U["date"] = ads_df[date_c] if date_c else pd.NaT
U["campaign_name"] = ads_df[camp_c] if camp_c else "Unknown"
U["adset_name"]    = ads_df[adset_c] if adset_c else "Unknown"
U["ad_name"]       = ads_df[ad_c] if ad_c else "Unknown"
U["objective"]     = ads_df[obj_c] if obj_c else ""
U["audience_stage"]= ads_df[stage_c] if stage_c else ""
U["audience_name"] = ads_df[aud_c] if aud_c else ""

U["spend"]        = series_or_default(ads_df, [spend_col], 0.0)
U["impressions"]  = series_or_default(ads_df, [imp_col], 0.0)
U["clicks"]       = series_or_default(ads_df, [clk_col], 0.0)
U["ctr"]          = pd.to_numeric(ads_df[ctr_c], errors="coerce").fillna(0.0) if ctr_c else (U["clicks"]/U["impressions"]).replace([np.inf,-np.inf],0.0).fillna(0.0)
U["frequency"]    = series_or_default(ads_df, [freq_c], 0.0)
U["purchases"]    = series_or_default(ads_df, [purch_c], 0.0)
U["revenue_ads"]  = series_or_default(ads_df, [rev_c], 0.0)

is_meta   = (platform == "Meta Ads")
is_google = (platform == "Google Ads")

# Period + aggregates
period_start = pd.to_datetime(U["date"]).dropna().min() if "date" in U.columns else None
period_end   = pd.to_datetime(U["date"]).dropna().max() if "date" in U.columns else None
period_str   = f"{period_start.date()} to {period_end.date()}" if pd.notna(period_start) and pd.notna(period_end) else "Selected Period"

total_spend      = float(U["spend"].sum())
total_clicks     = int(U["clicks"].sum())
total_impr       = int(U["impressions"].sum())
avg_cpc          = (total_spend / total_clicks) if total_clicks>0 else 0.0
overall_ctr      = (total_clicks / total_impr) if total_impr>0 else 0.0
ads_reported_rev = float(U["revenue_ads"].sum()) or float(U["purchases"].sum()*assumed_aov)

backend_rev = None
if backend_df is not None and not backend_df.empty:
    b_rev_col = get_first(backend_df, ["backend_revenue","revenue","net_revenue"])
    if b_rev_col:
        backend_rev = float(pd.to_numeric(backend_df[b_rev_col], errors="coerce").fillna(0.0).sum())

current_revenue = float(backend_rev if backend_rev and backend_rev>0 else ads_reported_rev)

# ------------------------
# ✨ New: top KPI cards + charts (content unchanged)
# ------------------------
from plotly import express as px

# ---- KPI cards (fixed 4-tuple unpacking) ----
kcol1, kcol2, kcol3, kcol4, kcol5, kcol6 = st.columns(6)

metrics = [
    (kcol1, "Clicks", f"{total_clicks:,}", ""),
    (kcol2, "Impressions", f"{total_impr:,}", ""),
    (kcol3, "Average CPC", money(avg_cpc), ""),
    (kcol4, "CTR", pct(overall_ctr), ""),
    (kcol5, "Conversions/Purchases", f"{int(U['purchases'].sum()):,}", ""),
    (kcol6, "Amount Spent", money(total_spend), ""),
]

for col, title, value, sub in metrics:
    with col:
        st.markdown(f"""
        <div class="card">
          <p class="kpi-title">{title}</p>
          <p class="kpi-value">{value}</p>
          <p class="kpi-sub">{sub}</p>
        </div>
        """, unsafe_allow_html=True)


st.markdown("<div class='section-title'></div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([2,2,2])

# Line: clicks over time
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Clicks over time**")
    if "date" in U.columns and pd.notna(U["date"]).any():
        ts = U.groupby("date", dropna=True)["clicks"].sum().reset_index()
        fig = px.line(ts, x="date", y="clicks", markers=True,
                      labels={"date":"Date","clicks":"Clicks"},
                      title=None)
        fig.update_traces(hovertemplate="Date: %{x}<br>Clicks: %{y:,}")
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), hoverlabel=dict(font_family="Poppins"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.write("No dates in file.")
    st.markdown('</div>', unsafe_allow_html=True)

# Bar: top campaigns by clicks
with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Top campaigns by clicks**")
    if "campaign_name" in U.columns:
        topc = (U.groupby("campaign_name")["clicks"].sum()
                  .sort_values(ascending=False).head(8).reset_index())
        fig2 = px.bar(topc, x="campaign_name", y="clicks",
                      labels={"campaign_name":"Campaign","clicks":"Clicks"},
                      title=None)
        fig2.update_traces(hovertemplate="%{x}<br>Clicks: %{y:,}")
        fig2.update_layout(xaxis_tickangle=-30, margin=dict(l=10,r=10,t=10,b=60),
                           hoverlabel=dict(font_family="Poppins"))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
        st.write("No campaign field in file.")
    st.markdown('</div>', unsafe_allow_html=True)

# Donut: audience/segment share by clicks
with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Audience / Segment share**")
    pie_field = "audience_name" if "audience_name" in U.columns and U["audience_name"].astype(str).str.len().max()>0 \
                else ("adset_name" if "adset_name" in U.columns else None)
    if pie_field:
        pie = (U.groupby(pie_field)["clicks"].sum()
                 .sort_values(ascending=False).head(6).reset_index())
        fig3 = px.pie(pie, names=pie_field, values="clicks", hole=.55)
        fig3.update_traces(textinfo="label+percent", hovertemplate="%{label}<br>Clicks: %{value:,} (%{percent})")
        fig3.update_layout(showlegend=False, margin=dict(l=10,r=10,t=10,b=10),
                           hoverlabel=dict(font_family="Poppins"))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    else:
        st.write("No audience/segment field to chart.")
    st.markdown('</div>', unsafe_allow_html=True)

# Mini table (like screenshot)
st.markdown('<div class="card">**Top rows (sample)**</div>', unsafe_allow_html=True)
sample_cols = [c for c in ["adset_name","campaign_name","clicks","impressions"] if c in U.columns]
if sample_cols:
    st.dataframe(U[sample_cols].head(12), use_container_width=True, height=300)
else:
    st.write("No matching columns to preview.")

# ------------------------
# The rest of your content (unchanged): FOMO banner → B/A table → Leaks → Downloads
# ------------------------

# (1) FOMO banner
# Simple leak calculations reused from your previous version
leaks = []

# Tracking mismatch
if backend_rev and backend_rev>0:
    tracking_diff = ads_reported_rev - backend_rev
    tracking_diff_pct = tracking_diff / backend_rev if backend_rev>0 else 0.0
    leaks.append({
        "name":"Tracking/Reporting Mismatch",
        "category":"Tracking Setup",
        "where":"Pixel & Conversions / Measurement",
        "root":"Ads platform reports more/less sales than your backend (duplicate/missing events or settings).",
        "impact":0.0,
        "impact_note":f"Risk: {money(tracking_diff)} difference ({pct(tracking_diff_pct)}). Fix to avoid bad decisions.",
        "actions":[
            "Turn on server-side Conversions API (Meta) or Enhanced Conversions (Google).",
            "Ensure only one Purchase/Conversion event fires per order.",
            "Use a consistent attribution window (e.g., 7-day click, 1-day view).",
            "Judge real sales by backend, not ad platform alone."
        ]
    })

# Wrong objectives (Meta)
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
    misalign_share = (bofu_wrong / bofu_spend) if bofu_spend>0 else 0.0
    impact_obj = current_revenue * misalign_share * 0.10
    leaks.append({
        "name":"Wrong Goal for Retargeting",
        "category":"Ad Account Setup",
        "where":"Retargeting (bottom of funnel)",
        "root":"Campaigns set to Clicks/Traffic instead of Purchase, so you get clickers not buyers.",
        "impact":max(0.0, impact_obj),
        "impact_note":f"~{pct(misalign_share)} of retargeting spend is on the wrong goal.",
        "actions":[
            "Switch all retargeting to **Purchase/Sales/Conversions** objective.",
            "Separate prospecting (TOFU) from retargeting (BOFU).",
            "Shift budget to campaigns with purchases."
        ]
    })

# Tired creatives
ad_perf = U.groupby("ad_name", dropna=True).agg(spend=("spend","sum"), ctr=("ctr","mean"), freq=("frequency","mean")).reset_index()
ad_perf["is_underperformer"] = (ad_perf["ctr"] < float(ctr_low)) & (ad_perf["freq"] > float(freq_high)) & (ad_perf["spend"] >= float(min_spend_considered))
under_spend = float(ad_perf.loc[ad_perf["is_underperformer"], "spend"].sum())
under_share = (under_spend / total_spend) if total_spend>0 else 0.0
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
        "Launch 3 new variations (UGC video, quick product demo, clear offer).",
        "Rotate creatives every 14 days or when frequency passes the limit."
    ]
})

# Audience overlap
overlap_impact = 0.0
overlap_note = "No audience/segment field found."
if "audience_name" in U.columns and U["audience_name"].astype(str).str.len().max()>0:
    aud_counts = U.groupby(["campaign_name","audience_name"]).size().reset_index(name="count")
    dupe = aud_counts.groupby("audience_name")["campaign_name"].nunique()
    risky_aud = dupe[dupe>1].index.tolist()
    overlap_spend = float(U[U["audience_name"].isin(risky_aud)]["spend"].sum())
    overlap_share = (overlap_spend / total_spend) if total_spend>0 else 0.0
    overlap_impact = current_revenue * overlap_share * float(uplift_overlap)
    overlap_note = f"≈{pct(overlap_share)} of spend hits the same audiences across campaigns."
leaks.append({
    "name":"Paying Twice for the Same People",
    "category":"Audience",
    "where":"Targeting across campaigns",
    "root":"Different campaigns chase the same people at the same time → wasted spend.",
    "impact":max(0.0, overlap_impact),
    "impact_note":overlap_note,
    "actions":[
        "Add exclusions: TOFU excludes MOFU/BOFU.",
        "Seed lookalikes from best buyers (high AOV).",
        "Simplify structures and monitor overlap."
    ]
})

# Mobile drop-offs (web if available; else estimate)
mobile_impact = 0.0
mobile_note = "Estimated potential."
if web_df is not None and not web_df.empty:
    dev_col = get_first(web_df, ["device"])
    ses_col = get_first(web_df, ["sessions"])
    cr_col  = get_first(web_df, ["conversion_rate","purchase_rate"])
    load_c  = get_first(web_df, ["avg_page_load_time"])
    if dev_col and ses_col and cr_col:
        w = web_df.copy()
        w[dev_col] = w[dev_col].astype(str).str.lower()
        mobile_cr = pd.to_numeric(w.loc[w[dev_col]=="mobile", cr_col], errors="coerce").dropna().mean()
        desktop_cr= pd.to_numeric(w.loc[w[dev_col]=="desktop",cr_col], errors="coerce").dropna().mean()
        mobile_s  = pd.to_numeric(w.loc[w[dev_col]=="mobile", ses_col], errors="coerce").dropna().sum()
        target_m  = desktop_cr * float(mobile_cr_target_factor) if desktop_cr else None
        if target_m and mobile_cr and mobile_s and (target_m > mobile_cr):
            delta = max(0.0, target_m - mobile_cr)
            mobile_impact = delta * mobile_s * float(assumed_aov)
        if load_c and load_c in w.columns:
            avg_m = pd.to_numeric(w.loc[w[dev_col]=="mobile", load_c], errors="coerce").dropna().mean()
            if avg_m and avg_m>3.0: mobile_note = f"Mobile pages ~{avg_m:.1f}s. Faster pages convert better."
    else:
        mobile_impact = current_revenue * 0.05
else:
    mobile_impact = current_revenue * 0.05

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
        "Match landing page to ad (product, price, promo)."
    ]
})

# Totals
money_leaks = [l for l in leaks if l["impact"]>0]
risk_leaks  = [l for l in leaks if l["impact"]==0]
money_leaks.sort(key=lambda x: x["impact"], reverse=True)

total_recoverable = float(sum(l["impact"] for l in money_leaks))
annual_leak = total_recoverable * 12.0

st.markdown(f"### You’re likely leaking **{money(total_recoverable)} / month** right now.")
st.markdown(f"That’s **{money(annual_leak)} / year** if nothing changes. Fix the top items this week.")

# Before vs After
after_revenue = current_revenue + total_recoverable
delta = after_revenue - current_revenue
bt = pd.DataFrame({
    "Scenario":["Current (est.)","After Fixes (est.)","Gain"],
    "Monthly Revenue":[current_revenue, after_revenue, delta],
    "Annual Revenue":[current_revenue*12, after_revenue*12, delta*12],
}).round(2)
st.subheader("Revenue: Before vs After (Estimates)")
st.table(bt.style.format({"Monthly Revenue":"${:,.0f}","Annual Revenue":"${:,.0f}"}))

# =========================
# =========================
# Milestones (5 priority leaks) — Redesigned UI (tabs + cards like screenshot)
# =========================

# ---- minimal CSS to mimic the look (Poppins already loaded above) ----
st.markdown("""
<style>
  .pillbar {display:flex; gap:8px; margin:.5rem 0 1rem;}
  .pill {padding:6px 12px; border:1px solid #e5e7eb; border-radius:999px; background:#fff; color:#111827; font-size:13px;}
  .cardx {background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:16px; box-shadow:0 1px 2px rgba(0,0,0,.04);}
  .kpi {display:flex; flex-direction:column; gap:4px;}
  .kpi .label {font-size:12px; color:#6b7280;}
  .kpi .value {font-size:22px; font-weight:600;}
  .badge-up {display:inline-block; padding:2px 8px; background:#ecfdf5; color:#059669; border-radius:999px; font-size:12px; margin-left:6px;}
  .muted {color:#6b7280;}
  .tiny {font-size:12px; color:#6b7280;}
  .list-tight li {margin:2px 0;}
  .tbl .row_heading, .tbl .blank {display:none;}
</style>
""", unsafe_allow_html=True)

st.subheader("Milestone Plan — 5 Priority Leaks")

# 1) Priority order (same names as your leak cards)
priority_order = [
    "Tracking/Reporting Mismatch",     # Phase 1 (risk control – not counted in $ but critical)
    "Wrong Goal for Retargeting",      # Phase 2
    "Tired/Weak Ads",                  # Phase 3
    "Paying Twice for the Same People",# Phase 4
    "Mobile Drop-offs",                # Phase 5
]

# 2) Default durations (days)
default_durations = {
    "Tracking/Reporting Mismatch": 3,
    "Wrong Goal for Retargeting": 1,
    "Tired/Weak Ads": 7,
    "Paying Twice for the Same People": 2,
    "Mobile Drop-offs": 14,
}

# 3) Build rows from your existing `leaks` list
leak_by_name = {L["name"]: L for L in leaks}
rows = []
cursor = pd.Timestamp(date.today())
for idx, name in enumerate(priority_order, start=1):
    L = leak_by_name.get(name, {"category":"—","where":"—","root":"—","impact":0.0,"actions":[]})
    duration = int(default_durations.get(name, 7))
    start_dt = cursor
    end_dt   = cursor + pd.Timedelta(days=duration)
    cursor   = end_dt + pd.Timedelta(days=1)

    monthly = float(max(0.0, L.get("impact", 0.0)))
    annual  = monthly * 12.0
    share   = (monthly / total_recoverable) if total_recoverable > 0 else 0.0

    rows.append({
        "Phase": f"{idx}. {name}",
        "Name": name,
        "Type": L.get("category","—"),
        "Where": L.get("where","—"),
        "Root cause": L.get("root","—"),
        "Monthly Recovery": monthly,
        "Annual Recovery": annual,
        "% of Total": share,
        "Duration (days)": duration,
        "Start": start_dt.date(),
        "Due": end_dt.date(),
        "Actions": L.get("actions", []),
        "Note": L.get("impact_note",""),
    })

milestones_df = pd.DataFrame(rows)

# ---- OVERVIEW header pills (purely visual, tabs are below) ----
# --- Pill-styled real tabs (replace the pillbar you removed) ---
st.markdown("""
<style>
div[data-baseweb="tab-list"] { gap: 8px; }
button[role="tab"] {
  background:#fff; border:1px solid #e5e7eb; border-radius:999px;
  padding:6px 12px; font-size:13px; color:#111827; box-shadow:0 1px 2px rgba(0,0,0,.04);
}
button[aria-selected="true"] {
  background:#eef2ff; border-color:#6366f1; color:#3730a3; font-weight:600;
}
</style>
""", unsafe_allow_html=True)

tab_labels = ["Overview"] + [r["Phase"] for r in rows]
tabs = st.tabs(tab_labels)

# OVERVIEW
with tabs[0]:
    show_cols = ["Phase","Type","Where","Monthly Recovery","Duration (days)","Start","Due"]
    st.table(
        milestones_df[show_cols].style.format({
            "Monthly Recovery": "${:,.0f}",
            "Duration (days)": "{:,.0f}"
        })
    )

# PHASE TABS
for i, r in enumerate(rows, start=1):
    with tabs[i]:
        pcol1, pcol2, pcol3 = st.columns([2,1,1])
        with pcol1:
            st.write(f"**Leak:** {r['Name']}  \n*{r['Type']} • {r['Where']}*")
        with pcol2:
            st.write(f"**Monthly recovery:** ${r['Monthly Recovery']:,.0f}  \n_{r['% of Total']*100:,.0f}% of total_")
        with pcol3:
            st.write(f"**Duration:** {int(r['Duration (days)'])} days  \n_{r['Start']} → {r['Due']}_")
        st.write("**Root cause**")
        st.write(r["Root cause"])
        st.write("**What to do this phase**")
        if r["Actions"]:
            st.markdown("\n".join([f"- {a}" for a in r["Actions"]]))
        if r["Note"]:
            st.caption(r["Note"])


# ---- Overview KPI cards ----
t1, t2, t3 = st.columns(3)
total_monthly = float(milestones_df["Monthly Recovery"].sum())
total_annual  = float(milestones_df["Annual Recovery"].sum())
total_days    = int(milestones_df["Duration (days)"].sum())

with t1:
    st.markdown(f'<div class="cardx kpi"><span class="label">Estimated recovery this month</span>'
                f'<span class="value">${total_monthly:,.0f}</span>'
                f'</div>', unsafe_allow_html=True)
with t2:
    st.markdown(f'<div class="cardx kpi"><span class="label">Annualized recovery</span>'
                f'<span class="value">${total_annual:,.0f}</span>'
                f'</div>', unsafe_allow_html=True)
with t3:
    st.markdown(f'<div class="cardx kpi"><span class="label">Total execution window</span>'
                f'<span class="value">{total_days} days</span>'
                f'<span class="tiny muted">Phases run back-to-back from today</span>'
                f'</div>', unsafe_allow_html=True)

st.markdown("")

# ---- Tabs: Overview + one tab per phase ----
tab_labels = ["Overview"] + [r["Phase"] for r in rows]
tabs = st.tabs(tab_labels)

# OVERVIEW TAB (table like screenshot)
with tabs[0]:
    st.markdown("**Roadmap this week**")
    show_cols = ["Phase","Type","Where","Monthly Recovery","Duration (days)","Start","Due"]
    st.markdown('<div class="cardx">', unsafe_allow_html=True)
    st.table(
        milestones_df[show_cols].style.format({
            "Monthly Recovery": "${:,.0f}",
            "Duration (days)": "{:,.0f}"
        }).set_table_attributes('class="tbl"')
    )
    st.markdown('</div>', unsafe_allow_html=True)

# PHASE TABS
for i, r in enumerate(rows, start=1):
    with tabs[i]:
        pcol1, pcol2, pcol3 = st.columns([2,1,1])
        with pcol1:
            st.markdown(f'<div class="cardx"><div class="kpi">'
                        f'<span class="label">Leak</span>'
                        f'<span class="value">{r["Name"]}</span>'
                        f'<span class="tiny muted">{r["Type"]} • {r["Where"]}</span>'
                        f'</div></div>', unsafe_allow_html=True)
        with pcol2:
            st.markdown(f'<div class="cardx kpi"><span class="label">Monthly recovery</span>'
                        f'<span class="value">${r["Monthly Recovery"]:,.0f}'
                        f'<span class="badge-up">{r["% of Total"]*100:,.0f}% of total</span></span>'
                        f'</div>', unsafe_allow_html=True)
        with pcol3:
            st.markdown(f'<div class="cardx kpi"><span class="label">Duration</span>'
                        f'<span class="value">{int(r["Duration (days)"])} days</span>'
                        f'<span class="tiny muted">{r["Start"]} → {r["Due"]}</span>'
                        f'</div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown("**Root cause**")
        st.markdown(f'<div class="cardx">{r["Root cause"]}</div>', unsafe_allow_html=True)

        st.markdown("**What to do this phase**")
        st.markdown('<div class="cardx">', unsafe_allow_html=True)
        if r["Actions"]:
            st.markdown("\n".join([f"- {a}" for a in r["Actions"]]), unsafe_allow_html=True)
        else:
            st.markdown("- Actions will appear here when available.")
        if r["Note"]:
            st.caption(r["Note"])
        st.markdown('</div>', unsafe_allow_html=True)

# ---- Download milestones as Markdown (unchanged logic) ----
milestones_md = ["# Milestones — 5 Priority Leaks\n"]
for r in rows:
    milestones_md += [
        f"## {r['Phase']}",
        f"- Type: {r['Type']}",
        f"- Where: {r['Where']}",
        f"- Root cause: {r['Root cause']}",
        f"- Est. monthly recovery: ${r['Monthly Recovery']:,.0f}",
        f"- Est. annual recovery: ${r['Annual Recovery']:,.0f}",
        f"- Duration: {int(r['Duration (days)'])} days",
        f"- Start → Due: {r['Start']} → {r['Due']}",
        ""
    ]
st.download_button(
    "⬇️ Download Milestones (.md)",
    data="\n".join(milestones_md).encode("utf-8"),
    file_name="milestones_5_priority_leaks.md",
    mime="text/markdown"
)


# Leak cards
def leak_card(L):
    st.markdown(f"#### {L['name']}")
    st.markdown(f"- **Type:** {L['category']}")
    st.markdown(f"- **Where:** {L['where']}")
    st.markdown(f"- **Root cause:** {L['root']}")
    if L["impact"]>0:
        st.markdown(f"- **Estimated revenue impact:** {money(L['impact'])} / month")
    if L.get("impact_note"):
        st.markdown(f"- **Note:** {L['impact_note']}")
    st.markdown("**What to do next:**")
    st.markdown("\n".join([f"  1. {a}" if i==0 else f"  {i+1}. {a}" for i,a in enumerate(L["actions"])]))
    st.markdown("---")

st.subheader("Your Biggest Leaks (Money First)")
for L in money_leaks: leak_card(L)
if risk_leaks:
    st.subheader("Fix These to Avoid Bad Decisions (No direct $ but critical)")
    for L in risk_leaks: leak_card(L)

# Downloads
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
<body style="font-family:Poppins,Arial,Helvetica,sans-serif;max-width:900px;margin:40px auto;line-height:1.5;">
{report_md.decode('utf-8').replace('\\n','<br/>\\n')}
</body></html>
""".encode("utf-8")
st.download_button("⬇️ Download Report (.html)", data=html_bytes, file_name="revenue_leak_report.html", mime="text/html")

def pdf_from_md(md_text):
    if not REPORTLAB_AVAILABLE: return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", fontSize=18, leading=22, spaceAfter=8))
    styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14))
    story = [Paragraph("Revenue Leak Report", styles["H1"]), Spacer(1, 6)]
    for block in md_text.split("\n\n"):
        story.append(Paragraph(block.replace("\n","<br/>"), styles["Body"]))
        story.append(Spacer(1, 3))
    doc.build(story); return buf.getvalue()

if REPORTLAB_AVAILABLE:
    pdf_bytes = pdf_from_md(report_md.decode("utf-8"))
    if pdf_bytes:
        st.download_button("⬇️ Download Report (.pdf)", data=pdf_bytes, file_name="revenue_leak_report.pdf", mime="application/pdf")
