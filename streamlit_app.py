# streamlit_app.py
# AI-Powered Revenue Leak Dashboard (Meta + Google ready)

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- Page ----------
st.set_page_config(
    page_title="AI Revenue Leak Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal modern styling
st.markdown(
    """
    <style>
      .metric-card {background:#0f172a;border-radius:16px;padding:16px;color:#e2e8f0}
      .metric-label {font-size:.9rem;opacity:.8}
      .metric-value {font-size:1.6rem;font-weight:700}
      .good{color:#10b981}.warn{color:#f59e0b}.bad{color:#ef4444}
      .stDataFrame{border-radius:12px;overflow:hidden}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 AI-Powered Revenue Leak Dashboard")
st.caption("Upload a CSV → get investor-ready metrics, beautiful charts, and an actionable leak report.")

# ---------- Helpers ----------
def norm(col: str) -> str:
    return re.sub(r"\s+", "_", col.strip().lower())

def pick(cols, *alts):
    for a in alts:
        if a in cols:
            return a
    return None

def to_numeric(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    if s.dtype.kind in "ifu":
        return pd.to_numeric(s, errors="coerce").fillna(0.0)
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("%", "", regex=False)
         .apply(lambda x: pd.to_numeric(x, errors="coerce"))
         .fillna(0.0)
    )

# ---------- Sidebar ----------
with st.sidebar:
    st.header("1) Upload CSV")
    up = st.file_uploader("Drag & drop your CSV", type=["csv"])
    st.markdown("—")
    st.header("2) Options")
    roas_breakeven = st.number_input(
        "Breakeven ROAS", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
        help="Anything below this is considered wasted spend."
    )
    fatigue_freq = st.number_input("Fatigue frequency threshold", 1.0, 15.0, 5.0, 0.5)
    fatigue_ctr = st.number_input("Fatigue CTR threshold (%)", 0.0, 10.0, 0.5, 0.1)
    st.markdown("—")
    st.caption("Supports Meta & Google exports. Names like amount_spent, results, purchase_roas are auto-mapped.")

if not up:
    st.info("Upload a CSV to begin. You can use the demo CSV or any Meta/Google export.")
    st.stop()

# ---------- Load & Normalize ----------
df_raw = pd.read_csv(up)
df_raw.columns = [norm(c) for c in df_raw.columns]
cols = set(df_raw.columns)

DATE = pick(cols, "date", "reporting_starts", "reporting_ends", "day")
CAMPAIGN = pick(cols, "campaign", "campaign_name")
ADSET = pick(cols, "ad_set", "adset", "adset_name", "ad_set_name")
PLACEMENT = pick(cols, "placement")
SPEND = pick(cols, "spend_usd", "amount_spent_(usd)", "amount_spent", "spend", "amount_spent_(inr)")
CLICKS = pick(cols, "clicks")
IMPS = pick(cols, "impressions")
CTR = pick(cols, "ctr_%", "ctr")
CPC = pick(cols, "cpc_usd", "cpc")
CPM = pick(cols, "cpm_usd", "cpm")
ATC = pick(cols, "add_to_cart", "adds_to_cart")
PUR = pick(cols, "purchases", "results", "conversions")
REV = pick(cols, "conversion_value_usd", "purchase_value", "revenue", "conversion_value")
ROAS = pick(cols, "roas", "purchase_roas_(return_on_ad_spend)")
FREQ = pick(cols, "frequency")

w = pd.DataFrame()
w["date"] = pd.to_datetime(df_raw.get(DATE), errors="coerce")
w["campaign"] = df_raw.get(CAMPAIGN, "-")
w["ad_set"] = df_raw.get(ADSET, "-")
w["placement"] = df_raw.get(PLACEMENT, "-")

w["spend"] = to_numeric(df_raw.get(SPEND))
w["imps"] = to_numeric(df_raw.get(IMPS))
w["clicks"] = to_numeric(df_raw.get(CLICKS))
w["ctr"] = to_numeric(df_raw.get(CTR))
w.loc[w["ctr"].eq(0) & w["imps"].gt(0), "ctr"] = (w["clicks"] / w["imps"]) * 100
w["cpc"] = to_numeric(df_raw.get(CPC))
w.loc[w["cpc"].eq(0) & w["clicks"].gt(0), "cpc"] = w["spend"] / w["clicks"]
w["cpm"] = to_numeric(df_raw.get(CPM))
w.loc[w["cpm"].eq(0) & w["imps"].gt(0), "cpm"] = (w["spend"] / w["imps"]) * 1000
w["atc"] = to_numeric(df_raw.get(ATC))
w["purchases"] = to_numeric(df_raw.get(PUR))
w["revenue"] = to_numeric(df_raw.get(REV))
w["roas"] = to_numeric(df_raw.get(ROAS))
w.loc[w["roas"].eq(0) & w["spend"].gt(0) & w["revenue"].gt(0), "roas"] = w["revenue"] / w["spend"]
w["freq"] = to_numeric(df_raw.get(FREQ))

# Derived
w["aov"] = np.where(w["purchases"].gt(0), w["revenue"]/w["purchases"], 0.0)
w["cvr"] = np.where(w["clicks"].gt(0), w["purchases"]/w["clicks"]*100.0, 0.0)

# ---------- KPIs ----------
T_SPEND = float(w["spend"].sum())
T_REV = float(w["revenue"].sum())
T_PUR = int(w["purchases"].sum())
ACC_ROAS = (T_REV/T_SPEND) if T_SPEND>0 else 0.0

w["waste_low_roas"] = np.where(w["roas"].lt(roas_breakeven), w["spend"], 0.0)
w["waste_fatigue"] = np.where((w["freq"].gt(fatigue_freq)) & (w["ctr"].lt(fatigue_ctr)), w["spend"]*0.5, 0.0)
half_acc = max(roas_breakeven, 0.5*ACC_ROAS) if ACC_ROAS>0 else roas_breakeven
plc = w.groupby("placement", dropna=False).agg(spend=("spend","sum"), roas=("roas","mean")).reset_index()
plc["plc_waste"] = np.where(plc["roas"].lt(half_acc), plc["spend"], 0.0)
w = w.merge(plc[["placement","plc_waste"]], on="placement", how="left").fillna({"plc_waste":0.0})
w["waste"] = w[["waste_low_roas","waste_fatigue","plc_waste"]].sum(axis=1)
WASTE = float(w["waste"].sum())
EFF_NOW = max(T_SPEND - WASTE, 0.0)

c1,c2,c3,c4 = st.columns(4)
for lab, val, css in [
    ("Total Spend", f"${T_SPEND:,.0f}", ""),
    ("Revenue", f"${T_REV:,.0f}", "good" if T_REV>=T_SPEND else "warn"),
    ("ROAS", f"{ACC_ROAS:.2f}", "good" if ACC_ROAS>=roas_breakeven else "bad"),
    ("Est. Wasted", f"${WASTE:,.0f}", "bad" if WASTE>0 else "good"),
]:
    with (c1 if lab=="Total Spend" else c2 if lab=="Revenue" else c3 if lab=="ROAS" else c4):
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>{lab}</div>"
            f"<div class='metric-value {css}'>{val}</div></div>", unsafe_allow_html=True
        )

st.divider()

# ---------- Tabs ----------
TAB1, TAB2, TAB3, TAB4 = st.tabs(["Overview", "Placements", "Campaigns/Ad sets", "Leaks & Fixes"])

# Overview
with TAB1:
    left, right = st.columns([2,1])

    if w["date"].notna().any():
        daily = w.groupby("date").agg(
            spend=("spend","sum"), revenue=("revenue","sum"), roas=("roas","mean")
        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=daily["date"], y=daily["spend"], name="Spend"), secondary_y=False)
        fig.add_trace(go.Bar(x=daily["date"], y=daily["revenue"], name="Revenue"), secondary_y=False)
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["roas"], name="ROAS", mode="lines+markers"), secondary_y=True)
        fig.update_layout(height=380, barmode="group", margin=dict(l=10,r=10,t=10,b=10))
        fig.update_yaxes(title_text="USD", secondary_y=False)
        fig.update_yaxes(title_text="ROAS", secondary_y=True)
        left.plotly_chart(fig, use_container_width=True)
    else:
        left.info("No date column found; showing aggregate visuals only.")

    imps = float(w["imps"].sum()); clicks = float(w["clicks"].sum()); atc = float(w["atc"].sum()); pur = float(w["purchases"].sum())
    funnel = pd.DataFrame({"Stage":["Impressions","Clicks","Add to Cart","Purchases"], "Value":[imps,clicks,atc,pur]})
    figf = px.funnel(funnel, x="Value", y="Stage")
    figf.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10))
    right.plotly_chart(figf, use_container_width=True)

# Placements
with TAB2:
    plc_perf = w.groupby("placement", dropna=False).agg(
        spend=("spend","sum"), revenue=("revenue","sum"), roas=("roas","mean"),
        clicks=("clicks","sum"), imps=("imps","sum"), waste=("waste","sum")
    ).reset_index().sort_values("spend", ascending=False)

    colA, colB = st.columns([2,1])
    fig1 = px.bar(plc_perf, x="placement", y="spend", color="waste",
                  title="Spend by Placement (color = estimated waste)")
    fig1.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
    colA.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(plc_perf, x="roas", y="spend", size="spend", color="placement",
                      title="ROAS vs Spend (size = Spend)")
    fig2.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
    colB.plotly_chart(fig2, use_container_width=True)

    st.dataframe(plc_perf.rename(columns={
        "spend":"Spend","revenue":"Revenue","roas":"ROAS","imps":"Impressions","waste":"Est. Waste"
    }), use_container_width=True)

# Campaigns / Ad sets
with TAB3:
    by_camp = w.groupby("campaign", dropna=False).agg(
        spend=("spend","sum"), revenue=("revenue","sum"), roas=("roas","mean"),
        purchases=("purchases","sum"), waste=("waste","sum")
    ).reset_index()
    figc = px.bar(by_camp.sort_values("spend", ascending=False),
                  x="campaign", y=["spend","revenue"], barmode="group",
                  title="Spend vs Revenue by Campaign")
    figc.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(figc, use_container_width=True)

    if "ad_set" in w.columns and w["ad_set"].nunique()>1:
        by_as = w.groupby(["campaign","ad_set"]).agg(
            spend=("spend","sum"), roas=("roas","mean"),
            purchases=("purchases","sum"), waste=("waste","sum")
        ).reset_index()
        st.dataframe(by_as.sort_values(["campaign","spend"], ascending=[True,False]).rename(columns={
            "spend":"Spend","roas":"ROAS","purchases":"Purchases","waste":"Est. Waste"
        }), use_container_width=True)

# Leaks & Fixes
with TAB4:
    leaks = []
    top_plc = plc_perf.sort_values("waste", ascending=False).head(3)
    for _, r in top_plc.iterrows():
        leaks.append({
            "Leak": f"Weak Placement – {r['placement']}",
            "Estimated Waste $": round(float(r['waste']),2),
            "ROAS": round(float(r['roas']),2),
            "Recommendation": "Exclude/downweight and reallocate to stronger placements",
            "Confidence": "High" if r['roas'] < max(roas_breakeven, 0.5*ACC_ROAS) else "Medium"
        })
    low_as = (
        w[(w["roas"].lt(roas_breakeven)) & (w["spend"].gt(max(100, 0.03*T_SPEND)))]
        .groupby("ad_set", dropna=False)
        .agg(spend=("spend","sum"), roas=("roas","mean"))
        .reset_index().sort_values("spend", ascending=False).head(2)
    )
    for _, r in low_as.iterrows():
        leaks.append({
            "Leak": f"Low ROAS Ad Set – {r['ad_set']}",
            "Estimated Waste $": round(float(r['spend']),2),
            "ROAS": round(float(r['roas']),2),
            "Recommendation": "Pause/reduce budget, refresh creatives, refine audience",
            "Confidence": "High"
        })

    leaks_df = pd.DataFrame(leaks).sort_values("Estimated Waste $", ascending=False)
    st.dataframe(leaks_df, use_container_width=True)

    figf = go.Figure()
    figf.add_bar(name="Effective (Now)", x=["Spend"], y=[max(T_SPEND - WASTE, 0.0)])
    figf.add_bar(name="Effective (After Fixes)", x=["Spend"], y=[T_SPEND])
    figf.update_layout(barmode="group", height=320, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="USD")
    st.plotly_chart(figf, use_container_width=True)

    st.markdown("**Fix Checklist**")
    st.markdown(
        "- Pause weak placements listed above\n"
        "- Kill zero-conversion ads/ad sets\n"
        "- Refresh creatives (shorter videos, stronger hooks)\n"
        "- Consolidate campaigns to exit learning\n"
        "- Set up server-side Conversion API / improve checkout"
    )

st.divider()
st.caption("Export charts via the Plotly toolbar. Tune thresholds from the sidebar.")

