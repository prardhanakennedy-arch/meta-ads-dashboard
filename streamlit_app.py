# streamlit_app.py
# Requirements:
# streamlit
# pandas
# numpy
# plotly

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------- Page ----------
st.set_page_config(page_title="AI Revenue Leak Report", page_icon="📊", layout="wide")
st.title("📊 AI-Powered Revenue Leak Report")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Upload CSV")
    up = st.file_uploader("Drag & drop Meta/Google CSV", type=["csv"])
    st.caption("Works with Meta & Google Ads CSV exports. USD only.")
    st.divider()
    st.header("Leak Rules")
    breakeven_roas = st.number_input("Breakeven ROAS", 0.1, 10.0, 1.0, 0.1)
    fatigue_freq = st.number_input("Fatigue if Frequency >", 1.0, 15.0, 5.0, 0.5)
    fatigue_ctr = st.number_input("Fatigue if CTR < (%)", 0.0, 10.0, 0.5, 0.1)

if not up:
    st.info("Upload a CSV to generate the dashboard.")
    st.stop()

# ---------- Helpers ----------
def norm(col: str) -> str:
    return re.sub(r"\s+", "_", str(col).strip().lower())

def pick(cols, *alts):
    for a in alts:
        if a in cols:
            return a
    return None

def to_num(s: pd.Series) -> pd.Series:
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

# ---------- Load & normalize ----------
raw = pd.read_csv(up)
raw.columns = [norm(c) for c in raw.columns]
cols = set(raw.columns)

DATE = pick(cols, "date", "day", "reporting_starts")
CAMPAIGN = pick(cols, "campaign", "campaign_name")
ADSET = pick(cols, "ad_set", "adset", "adset_name")
PLACEMENT = pick(cols, "placement", "platform_position")
SPEND = pick(cols, "amount_spent_(usd)", "spend_usd", "amount_spent", "spend")
IMPS = pick(cols, "impressions")
CLICKS = pick(cols, "clicks")
CTR = pick(cols, "ctr_%", "ctr")
CPC = pick(cols, "cpc_usd", "cpc")
CPM = pick(cols, "cpm_usd", "cpm")
ATC = pick(cols, "add_to_cart", "adds_to_cart")
PUR = pick(cols, "purchases", "results", "conversions")
REV = pick(cols, "conversion_value_(usd)", "conversion_value_usd", "conversion_value", "purchase_value", "revenue")
ROAS = pick(cols, "purchase_roas", "purchase_roas_(return_on_ad_spend)", "roas")
FREQ = pick(cols, "frequency")

w = pd.DataFrame()
w["date"] = pd.to_datetime(raw.get(DATE), errors="coerce")
w["campaign"] = raw.get(CAMPAIGN, "-")
w["ad_set"] = raw.get(ADSET, "-")
w["placement"] = raw.get(PLACEMENT, "-")

w["spend"] = to_num(raw.get(SPEND))
w["imps"] = to_num(raw.get(IMPS))
w["clicks"] = to_num(raw.get(CLICKS))
w["ctr"] = to_num(raw.get(CTR))
w.loc[w["ctr"].eq(0) & w["imps"].gt(0), "ctr"] = (w["clicks"] / w["imps"]) * 100
w["cpc"] = to_num(raw.get(CPC))
w.loc[w["cpc"].eq(0) & w["clicks"].gt(0), "cpc"] = w["spend"] / w["clicks"]
w["cpm"] = to_num(raw.get(CPM))
w.loc[w["cpm"].eq(0) & w["imps"].gt(0), "cpm"] = (w["spend"] / w["imps"]) * 1000
w["atc"] = to_num(raw.get(ATC))
w["purchases"] = to_num(raw.get(PUR))
w["revenue"] = to_num(raw.get(REV))
w["roas"] = to_num(raw.get(ROAS))
w.loc[w["roas"].eq(0) & w["spend"].gt(0) & w["revenue"].gt(0), "roas"] = w["revenue"] / w["spend"]
w["freq"] = to_num(raw.get(FREQ))
w["cvr"] = np.where(w["clicks"].gt(0), w["purchases"]/w["clicks"]*100.0, 0.0)
w["aov"] = np.where(w["purchases"].gt(0), w["revenue"]/w["purchases"], 0.0)

# ---------- KPIs ----------
TOTAL_SPEND = float(w["spend"].sum())
TOTAL_REV = float(w["revenue"].sum())
AVG_ROAS = (TOTAL_REV / TOTAL_SPEND) if TOTAL_SPEND > 0 else 0.0

# Leak estimation
w["waste_low_roas"] = np.where(w["roas"] < breakeven_roas, w["spend"], 0.0)
w["waste_fatigue"] = np.where((w["freq"] > fatigue_freq) & (w["ctr"] < fatigue_ctr), w["spend"] * 0.5, 0.0)

half_acc = max(breakeven_roas, 0.5 * AVG_ROAS) if AVG_ROAS > 0 else breakeven_roas
plc = w.groupby("placement", dropna=False).agg(spend=("spend","sum"), roas=("roas","mean")).reset_index()
plc["plc_waste"] = np.where(plc["roas"] < half_acc, plc["spend"], 0.0)
w = w.merge(plc[["placement", "plc_waste"]], on="placement", how="left").fillna({"plc_waste": 0.0})
w["waste"] = w[["waste_low_roas", "waste_fatigue", "plc_waste"]].sum(axis=1)

WASTED = float(w["waste"].sum())
EFFECTIVE_NOW = max(TOTAL_SPEND - WASTED, 0.0)
DAILY_WASTE = WASTED / 30.0 if WASTED > 0 else 0.0

# ---------- Section A — Executive Summary ----------
st.markdown(f"### 🚨 You're leaking **~${WASTED:,.0f}** this month")
st.markdown("A leak is ad spend that doesn’t bring sales. Plug leaks, move budget into what works.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Spend", f"${TOTAL_SPEND:,.0f}")
c2.metric("Wasted Spend", f"${WASTED:,.0f}")
c3.metric("Avg ROAS", f"{AVG_ROAS:.2f}")
c4.metric("Savings Potential", f"${WASTED:,.0f}")

st.markdown(f"**Daily leak:** ${DAILY_WASTE:,.0f} • **Weekly:** ${DAILY_WASTE*7:,.0f} • **Monthly:** ${WASTED:,.0f}")

donut = go.Figure(data=[go.Pie(labels=["Effective Spend", "Wasted Spend"],
                               values=[EFFECTIVE_NOW, WASTED], hole=0.6)])
donut.update_layout(height=280, template="plotly_white", showlegend=False)
st.plotly_chart(donut, use_container_width=True)

st.divider()

# ---------- Section B — Account at a Glance ----------
st.subheader("📈 Account at a Glance")
if w["date"].notna().any():
    daily = w.groupby("date").agg(spend=("spend","sum"), revenue=("revenue","sum")).reset_index()
    fig_grad = go.Figure()
    fig_grad.add_trace(go.Scatter(x=daily["date"], y=daily["spend"], mode="lines", name="Spend"))
    fig_grad.add_trace(go.Scatter(x=daily["date"], y=daily["revenue"], mode="lines", name="Revenue"))
    fig_grad.update_layout(template="plotly_white", height=280, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_grad, use_container_width=True)

stats = pd.DataFrame({
    "Metric": ["Purchases", "CTR %", "Avg Frequency"],
    "Value": [int(w["purchases"].sum()),
              f"{(w['clicks'].sum()/max(w['imps'].sum(),1)*100):.1f}%",
              f"{w['freq'].replace(0,np.nan).mean():.1f}"]
})
st.dataframe(stats, hide_index=True, use_container_width=True)

funnel_df = pd.DataFrame({
    "Stage": ["Impressions", "Clicks", "Add to Cart", "Purchases"],
    "Value": [w["imps"].sum(), w["clicks"].sum(), w["atc"].sum(), w["purchases"].sum()]
})
funnel = px.funnel(funnel_df, x="Value", y="Stage", template="plotly_white")
funnel.update_layout(height=300)
st.plotly_chart(funnel, use_container_width=True)

st.divider()

# ---------- Section C — Top 5 Leaks ----------
st.subheader("🔍 Top 5 Money Leaks")
plc_perf = (w.groupby("placement", dropna=False)
              .agg(spend=("spend","sum"), roas=("roas","mean"), waste=("waste","sum"))
              .reset_index()
              .sort_values("waste", ascending=False))

leaks = []
for _, r in plc_perf.head(3).iterrows():
    leaks.append({
        "Leak": r["placement"],
        "Wasted $": round(r["waste"],0),
        "Why": f"ROAS {r['roas']:.2f} — underperforming",
        "Fix": "Shift budget to stronger placements",
        "Confidence": "High" if r["roas"] < breakeven_roas else "Medium"
    })

low_as = (w[(w["roas"] < breakeven_roas) & (w["spend"] > TOTAL_SPEND*0.03)]
          .groupby("ad_set", dropna=False)
          .agg(spend=("spend","sum"), roas=("roas","mean"))
          .reset_index()
          .sort_values("spend", ascending=False)
          .head(2))
for _, r in low_as.iterrows():
    leaks.append({
        "Leak": f"Ad set {r['ad_set']}",
        "Wasted $": round(r["spend"],0),
        "Why": f"ROAS {r['roas']:.2f} — below breakeven",
        "Fix": "Pause or refresh creative",
        "Confidence": "High"
    })

leaks_df = pd.DataFrame(leaks).sort_values("Wasted $", ascending=False).head(5)
st.dataframe(leaks_df, hide_index=True, use_container_width=True)

if not plc_perf.empty:
    fig_waste = go.Figure(data=[go.Pie(labels=plc_perf.head(5)["placement"],
                                       values=plc_perf.head(5)["waste"], hole=0.5)])
    fig_waste.update_layout(template="plotly_white", height=250, showlegend=True)
    st.plotly_chart(fig_waste, use_container_width=True)

st.divider()

# ---------- Section D — Fix Checklist ----------
st.subheader("✅ Fix Checklist")
recov = [
    ("Pause worst placements", WASTED*0.6),
    ("Refresh creatives", WASTED*0.3),
    ("Kill zero-conversion ads", WASTED*0.05),
    ("Cap frequency", WASTED*0.04),
    ("Exclude low-intent surfaces", WASTED*0.01),
]
recov_df = pd.DataFrame(recov, columns=["Action","Est. Recovery $"]).round(0)
st.dataframe(recov_df, hide_index=True, use_container_width=True)

bar = go.Figure(data=[go.Bar(x=recov_df["Action"], y=recov_df["Est. Recovery $"])])
bar.update_layout(template="plotly_white", height=260, margin=dict(b=80))
st.plotly_chart(bar, use_container_width=True)

st.markdown(f"**Total Recovery Potential: ${WASTED:,.0f}/month**")

st.divider()

# ---------- Section E — Forecast ----------
st.subheader("⚡ Forecast")
forecast = go.Figure()
forecast.add_bar(name="Effective Now", x=["Spend"], y=[EFFECTIVE_NOW])
forecast.add_bar(name="After Fixes", x=["Spend"], y=[TOTAL_SPEND])
forecast.update_layout(barmode="group", template="plotly_white", height=250)
st.plotly_chart(forecast, use_container_width=True)

proj = pd.DataFrame({
    "Horizon":["Month","Quarter","Year"],
    "Loss":[WASTED,WASTED*3,WASTED*12]
})
loss = go.Figure(data=[go.Bar(x=proj["Horizon"], y=proj["Loss"], marker_color="crimson")])
loss.update_layout(template="plotly_white", height=250)
st.plotly_chart(loss, use_container_width=True)

st.caption(f"⚠️ Every day you delay ≈ ${DAILY_WASTE:,.0f} leaks out.")

st.divider()

# ---------- Section F — Opportunity Map ----------
st.subheader("🗺️ Opportunity Map")
if not leaks_df.empty:
    pts = []
    for _, r in leaks_df.iterrows():
        impact = float(r["Wasted $"]) / max(WASTED,1)
        effort = 0.35 if "Ad set" not in str(r["Leak"]) else 0.55
        pts.append((r["Leak"], impact, effort))
    ie = pd.DataFrame(pts, columns=["Leak","Impact","Effort"])
    fig_ie = go.Figure()
    fig_ie.add_shape(type="line", x0=0.5, x1=0.5, y0=0, y1=1, line=dict(color="lightgray"))
    fig_ie.add_shape(type="line", x0=0, x1=1, y0=0.5, y1=0.5, line=dict(color="lightgray"))
    fig_ie.add_trace(go.Scatter(x=ie["Effort"], y=ie["Impact"], mode="markers+text",
                                text=ie["Leak"], textposition="top center"))
    fig_ie.update_layout(template="plotly_white", height=300, xaxis_title="Effort", yaxis_title="Impact")
    st.plotly_chart(fig_ie, use_container_width=True)

st.divider()

# ---------- Section G — Benchmarks ----------
st.subheader("📊 Benchmarks")
bench = pd.DataFrame({
    "Metric":["Your CTR %","Benchmark CTR %","Your ROAS","Benchmark ROAS","Your Frequency","Benchmark Max Freq"],
    "Value":[f"{(w['clicks'].sum()/max(w['imps'].sum(),1)*100):.1f}%", "1.5",
             f"{AVG_ROAS:.2f}", "4.0",
             f"{w['freq'].replace(0,np.nan).mean():.1f}", "4.0"]
})
st.dataframe(bench, hide_index=True, use_container_width=True)

st.divider()

# ---------- Section H — CTA ----------
st.markdown("🚀 We can implement these fixes, monitor results, and run monthly audits so leaks don’t reopen.")
st.link_button("📅 Book your fix call", "https://calendly.com/", type="primary")
