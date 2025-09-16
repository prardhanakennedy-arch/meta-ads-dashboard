# streamlit_app.py — Investor-ready, minimal UI (USD only)

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------- Page ----------
st.set_page_config(page_title="AI Revenue Leak Report", page_icon="📊", layout="wide")
st.title("📊 AI-Powered Revenue Leak Report")

# ---------- Sidebar: Upload + simple options ----------
with st.sidebar:
    st.header("Upload CSV")
    up = st.file_uploader("Drag & drop Meta/Ads CSV", type=["csv"])
    st.caption("Columns like Amount Spent (USD), Impressions, Clicks, CTR, Purchases, Conversion Value (USD), Purchase ROAS, Frequency are supported.")
    st.divider()
    st.header("Leak Rules")
    breakeven_roas = st.number_input("Breakeven ROAS", 0.1, 10.0, 1.0, 0.1)
    fatigue_freq = st.number_input("Fatigue: Frequency >", 1.0, 15.0, 5.0, 0.5)
    fatigue_ctr = st.number_input("Fatigue: CTR < (%)", 0.0, 10.0, 0.5, 0.1)

if not up:
    st.info("Upload a CSV to generate the investor-ready one-page dashboard.")
    st.stop()

# ---------- Helpers ----------
def norm(col: str) -> str:
    return re.sub(r"\s+", "_", col.strip().lower())

def pick(cols, *alts):
    for a in alts:
        if a in cols: return a
    return None

def to_num(s: pd.Series) -> pd.Series:
    if s is None: return pd.Series(dtype=float)
    if s.dtype.kind in "ifu": return pd.to_numeric(s, errors="coerce").fillna(0.0)
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("%", "", regex=False)
         .apply(lambda x: pd.to_numeric(x, errors="coerce"))
         .fillna(0.0)
    )

# ---------- Load & normalize BEFORE any plotting ----------
raw = pd.read_csv(up)
raw.columns = [norm(c) for c in raw.columns]
cols = set(raw.columns)

DATE = pick(cols, "date", "day", "reporting_starts")
CAMPAIGN = pick(cols, "campaign", "campaign_name")
ADSET = pick(cols, "ad_set", "adset", "ad_set_name", "adset_name")
PLACEMENT = pick(cols, "placement")
SPEND = pick(cols, "amount_spent_(usd)", "spend_usd", "amount_spent", "spend")
IMPS = pick(cols, "impressions")
CLICKS = pick(cols, "clicks")
CTR = pick(cols, "ctr_%", "ctr")
PUR = pick(cols, "purchases", "results")
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
w.loc[w["ctr"].eq(0) & w["imps"].gt(0), "ctr"] = (w["clicks"]/w["imps"])*100
w["purchases"] = to_num(raw.get(PUR))
w["revenue"] = to_num(raw.get(REV))
w["roas"] = to_num(raw.get(ROAS))
w.loc[w["roas"].eq(0) & w["spend"].gt(0) & w["revenue"].gt(0), "roas"] = w["revenue"]/w["spend"]
w["freq"] = to_num(raw.get(FREQ))

# KPIs
TOTAL_SPEND = float(w["spend"].sum())
TOTAL_REV = float(w["revenue"].sum())
AVG_ROAS = (TOTAL_REV / TOTAL_SPEND) if TOTAL_SPEND > 0 else 0.0

# Estimate waste = (ROAS < breakeven) + fatigue (freq>th & ctr<th) + poor placements rel. to account
w["waste_low_roas"] = np.where(w["roas"] < breakeven_roas, w["spend"], 0.0)
w["waste_fatigue"] = np.where((w["freq"] > fatigue_freq) & (w["ctr"] < fatigue_ctr), w["spend"]*0.5, 0.0)
half_acc = max(breakeven_roas, 0.5 * AVG_ROAS) if AVG_ROAS > 0 else breakeven_roas
plc = w.groupby("placement", dropna=False).agg(spend=("spend","sum"), roas=("roas","mean")).reset_index()
plc["plc_waste"] = np.where(plc["roas"] < half_acc, plc["spend"], 0.0)
w = w.merge(plc[["placement","plc_waste"]], on="placement", how="left").fillna({"plc_waste": 0.0})
w["waste"] = w[["waste_low_roas","waste_fatigue","plc_waste"]].sum(axis=1)

WASTED = float(w["waste"].sum())
EFFECTIVE_NOW = max(TOTAL_SPEND - WASTED, 0.0)
DAILY_WASTE = WASTED/30.0 if WASTED>0 else 0.0

# ---------- 1) Executive Summary ----------
st.markdown(
    f"### 🚨 You're leaking **~${WASTED:,.0f}** this month\n"
    "Fixing these leaks now can put this money back into sales."
)
st.markdown("**What’s a leak?** Any ad spend that doesn’t bring sales. We plug leaks and move budget into what *does* work.")

k1,k2,k3,k4 = st.columns(4)
k1.metric("Total Spend", f"${TOTAL_SPEND:,.0f}")
k2.metric("Wasted Spend (leak)", f"${WASTED:,.0f}")
k3.metric("Average ROAS", f"{AVG_ROAS:.2f}")
k4.metric("Potential Monthly Savings", f"${WASTED:,.0f}")

# Donut: Effective vs Waste
donut = go.Figure(data=[go.Pie(
    labels=["Effective Spend","Wasted Spend"],
    values=[EFFECTIVE_NOW, WASTED],
    hole=0.6, textinfo="label+percent"
)])
donut.update_layout(height=250, margin=dict(l=10,r=10,t=0,b=0), showlegend=False, template="plotly_white")
st.plotly_chart(donut, use_container_width=True)
st.caption(f"Every day you wait ≈ **${DAILY_WASTE:,.0f}** leaks out of your ads.")

st.divider()

# ---------- 2) Account at a Glance ----------
st.subheader("2) Your Account at a Glance")

cA, cB = st.columns([2,1])

# Gradient line: Spend vs Revenue (build AFTER w exists)
if w["date"].notna().any():
    daily = w.groupby("date").agg(spend=("spend","sum"), revenue=("revenue","sum")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["spend"], mode="lines",
        line=dict(width=5, color="rgba(59,130,246,1)"),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.18)", name="Spend"
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["revenue"], mode="lines",
        line=dict(width=5, color="rgba(34,197,94,1)"),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.18)", name="Revenue"
    ))
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10), template="plotly_white", xaxis_title="", yaxis_title="USD")
    cA.plotly_chart(fig, use_container_width=True)
else:
    cA.info("No date column detected; showing totals only.")

# Simple KPIs (non-technical)
ctr_overall = (w["clicks"].sum()/w["imps"].sum()*100.0) if w["imps"].sum()>0 else 0.0
freq_mean = w["freq"].replace(0,np.nan).mean() if "freq" in w.columns else 0.0
stats = pd.DataFrame({
    "Metric":["Total Purchases","CTR (engagement)","Ad Frequency (avg)"],
    "Value":[int(w['purchases'].sum()), f"{ctr_overall:.1f}%", f"{freq_mean:.1f}"]
})
cB.dataframe(stats, hide_index=True, use_container_width=True)

st.divider()

# ---------- 3) Top 5 Money Leaks ----------
st.subheader("3) Top 5 Money Leaks")
# Placement-driven leaks first
plc_perf = (w.groupby("placement", dropna=False)
              .agg(spend=("spend","sum"), roas=("roas","mean"), waste=("waste","sum"))
              .reset_index().sort_values("waste", ascending=False))
leaks = []
for _, r in plc_perf.head(3).iterrows():
    leaks.append({
        "Leak": f"{r['placement']}",
        "Wasted $": round(float(r["waste"]),2),
        "Why it matters": f"ROAS {r['roas']:.2f} — budget not returning sales",
        "Fix": "Downweight/exclude & move budget to stronger placements",
        "Confidence": "High" if r["roas"]<breakeven_roas else "Medium"
    })

# Low-ROAS ad sets
low_as = (w[(w["roas"]<breakeven_roas) & (w["spend"]>max(100,0.03*TOTAL_SPEND))]
          .groupby("ad_set", dropna=False)
          .agg(spend=("spend","sum"), roas=("roas","mean")).reset_index()
          .sort_values("spend", ascending=False).head(2))
for _, r in low_as.iterrows():
    leaks.append({
        "Leak": f"Ad set: {r['ad_set']}",
        "Wasted $": round(float(r["spend"]),2),
        "Why it matters": f"ROAS {r['roas']:.2f} — below breakeven",
        "Fix": "Pause/reduce, refresh creative, refine audience",
        "Confidence": "High"
    })

leaks_df = pd.DataFrame(leaks).sort_values("Wasted $", ascending=False).head(5)
st.dataframe(leaks_df, hide_index=True, use_container_width=True)

st.divider()

# ---------- 4) Fix Checklist – Recover $ per action ----------
st.subheader("4) Fix Checklist – Plug the leaks & recover monthly")
# Simple recovery mapping (illustrative)
recov = [
    ("Pause worst placements (top 2)", min(WASTED*0.48, WASTED*0.60)),
    ("Refresh creatives (shorter, stronger hooks)", min(WASTED*0.30, WASTED*0.35)),
    ("Kill zero-conversion ads", min(WASTED*0.03, WASTED*0.05)),
    ("Cap frequency & broaden audience", min(WASTED*0.06, WASTED*0.08)),
    ("Exclude low-intent surfaces (e.g., Explore)", min(WASTED*0.03, WASTED*0.05)),
]
recov_df = pd.DataFrame(recov, columns=["Action","Est. Monthly Recovery $"])
recov_df["Est. Monthly Recovery $"] = recov_df["Est. Monthly Recovery $"].round(0)
st.dataframe(recov_df, hide_index=True, use_container_width=True)

st.markdown(f"**Total Recovery Potential:** **${WASTED:,.0f}/month**")

st.divider()

# ---------- 5) Forecast – Act Now ----------
st.subheader("5) Forecast – What happens if you act now?")
forecast = go.Figure()
forecast.add_bar(name="Effective (Now)", x=["Spend"], y=[EFFECTIVE_NOW])
forecast.add_bar(name="Effective (After Fixes)", x=["Spend"], y=[TOTAL_SPEND])
forecast.update_layout(barmode="group", height=260, margin=dict(l=10,r=10,t=10,b=10), template="plotly_white", yaxis_title="USD")
st.plotly_chart(forecast, use_container_width=True)
st.caption(f"⚠️ Every day you delay, ~**${DAILY_WASTE:,.0f}** leaks out of your ads.")

st.divider()

# ---------- 6) Next Steps / CTA ----------
st.subheader("6) Next Steps – Don’t let this month slip away")
st.markdown(
    "- Run monthly AI-powered audits\n"
    "- Fix the leaks (placements, creatives, tracking)\n"
    "- Continuous monitoring so leaks never reopen"
)
st.link_button("📅 Book your fix call", "https://calendly.com/", type="primary")
