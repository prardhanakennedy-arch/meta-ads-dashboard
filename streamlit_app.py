# streamlit_app.py — AI-Powered Revenue Leak Report (USD-only, investor-ready)

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
    st.caption("Supported columns: Spend, Impressions, Clicks, CTR, CPC, CPM, Add to Cart, Purchases, Conversion Value, Purchase ROAS, Frequency, Date, Campaign, Ad Set, Placement.")
    st.divider()
    st.header("Leak Rules")
    breakeven_roas = st.number_input("Breakeven ROAS", 0.1, 10.0, 1.0, 0.1)
    fatigue_freq = st.number_input("Fatigue if Frequency >", 1.0, 15.0, 5.0, 0.5)
    fatigue_ctr = st.number_input("Fatigue if CTR < (%)", 0.0, 10.0, 0.5, 0.1)

if not up:
    st.info("Upload a CSV to generate the one-page investor-ready dashboard.")
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

DATE = pick(cols, "date", "day", "reporting_starts", "reporting_date")
CAMPAIGN = pick(cols, "campaign", "campaign_name")
ADSET = pick(cols, "ad_set", "adset", "adset_name", "ad_set_name")
PLACEMENT = pick(cols, "placement", "platform_position")
SPEND = pick(cols, "amount_spent_(usd)", "spend_usd", "amount_spent", "spend")
IMPS = pick(cols, "impressions")
CLICKS = pick(cols, "clicks")
CTR = pick(cols, "ctr_%", "ctr")
CPC = pick(cols, "cpc_usd", "cpc")
CPM = pick(cols, "cpm_usd", "cpm")
ATC = pick(cols, "add_to_cart", "adds_to_cart")
PUR = pick(cols, "purchases", "results", "purchase")
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

# Leak estimation rules
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

# ---------- 1) Executive Summary (FOMO) ----------
st.markdown(
    f"### 🚨 You're leaking **~${WASTED:,.0f}** this month\n"
    "Fixing these leaks now can put this money back into sales."
)
st.markdown("**What’s a leak?** Any ad spend that doesn’t bring sales. We plug leaks and move budget into what *does* work.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Spend", f"${TOTAL_SPEND:,.0f}")
k2.metric("Wasted Spend (leak)", f"${WASTED:,.0f}")
k3.metric("Average ROAS", f"{AVG_ROAS:.2f}")
k4.metric("Potential Monthly Savings", f"${WASTED:,.0f}")

st.markdown(f"#### ⏳ Daily leak: **${DAILY_WASTE:,.0f}**  •  Weekly: **${DAILY_WASTE*7:,.0f}**  •  Monthly: **${WASTED:,.0f}**")
st.caption("Every day you delay, this amount likely leaks from your ads.")

# Donut: Effective vs Wasted
donut = go.Figure(data=[go.Pie(labels=["Effective Spend", "Wasted Spend"], values=[EFFECTIVE_NOW, WASTED], hole=0.6, textinfo="label+percent")])
donut.update_layout(height=260, margin=dict(l=10, r=10, t=0, b=0), template="plotly_white", showlegend=False)
st.plotly_chart(donut, use_container_width=True)

st.divider()

# ---------- 2) Your Account at a Glance ----------
st.subheader("2) Your Account at a Glance")
colA, colB = st.columns([2, 1])

# Gradient line: Spend vs Revenue
if w["date"].notna().any():
    daily = w.groupby("date").agg(spend=("spend", "sum"), revenue=("revenue", "sum")).reset_index()
    fig_grad = go.Figure()
    fig_grad.add_trace(go.Scatter(x=daily["date"], y=daily["spend"], mode="lines",
                                  line=dict(width=5, color="rgba(59,130,246,1)"),
                                  fill="tozeroy", fillcolor="rgba(59,130,246,0.18)", name="Spend"))
    fig_grad.add_trace(go.Scatter(x=daily["date"], y=daily["revenue"], mode="lines",
                                  line=dict(width=5, color="rgba(34,197,94,1)"),
                                  fill="tozeroy", fillcolor="rgba(34,197,94,0.18)", name="Revenue"))
    fig_grad.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                           xaxis_title="", yaxis_title="USD")
    colA.plotly_chart(fig_grad, use_container_width=True)
else:
    colA.info("No date column detected; showing totals only.")

# Simple KPIs
ctr_overall = (w["clicks"].sum() / w["imps"].sum() * 100.0) if w["imps"].sum() > 0 else 0.0
freq_mean = w["freq"].replace(0, np.nan).mean() if "freq" in w.columns else 0.0
stats = pd.DataFrame({"Metric": ["Total Purchases", "CTR (engagement)", "Ad Frequency (avg)"],
                      "Value": [int(w["purchases"].sum()), f"{ctr_overall:.1f}%", f"{freq_mean:.1f}"]})
colB.dataframe(stats, hide_index=True, use_container_width=True)

# Funnel (Impressions → Clicks → ATC → Purchases)
funnel_df = pd.DataFrame({"Stage": ["Impressions", "Clicks", "Add to Cart", "Purchases"],
                          "Value": [w["imps"].sum(), w["clicks"].sum(), w["atc"].sum(), w["purchases"].sum()]})
fig_funnel = px.funnel(funnel_df, x="Value", y="Stage", template="simple_white")
fig_funnel.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
st.subheader("Where revenue leaks in your funnel")
st.plotly_chart(fig_funnel, use_container_width=True)
st.caption("Fix ideas: improve hooks (impressions→clicks), landing page (clicks→add to cart), checkout (add to cart→purchases).")

st.divider()

# ---------- 3) Top 5 Money Leaks ----------
st.subheader("3) Top 5 Money Leaks")

# Placement waste
plc_perf = (w.groupby("placement", dropna=False)
              .agg(spend=("spend", "sum"), roas=("roas", "mean"), waste=("waste", "sum"))
              .reset_index()
              .sort_values("waste", ascending=False))

leaks = []
for _, r in plc_perf.head(3).iterrows():
    leaks.append({
        "Leak": f"{r['placement']}",
        "Wasted $": round(float(r["waste"]), 2),
        "Why it matters": f"ROAS {r['roas']:.2f} — budget not returning sales",
        "Fix": "Downweight/exclude & move budget to stronger placements",
        "Confidence": "High" if r["roas"] < breakeven_roas else "Medium"
    })

# Low-ROAS ad sets
low_as = (w[(w["roas"] < breakeven_roas) & (w["spend"] > max(100, 0.03 * TOTAL_SPEND))]
          .groupby("ad_set", dropna=False)
          .agg(spend=("spend", "sum"), roas=("roas", "mean"))
          .reset_index()
          .sort_values("spend", ascending=False)
          .head(2))
for _, r in low_as.iterrows():
    leaks.append({
        "Leak": f"Ad set: {r['ad_set']}",
        "Wasted $": round(float(r["spend"]), 2),
        "Why it matters": f"ROAS {r['roas']:.2f} — below breakeven",
        "Fix": "Pause/reduce, refresh creative, refine audience",
        "Confidence": "High"
    })

leaks_df = pd.DataFrame(leaks).sort_values("Wasted $", ascending=False).head(5)
st.dataframe(leaks_df, hide_index=True, use_container_width=True)

# Waste share by placement (pie)
if not plc_perf.empty:
    waste_by_plc = plc_perf.head(5)
    fig_waste_pie = go.Figure(data=[go.Pie(labels=waste_by_plc["placement"], values=waste_by_plc["waste"], hole=0.5, textinfo="label+percent")])
    fig_waste_pie.update_layout(height=260, template="plotly_white", margin=dict(l=10, r=10, t=0, b=0), showlegend=False)
    st.plotly_chart(fig_waste_pie, use_container_width=True)

st.divider()

# ---------- 4) Fix Checklist – Recover $ ----------
st.subheader("4) Fix Checklist – Plug the leaks & recover monthly")

# Recovery estimates (illustrative split of total waste)
recov = [
    ("Pause worst placements (top 2)", min(WASTED * 0.55, WASTED * 0.65)),
    ("Refresh creatives (shorter, stronger hooks)", min(WASTED * 0.25, WASTED * 0.35)),
    ("Kill zero-conversion ads", min(WASTED * 0.03, WASTED * 0.05)),
    ("Cap frequency & broaden audience", min(WASTED * 0.05, WASTED * 0.08)),
    ("Exclude low-intent surfaces (e.g., Explore)", min(WASTED * 0.02, WASTED * 0.05)),
]
recov_df = pd.DataFrame(recov, columns=["Action", "Est. Monthly Recovery $"])
recov_df["Est. Monthly Recovery $"] = recov_df["Est. Monthly Recovery $"].round(0)
st.dataframe(recov_df, hide_index=True, use_container_width=True)
st.markdown(f"**Total Recovery Potential:** **${WASTED:,.0f}/month**")

# Recovery bar
fig_recovery = go.Figure(data=[go.Bar(x=recov_df["Action"], y=recov_df["Est. Monthly Recovery $"])])
fig_recovery.update_layout(height=260, template="plotly_white", margin=dict(l=10, r=10, t=10, b=80), yaxis_title="USD", xaxis_tickangle=-20)
st.plotly_chart(fig_recovery, use_container_width=True)

st.divider()

# ---------- 5) If you do nothing… (loss forecast) ----------
st.subheader("5) Forecast – What happens if you act now?")
forecast = go.Figure()
forecast.add_bar(name="Effective (Now)", x=["Spend"], y=[EFFECTIVE_NOW])
forecast.add_bar(name="Effective (After Fixes)", x=["Spend"], y=[TOTAL_SPEND])
forecast.update_layout(barmode="group", height=260, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), yaxis_title="USD")
st.plotly_chart(forecast, use_container_width=True)

st.subheader("If you do nothing… expected money lost")
proj = pd.DataFrame({"Horizon": ["This month", "Quarter (90d)", "Year"], "Loss": [WASTED, WASTED * 3, WASTED * 12]})
fig_loss = go.Figure(data=[go.Bar(x=proj["Horizon"], y=proj["Loss"], marker_color="crimson")])
fig_loss.update_layout(height=280, template="plotly_white", yaxis_title="USD", margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_loss, use_container_width=True)
st.caption(f"⚠️ Every day you delay, ~**${DAILY_WASTE:,.0f}** leaks out of your ads.")

st.divider()

# ---------- 6) Opportunity map (Impact vs Effort) ----------
st.subheader("6) Opportunity map — do these first")
def _ie_score(row):
    impact = min(1.0, float(row["Wasted $"]) / max(WASTED, 1)) if "Wasted $" in row else 0.5
    effort = 0.35 if "Ad set" not in str(row["Leak"]) else 0.55
    return impact, effort

if not leaks_df.empty:
    pts = []
    for _, r in leaks_df.iterrows():
        imp, eff = _ie_score(r)
        pts.append((r["Leak"], imp, eff))
    ie = pd.DataFrame(pts, columns=["Leak", "Impact", "Effort"])
    fig_ie = go.Figure()
    fig_ie.add_shape(type="line", x0=0.5, x1=0.5, y0=0, y1=1, line=dict(color="lightgray"))
    fig_ie.add_shape(type="line", x0=0, x1=1, y0=0.5, y1=0.5, line=dict(color="lightgray"))
    fig_ie.add_trace(go.Scatter(x=ie["Effort"], y=ie["Impact"], mode="markers+text", text=ie["Leak"], textposition="top center", marker=dict(size=14)))
    fig_ie.update_layout(height=320, template="plotly_white", xaxis_title="Effort", yaxis_title="Impact", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_ie, use_container_width=True)

st.divider()

# ---------- 7) Benchmarks (context) ----------
st.subheader("7) Benchmarks — are you ahead or behind?")
bench = {"CTR % (good)": 1.5, "ROAS (good)": 4.0, "Freq (max)": 4.0}
bm_ctr = (w["clicks"].sum()/w["imps"].sum()*100.0) if w["imps"].sum() > 0 else 0.0
bm_freq = w["freq"].replace(0, np.nan).mean() if "freq" in w.columns else 0.0
bm = pd.DataFrame({
    "Metric": ["Your CTR %", "Benchmark CTR %", "Your ROAS", "Benchmark ROAS", "Your Frequency", "Benchmark Max Freq"],
    "Value": [round(bm_ctr, 1), bench["CTR % (good)"], round(AVG_ROAS, 2), bench["ROAS (good)"], round(bm_freq, 1), bench["Freq (max)"]],
})
st.dataframe(bm, hide_index=True, use_container_width=True)

# Optional: Meta vs Google split (if campaign names contain "google")
def _platform(name):
    s = str(name).lower()
    return "Google" if "google" in s else "Meta"
if "campaign" in w.columns:
    w["platform"] = w["campaign"].apply(_platform)
    by_plat = w.groupby("platform").agg(spend=("spend","sum"), revenue=("revenue","sum"), waste=("waste","sum")).reset_index()
    if by_plat["platform"].nunique() > 1:
        fig_plat = go.Figure(data=[
            go.Bar(name="Spend", x=by_plat["platform"], y=by_plat["spend"]),
            go.Bar(name="Revenue", x=by_plat["platform"], y=by_plat["revenue"]),
            go.Bar(name="Wasted", x=by_plat["platform"], y=by_plat["waste"]),
        ])
        fig_plat.update_layout(barmode="group", height=280, template="plotly_white", yaxis_title="USD", margin=dict(l=10, r=10, t=10, b=10))
        st.subheader("Channel mix — Meta vs Google")
        st.plotly_chart(fig_plat, use_container_width=True)

st.divider()
st.markdown("**Next steps** — We can implement these fixes for you, monitor results, and run monthly audits so leaks don’t reopen. Book a call to get started.")
st.link_button("📅 Book your fix call", "https://calendly.com/", type="primary")
