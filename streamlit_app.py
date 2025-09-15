# streamlit_app.py
# --- AI-Powered Revenue Leak Dashboard (Meta Ads) ---
# Drag & drop a CSV exported from Meta Ads Manager and get a 1-page dashboard
# with Executive Summary, Top Leaks, Fix Checklist, and Forecast visuals.

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="AI Revenue Leak Dashboard", page_icon="📊", layout="wide")
st.title("📊 AI-Powered Revenue Leak Dashboard (Meta Ads)")
st.caption("Upload a Meta Ads CSV → get instant summary, leaks, and fix checklist. Reusable for multiple clients.")

# ------------------------------
# Helpers
# ------------------------------

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s

# Try to find the first present key among alternatives
def pick(colnames, *alts):
    for a in alts:
        if a in colnames:
            return a
    return None

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [_norm(c) for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def coerce_numeric(s):
    if s is None:
        return pd.Series(dtype=float)
    if s.dtype.kind in "ifu":
        return pd.to_numeric(s, errors='coerce').fillna(0.0)
    # strip commas, %
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("%", "", regex=False)
         .apply(lambda x: pd.to_numeric(x, errors='coerce'))
         .fillna(0.0)
    )

# Compute leaks given a normalized dataframe
@st.cache_data(show_spinner=False)
def analyze(df: pd.DataFrame):
    cols = set(df.columns)

    # Map flexible schemas (Meta exports vary a lot)
    campaign_col = pick(cols, "campaign", "campaign_name")
    adset_col    = pick(cols, "ad_set", "adset", "ad_set_name", "adset_name")
    placement_col= pick(cols, "placement")

    # Spend can be amount_spent_(inr) or amount_spent or spend
    spend_col = pick(cols, "amount_spent_(inr)", "amount_spent", "spend", "spend_inr")
    # Results / purchases
    results_col = pick(cols, "results", "purchases", "purchase")
    # ROAS
    roas_col = pick(cols, "purchase_roas_(return_on_ad_spend)", "purchase_roas", "roas")
    # Engagement
    ctr_col = pick(cols, "ctr")
    clicks_col = pick(cols, "clicks")
    imps_col = pick(cols, "impressions")
    freq_col = pick(cols, "frequency")
    cpr_col  = pick(cols, "cost_per_result", "cpr")

    # Create a working frame
    work = pd.DataFrame()
    for name, col in [
        ("campaign", campaign_col),
        ("ad_set", adset_col),
        ("placement", placement_col)
    ]:
        work[name] = df.get(col, pd.Series([None]*len(df)))

    work["spend"] = coerce_numeric(df.get(spend_col, pd.Series([0]*len(df))))
    work["results"] = coerce_numeric(df.get(results_col, pd.Series([0]*len(df))))
    work["roas"] = coerce_numeric(df.get(roas_col, pd.Series([0]*len(df))))
    work["impressions"] = coerce_numeric(df.get(imps_col, pd.Series([0]*len(df))))
    work["clicks"] = coerce_numeric(df.get(clicks_col, pd.Series([0]*len(df))))
    work["ctr"] = coerce_numeric(df.get(ctr_col, pd.Series([0]*len(df))))
    # derive CTR if missing
    work.loc[work["ctr"].eq(0) & work["impressions"].gt(0), "ctr"] = (
        work.loc[work["ctr"].eq(0) & work["impressions"].gt(0), "clicks"] / work.loc[work["ctr"].eq(0) & work["impressions"].gt(0), "impressions"] * 100.0
    )

    work["frequency"] = coerce_numeric(df.get(freq_col, pd.Series([0]*len(df))))
    work["cpr"] = coerce_numeric(df.get(cpr_col, pd.Series([0]*len(df))))

    # Totals & key figures
    total_spend = float(work["spend"].sum())
    # If no roas column, approximate: assume revenue unknown; set roas by results*avg_value/spend —
    # but we keep it simple and use provided ROAS when available.
    avg_roas = float(work["roas"][work["roas"]>0].mean()) if (work["roas"]>0).any() else 0.0

    # Estimated wasted spend rules
    #  - Low ROAS (<1.0) => waste entire spend of that row
    #  - Underperforming placements (<max(1.0, 0.5 * account_roas)))
    #  - High frequency (>5) with low CTR (<0.5%) counts as likely waste
    
    acct_breakeven = 1.0
    acct_half_roas = max(acct_breakeven, 0.5 * avg_roas) if avg_roas>0 else 1.0

    work["waste_low_roas"] = np.where(work["roas"].lt(1.0), work["spend"], 0.0)
    work["waste_fatigue"] = np.where((work["frequency"].gt(5)) & (work["ctr"].lt(0.5)), work["spend"] * 0.5, 0.0)

    # Placement underperformance (aggregate first)
    if placement_col is not None:
        plc = work.groupby("placement", dropna=False).agg(spend=("spend","sum"), roas=("roas","mean")).reset_index()
        plc["plc_waste"] = np.where(plc["roas"].lt(acct_half_roas), plc["spend"], 0.0)
        # Map back
        work = work.merge(plc[["placement","plc_waste"]], on="placement", how="left")
    else:
        work["plc_waste"] = 0.0

    work["estimated_waste"] = work[["waste_low_roas","waste_fatigue","plc_waste"]].sum(axis=1)

    # Build leak items
    leaks = []

    # 1) Placement leaks (top 3)
    if placement_col is not None:
        top_plc = (
            work.groupby("placement", dropna=False)
                .agg(spend=("spend","sum"), roas=("roas","mean"), waste=("estimated_waste","sum"))
                .sort_values("waste", ascending=False)
                .head(3)
                .reset_index()
        )
        for _, r in top_plc.iterrows():
            leaks.append({
                "Leak": f"Weak Placement – {r['placement']}",
                "Wasted_₹": float(r['waste']),
                "ROAS": float(r['roas']),
                "Recommendation": "Exclude/downweight and reallocate to stronger placements",
                "Confidence": "High" if r['roas'] < 1.0 else "Medium"
            })

    # 2) Low-ROAS ad sets
    if adset_col is not None:
        low_as = (
            work[work["roas"].lt(1.0) & work["spend"].gt(max(1000, 0.03*total_spend))]
                .groupby("ad_set", dropna=False)
                .agg(spend=("spend","sum"), roas=("roas","mean"))
                .reset_index()
                .sort_values("spend", ascending=False)
                .head(2)
        )
        for _, r in low_as.iterrows():
            leaks.append({
                "Leak": f"Low ROAS Ad Set – {r['ad_set']}",
                "Wasted_₹": float(r['spend']),
                "ROAS": float(r['roas']),
                "Recommendation": "Pause/reduce budget, refresh creatives, refine audience",
                "Confidence": "High"
            })

    leaks_df = pd.DataFrame(leaks)
    wasted_total = float(leaks_df["Wasted_₹"].sum()) if not leaks_df.empty else 0.0

    # Forecast (simple): remove waste, keep spend constant
    effective_now = max(total_spend - wasted_total, 0.0)
    effective_after = total_spend

    out = {
        "work": work,
        "leaks": leaks_df.sort_values("Wasted_₹", ascending=False).head(5),
        "totals": {
            "total_spend": total_spend,
            "avg_roas": avg_roas,
            "wasted_total": wasted_total,
            "effective_now": effective_now,
            "effective_after": effective_after,
            "daily_waste": wasted_total/30.0 if wasted_total>0 else 0.0
        }
    }
    return out

# ------------------------------
# UI
# ------------------------------
left, right = st.columns([1,1])
with left:
    client = st.text_input("Client Name", placeholder="Acme Co.")
with right:
    daterange = st.text_input("Date Range", placeholder="e.g., 2025-08-01 to 2025-08-31")

uploaded = st.file_uploader("Drag & Drop Meta Ads CSV here", type=["csv"])  

if not uploaded:
    st.info("Upload a Meta Ads CSV export to populate the dashboard. Columns like `amount_spent_(inr)`, `results`, `purchase_roas_(return_on_ad_spend)`, `placement` are supported.")
    st.stop()

# Load + analyze
raw_df = load_csv(uploaded)
result = analyze(raw_df)
work = result["work"]
leaks = result["leaks"]
T = result["totals"]

# ------------------------------
# 1) Executive Summary
# ------------------------------
st.subheader("1) Executive Summary")
met1, met2, met3, met4 = st.columns(4)
met1.metric("Total Spend", f"₹{T['total_spend']:,.0f}")
met2.metric("Estimated Wasted Spend", f"₹{T['wasted_total']:,.0f}", help="Budget likely not returning sales")
met3.metric("Average ROAS", f"{T['avg_roas']:.2f}")
met4.metric("Potential Monthly Savings", f"₹{T['wasted_total']:,.0f}")

# Pie: Effective vs Waste
pie = go.Figure(data=[go.Pie(labels=["Effective Spend","Wasted Spend"], values=[T['effective_now'], T['wasted_total']], hole=0.35)])
pie.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10))
st.plotly_chart(pie, use_container_width=True)

st.caption(f"⚠️ Every day you wait ≈ ₹{T['daily_waste']:,.0f} leaks out of your ads.")

# ------------------------------
# 2) Top Leaks (Headlines)
# ------------------------------
st.subheader("2) Top Leaks (Headlines)")
if leaks.empty:
    st.success("No major leaks detected under current thresholds. Nice work!")
else:
    leaks_show = leaks.rename(columns={"Wasted_₹":"Wasted (₹)"})
    st.dataframe(leaks_show, use_container_width=True)

# ------------------------------
# 3) Fix Checklist (Recoverable Revenue)
# ------------------------------
st.subheader("3) Fix Checklist (Recoverable Revenue)")
q1, q2 = st.columns(2)
with q1:
    st.markdown("**Quick Wins (This Week)**")
    st.markdown("- Pause weak placements shown above (Feed, Reels, Explore)\n- Kill zero-conversion ads/ad sets\n- Upload fresh creatives (shorter videos, stronger hooks/CTAs)")
with q2:
    st.markdown("**Structural Fixes (This Month+)**")
    st.markdown("- Consolidate campaigns to exit learning\n- Improve checkout funnel (reduce cart drops)\n- Set up server-side Conversion API for more accurate ROAS")

# Before vs After bar
bar = go.Figure()
bar.add_bar(name="Effective (Now)", x=["Spend"], y=[T['effective_now']])
bar.add_bar(name="Effective (After Fixes)", x=["Spend"], y=[T['effective_after']])
bar.update_layout(barmode='group', height=300, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="₹")
st.plotly_chart(bar, use_container_width=True)

# ------------------------------
# 4) Forecast Highlights
# ------------------------------
st.subheader("4) Forecast Highlights")
colA, colB, colC = st.columns(3)
colA.metric("Effective Spend (Now)", f"₹{T['effective_now']:,.0f}")
colB.metric("Effective Spend (After Fixes)", f"₹{T['effective_after']:,.0f}")
colC.metric("Daily Waste", f"₹{T['daily_waste']:,.0f}")

# ------------------------------
# Downloads
# ------------------------------
st.divider()
st.markdown("### Downloads")
# Prepare downloadable CSVs
leaks_csv = leaks.to_csv(index=False).encode('utf-8') if not leaks.empty else b""
st.download_button("⬇️ Download Top Leaks (CSV)", data=leaks_csv, file_name="top_leaks.csv", mime="text/csv", disabled=leaks_csv==b"")

clean_cols = [c for c in work.columns if c not in ("waste_low_roas","waste_fatigue","plc_waste")]
work_csv = work[clean_cols].to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Cleaned Data (CSV)", data=work_csv, file_name="cleaned_meta_ads.csv", mime="text/csv")

st.caption("Tip: Save this app as a template. For each new client, just upload their CSV — everything updates automatically.")

# Footer
st.write("—")
st.caption("Built for non-technical founders: simple language, clear actions, real savings.")
