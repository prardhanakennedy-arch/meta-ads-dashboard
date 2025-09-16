# Requirements: streamlit, pandas, numpy, plotly
# Run with: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Revenue Leak Report", layout="wide", initial_sidebar_state="expanded")

# Helper functions
def normalize_columns(df):
    """Normalize column names for flexible matching"""
    normalized = {}
    for col in df.columns:
        normalized[col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')] = col
    return normalized

def pick(cols_dict, *names):
    """Pick first matching column from options"""
    for name in names:
        if name in cols_dict:
            return cols_dict[name]
    return None

def to_num(series):
    """Convert to numeric, filling NaN with 0"""
    return pd.to_numeric(series, errors='coerce').fillna(0)

def build_leaks(df, breakeven_roas, fatigue_freq, fatigue_ctr):
    """Calculate waste for each row"""
    df = df.copy()
    df['waste'] = 0.0
    
    # Low ROAS waste
    low_roas_mask = df['roas'] < breakeven_roas
    df.loc[low_roas_mask, 'waste'] += df.loc[low_roas_mask, 'spend']
    
    # Fatigue waste (partial)
    if 'frequency' in df.columns:
        fatigue_mask = (df['frequency'] > fatigue_freq) & (df['ctr'] < fatigue_ctr)
        df.loc[fatigue_mask, 'waste'] += df.loc[fatigue_mask, 'spend'] * 0.5
    
    # Weak placement waste
    if 'placement' in df.columns:
        placement_roas = df.groupby('placement')['roas'].mean()
        account_roas = df['roas'].mean()
        weak_threshold = max(breakeven_roas, 0.5 * account_roas)
        
        for placement in placement_roas.index:
            if placement_roas[placement] < weak_threshold:
                placement_mask = df['placement'] == placement
                df.loc[placement_mask, 'waste'] += df.loc[placement_mask, 'spend']
    
    return df

def kpis(df):
    """Calculate key metrics"""
    total_spend = df['spend'].sum()
    total_revenue = df['revenue'].sum()
    total_clicks = df['clicks'].sum()
    total_purchases = df['purchases'].sum()
    total_impressions = df['impressions'].sum()
    total_atc = df.get('add_to_cart', pd.Series([0])).sum()
    
    avg_roas = total_revenue / total_spend if total_spend > 0 else 0
    cvr = (total_purchases / total_clicks * 100) if total_clicks > 0 else 0
    aov = total_revenue / total_purchases if total_purchases > 0 else 0
    ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    avg_freq = df.get('frequency', pd.Series([0])).mean()
    
    return {
        'total_spend': total_spend,
        'total_revenue': total_revenue,
        'avg_roas': avg_roas,
        'cvr': cvr,
        'aov': aov,
        'ctr': ctr,
        'avg_freq': avg_freq,
        'total_clicks': total_clicks,
        'total_purchases': total_purchases,
        'total_impressions': total_impressions,
        'total_atc': total_atc
    }

# Sidebar controls
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("**Upload CSV & set leak rules**")

uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=['csv'])
st.sidebar.markdown("💡 *Works with Meta & Google exports. USD only.*")

st.sidebar.markdown("### Leak Detection Rules")
breakeven_roas = st.sidebar.number_input("Breakeven ROAS", value=1.0, min_value=0.1, step=0.1)
fatigue_freq = st.sidebar.number_input("Fatigue Frequency Threshold", value=5.0, min_value=1.0, step=0.5)
fatigue_ctr = st.sidebar.number_input("Fatigue CTR Threshold %", value=0.5, min_value=0.1, step=0.1)

if uploaded_file is not None:
    # Load and process data
    try:
        df = pd.read_csv(uploaded_file)
        cols = normalize_columns(df)
        
        # Map columns
        date_col = pick(cols, 'date', 'day', 'reporting_starts')
        campaign_col = pick(cols, 'campaign', 'campaign_name')
        adset_col = pick(cols, 'ad_set', 'adset', 'adset_name')
        placement_col = pick(cols, 'placement', 'platform_position')
        spend_col = pick(cols, 'amount_spent_usd', 'spend_usd', 'amount_spent', 'spend')
        impressions_col = pick(cols, 'impressions')
        clicks_col = pick(cols, 'clicks')
        ctr_col = pick(cols, 'ctr_', 'ctr')
        cpc_col = pick(cols, 'cpc_usd', 'cpc')
        cpm_col = pick(cols, 'cpm_usd', 'cpm')
        atc_col = pick(cols, 'add_to_cart', 'adds_to_cart')
        purchases_col = pick(cols, 'purchases', 'results', 'conversions')
        revenue_col = pick(cols, 'conversion_value_usd', 'conversion_value', 'purchase_value', 'revenue')
        roas_col = pick(cols, 'purchase_roas_return_on_ad_spend', 'purchase_roas', 'roas')
        freq_col = pick(cols, 'frequency')
        
        # Check required columns
        required = [spend_col, impressions_col, clicks_col]
        missing = [col for col in ['spend', 'impressions', 'clicks'] if eval(f"{col}_col") is None]
        
        if missing:
            st.error(f"❌ Missing required columns: {', '.join(missing)}. Please check your CSV headers.")
            st.stop()
        
        # Build clean dataframe
        clean_df = pd.DataFrame()
        if date_col: clean_df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        if campaign_col: clean_df['campaign'] = df[campaign_col]
        if adset_col: clean_df['adset'] = df[adset_col]
        if placement_col: clean_df['placement'] = df[placement_col]
        
        clean_df['spend'] = to_num(df[spend_col])
        clean_df['impressions'] = to_num(df[impressions_col])
        clean_df['clicks'] = to_num(df[clicks_col])
        
        # Calculate or use existing metrics
        if ctr_col:
            clean_df['ctr'] = to_num(df[ctr_col])
        else:
            clean_df['ctr'] = np.where(clean_df['impressions'] > 0, 
                                     clean_df['clicks'] / clean_df['impressions'] * 100, 0)
        
        if cpc_col:
            clean_df['cpc'] = to_num(df[cpc_col])
        else:
            clean_df['cpc'] = np.where(clean_df['clicks'] > 0, 
                                     clean_df['spend'] / clean_df['clicks'], 0)
        
        if cpm_col:
            clean_df['cpm'] = to_num(df[cpm_col])
        else:
            clean_df['cpm'] = np.where(clean_df['impressions'] > 0, 
                                     clean_df['spend'] / clean_df['impressions'] * 1000, 0)
        
        if atc_col: clean_df['add_to_cart'] = to_num(df[atc_col])
        if purchases_col: clean_df['purchases'] = to_num(df[purchases_col])
        else: clean_df['purchases'] = 0
        
        if revenue_col: clean_df['revenue'] = to_num(df[revenue_col])
        else: clean_df['revenue'] = 0
        
        if roas_col:
            clean_df['roas'] = to_num(df[roas_col])
        else:
            clean_df['roas'] = np.where(clean_df['spend'] > 0, 
                                      clean_df['revenue'] / clean_df['spend'], 0)
        
        if freq_col: clean_df['frequency'] = to_num(df[freq_col])
        
        # Calculate leaks
        leak_df = build_leaks(clean_df, breakeven_roas, fatigue_freq, fatigue_ctr)
        metrics = kpis(leak_df)
        
        wasted = leak_df['waste'].sum()
        effective_spend = metrics['total_spend'] - wasted
        daily_waste = wasted / 30
        
        # Main dashboard
        st.title("💰 Revenue Leak Report")
        
        # Section A - Executive Summary
        st.markdown("## 🚨 Executive Summary")
        st.markdown(f"### You're leaking ~${wasted:,.0f} this month.")
        st.markdown("A leak is spend that doesn't bring sales. We plug leaks and move budget into what works.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Spend", f"${metrics['total_spend']:,.0f}")
        with col2:
            st.metric("Wasted Spend", f"${wasted:,.0f}")
        with col3:
            st.metric("Average ROAS", f"{metrics['avg_roas']:.2f}")
        with col4:
            st.metric("Monthly Savings Potential", f"${wasted:,.0f}")
        
        st.markdown(f"**Daily: ${daily_waste:.0f} • Weekly: ${daily_waste*7:.0f} • Monthly: ${wasted:.0f}**")
        
        # Modern donut chart with gradient-like colors
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Effective Spend', 'Wasted Spend'],
            values=[effective_spend, wasted],
            hole=.6,
            marker_colors=['#4ECDC4', '#FF6B9D'],
            textinfo='label+percent',
            textfont_size=14,
            marker_line=dict(color='white', width=3)
        )])
        fig_donut.update_layout(
            title=dict(text="Effective vs Wasted Spend", font_size=18, x=0.5),
            height=350, 
            template='plotly_white',
            showlegend=False,
            margin=dict(t=60, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
        # Section B - Account Overview
        st.markdown("## 📈 Account at a Glance")
        
        if 'date' in clean_df.columns and not clean_df['date'].isna().all():
            daily_data = clean_df.groupby('date').agg({
                'spend': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            # Modern gradient area chart
            fig_trend = go.Figure()
            
            # Revenue area with gradient
            fig_trend.add_trace(go.Scatter(
                x=daily_data['date'], y=daily_data['revenue'],
                fill='tozeroy',
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=6, color='#4ECDC4'),
                fillcolor='rgba(78, 205, 196, 0.3)'
            ))
            
            # Spend area with gradient
            fig_trend.add_trace(go.Scatter(
                x=daily_data['date'], y=daily_data['spend'],
                fill='tozeroy',
                mode='lines+markers',
                name='Spend',
                line=dict(color='#FF6B9D', width=3),
                marker=dict(size=6, color='#FF6B9D'),
                fillcolor='rgba(255, 107, 157, 0.3)'
            ))
            
            fig_trend.update_layout(
                title=dict(text="Spend vs Revenue Trend", font_size=18, x=0.5),
                height=350,
                template='plotly_white',
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=80, b=40, l=40, r=40),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig_trend.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            fig_trend.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("📅 No date column found - skipping trend analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Purchases", f"{metrics['total_purchases']:,.0f}")
        with col2:
            st.metric("CTR %", f"{metrics['ctr']:.2f}%")
        with col3:
            st.metric("Avg Frequency", f"{metrics['avg_freq']:.1f}")
        
        # Modern funnel chart with gradients
        funnel_data = [metrics['total_impressions'], metrics['total_clicks'], 
                      metrics.get('total_atc', metrics['total_clicks'] * 0.1), metrics['total_purchases']]
        
        fig_funnel = go.Figure()
        
        # Create gradient colors for funnel
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
        
        fig_funnel.add_trace(go.Funnel(
            y=['Impressions', 'Clicks', 'Add to Cart', 'Purchases'],
            x=funnel_data,
            texttemplate='%{x:,.0f}<br>%{percentPrevious}',
            textposition='inside',
            textfont=dict(color='white', size=14),
            marker=dict(
                color=colors,
                line=dict(width=2, color='white')
            ),
            connector=dict(line=dict(color='rgba(255,255,255,0.3)', dash='dot', width=2))
        ))
        
        fig_funnel.update_layout(
            title=dict(text="Conversion Funnel", font_size=18, x=0.5),
            height=400,
            template='plotly_white',
            margin=dict(t=60, b=40, l=40, r=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
        
        # Section C - Top Leaks
        st.markdown("## 🔍 Top 5 Money Leaks")
        
        # Build leaks table
        leaks_list = []
        
        if 'placement' in leak_df.columns:
            placement_waste = leak_df.groupby('placement')['waste'].sum().sort_values(ascending=False).head(3)
            for placement, waste in placement_waste.items():
                if waste > 0:
                    leaks_list.append({
                        'Leak': f"{placement} placement",
                        'Wasted $': f"${waste:,.0f}",
                        'Why it matters': "Low-converting placement drains budget",
                        'Fix': "Pause or reduce bids",
                        'Confidence': "High"
                    })
        
        if 'adset' in leak_df.columns:
            adset_waste = leak_df[leak_df['roas'] < breakeven_roas].groupby('adset')['waste'].sum().sort_values(ascending=False).head(2)
            for adset, waste in adset_waste.items():
                if waste > 0:
                    leaks_list.append({
                        'Leak': f"{adset} ad set",
                        'Wasted $': f"${waste:,.0f}",
                        'Why it matters': "Below breakeven performance",
                        'Fix': "Refresh creative or pause",
                        'Confidence': "High"
                    })
        
        if leaks_list:
            leaks_df = pd.DataFrame(leaks_list[:5])
            st.dataframe(leaks_df, use_container_width=True, hide_index=True)
        else:
            st.info("No major leaks detected with current thresholds")
        
        # Section D - Fix Checklist
        st.markdown("## ✅ Fix Checklist")
        
        actions = [
            ("Pause worst placements", wasted * 0.6),
            ("Refresh creatives", wasted * 0.3),
            ("Kill zero-conversion ads", wasted * 0.04),
            ("Cap frequency & broaden audience", wasted * 0.05),
            ("Exclude low-intent surfaces", wasted * 0.01)
        ]
        
        action_df = pd.DataFrame(actions, columns=['Action', 'Est. Monthly Recovery $'])
        action_df['Est. Monthly Recovery $'] = action_df['Est. Monthly Recovery $'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(action_df, use_container_width=True, hide_index=True)
        
        # Modern horizontal bar chart for recovery actions
        actions_df = pd.DataFrame(actions, columns=['Action', 'Recovery'])
        
        fig_recovery = go.Figure()
        fig_recovery.add_trace(go.Bar(
            y=actions_df['Action'],
            x=actions_df['Recovery'],
            orientation='h',
            marker=dict(
                color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'],
                cornerradius=10,
                line=dict(width=0)
            ),
            text=[f'${val:,.0f}' for val in actions_df['Recovery']],
            textposition='inside',
            textfont=dict(color='white', size=12)
        ))
        
        fig_recovery.update_layout(
            title=dict(text="Recovery by Action", font_size=18, x=0.5),
            height=350,
            template='plotly_white',
            showlegend=False,
            margin=dict(t=60, b=40, l=200, r=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig_recovery.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig_recovery.update_yaxes(showgrid=False)
        st.plotly_chart(fig_recovery, use_container_width=True)
        
        st.markdown(f"**Total Recovery Potential: ${wasted:,.0f} / month**")
        
        # Section E - Forecast
        st.markdown("## ⚡ Forecast & 'Do Nothing' Loss")
        
        # Modern comparison chart with gradients
        comparison_data = pd.DataFrame({
            'Category': ['Effective Spend Now', 'Total Spend (After Fixes)'],
            'Amount': [effective_spend, metrics['total_spend']],
            'Colors': ['#4ECDC4', '#667eea']
        })
        
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            x=comparison_data['Category'],
            y=comparison_data['Amount'],
            marker=dict(
                color=comparison_data['Colors'],
                cornerradius=15,
                line=dict(width=0)
            ),
            text=[f'${val:,.0f}' for val in comparison_data['Amount']],
            textposition='outside',
            textfont=dict(size=14, color='#333')
        ))
        
        fig_comparison.update_layout(
            title=dict(text="Effective Spend: Now vs After Fixes", font_size=18, x=0.5),
            height=350,
            template='plotly_white',
            showlegend=False,
            margin=dict(t=80, b=40, l=40, r=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig_comparison.update_xaxes(showgrid=False)
        fig_comparison.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Modern loss projection chart
        loss_data = pd.DataFrame({
            'Period': ['Month', 'Quarter', 'Year'],
            'Loss': [wasted, wasted * 3, wasted * 12]
        })
        
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Bar(
            x=loss_data['Period'],
            y=loss_data['Loss'],
            marker=dict(
                color=['#FF6B9D', '#FF8A80', '#FFAB91'],
                cornerradius=15,
                line=dict(width=0)
            ),
            text=[f'${val:,.0f}' for val in loss_data['Loss']],
            textposition='outside',
            textfont=dict(size=14, color='#333')
        ))
        
        fig_loss.update_layout(
            title=dict(text="If you do nothing... losses", font_size=18, x=0.5),
            height=350,
            template='plotly_white',
            showlegend=False,
            margin=dict(t=80, b=40, l=40, r=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig_loss.update_xaxes(showgrid=False)
        fig_loss.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        st.plotly_chart(fig_loss, use_container_width=True)
        
        st.markdown(f"Every day you delay ≈ ${daily_waste:.0f} leaks out.")
        
        # Section F - Opportunity Map
        st.markdown("## 🗺️ Opportunity Map (Impact vs Effort)")
        
        if leaks_list:
            impact_effort = []
            for i, leak in enumerate(leaks_list[:5]):
                waste_amount = float(leak['Wasted $'].replace('$', '').replace(',', ''))
                impact = waste_amount / wasted if wasted > 0 else 0
                effort = 0.35 if 'placement' in leak['Leak'] else 0.55
                impact_effort.append({
                    'Leak': leak['Leak'][:20] + '...' if len(leak['Leak']) > 20 else leak['Leak'],
                    'Impact': impact,
                    'Effort': effort
                })
            
            if impact_effort:
                opp_df = pd.DataFrame(impact_effort)
                
                # Modern scatter plot with bubble styling
                fig_scatter = go.Figure()
                
                # Add quadrant background colors
                fig_scatter.add_shape(type="rect", x0=0, y0=0.5, x1=0.5, y1=1, 
                                    fillcolor="rgba(255, 107, 157, 0.1)", line=dict(width=0))
                fig_scatter.add_shape(type="rect", x0=0.5, y0=0.5, x1=1, y1=1, 
                                    fillcolor="rgba(78, 205, 196, 0.1)", line=dict(width=0))
                fig_scatter.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=0.5, 
                                    fillcolor="rgba(200, 200, 200, 0.1)", line=dict(width=0))
                fig_scatter.add_shape(type="rect", x0=0.5, y0=0, x1=1, y1=0.5, 
                                    fillcolor="rgba(255, 171, 145, 0.1)", line=dict(width=0))
                
                # Add quadrant lines
                fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="rgba(128,128,128,0.5)", line_width=2)
                fig_scatter.add_vline(x=0.5, line_dash="dash", line_color="rgba(128,128,128,0.5)", line_width=2)
                
                # Add scatter points
                fig_scatter.add_trace(go.Scatter(
                    x=opp_df['Effort'],
                    y=opp_df['Impact'],
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color='#667eea',
                        line=dict(width=2, color='white'),
                        opacity=0.8
                    ),
                    text=opp_df['Leak'],
                    textposition="top center",
                    textfont=dict(size=12, color='#333'),
                    showlegend=False
                ))
                
                # Add quadrant labels
                fig_scatter.add_annotation(x=0.25, y=0.75, text="Quick Wins<br>(Low Effort, High Impact)", 
                                         showarrow=False, font=dict(size=10, color='#4ECDC4'))
                fig_scatter.add_annotation(x=0.75, y=0.75, text="Major Projects<br>(High Effort, High Impact)", 
                                         showarrow=False, font=dict(size=10, color='#667eea'))
                fig_scatter.add_annotation(x=0.25, y=0.25, text="Fill Ins<br>(Low Effort, Low Impact)", 
                                         showarrow=False, font=dict(size=10, color='#999'))
                fig_scatter.add_annotation(x=0.75, y=0.25, text="Thankless Tasks<br>(High Effort, Low Impact)", 
                                         showarrow=False, font=dict(size=10, color='#FF8A80'))
                
                fig_scatter.update_layout(
                    title=dict(text="Impact vs Effort Matrix", font_size=18, x=0.5),
                    height=500,
                    template='plotly_white',
                    xaxis=dict(title="Effort", range=[0, 1], showgrid=False),
                    yaxis=dict(title="Impact", range=[0, 1], showgrid=False),
                    margin=dict(t=80, b=60, l=60, r=60),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Section G - Benchmarks
        st.markdown("## 📊 Benchmarks")
        
        benchmark_data = pd.DataFrame({
            'Metric': ['CTR %', 'ROAS', 'Frequency'],
            'Your Performance': [f"{metrics['ctr']:.2f}%", f"{metrics['avg_roas']:.2f}", f"{metrics['avg_freq']:.1f}"],
            'Benchmark': ['1.50%', '4.00', '4.0']
        })
        st.dataframe(benchmark_data, use_container_width=True, hide_index=True)
        
        # Section H - CTA
        st.markdown("## 🎯 Next Steps")
        st.markdown("We'll implement fixes, monitor performance, and re-audit monthly.")
        
        if st.button("📞 Book your fix call", type="primary"):
            st.success("Redirecting to booking page... (placeholder)")
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.markdown("Please ensure your CSV has the required columns and proper formatting.")

else:
    st.markdown("## 👋 Welcome to Revenue Leak Report")
    st.markdown("Upload your Meta or Google Ads CSV to identify budget waste and optimization opportunities.")
    st.markdown("### Sample columns we look for:")
    st.markdown("- **Spend**: Amount Spent (USD), Spend")
    st.markdown("- **Performance**: Impressions, Clicks, CTR, Purchases")
    st.markdown("- **Revenue**: Conversion Value (USD), Revenue") 
    st.markdown("- **Targeting**: Campaign, Ad Set, Placement")
    
    st.info("💡 Works with standard Meta Business Manager and Google Ads exports. USD currency only.")
