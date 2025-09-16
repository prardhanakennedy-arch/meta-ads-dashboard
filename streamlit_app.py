# Gradient Spend vs Revenue Line
import plotly.graph_objects as go

daily = w.groupby("date").agg(
    spend=("spend","sum"),
    revenue=("revenue","sum")
).reset_index()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=daily["date"], y=daily["spend"],
    mode="lines", name="Spend",
    line=dict(width=4, color="rgba(59,130,246,1)"),
    fill="tozeroy",
    fillcolor="rgba(59,130,246,0.2)"
))
fig.add_trace(go.Scatter(
    x=daily["date"], y=daily["revenue"],
    mode="lines", name="Revenue",
    line=dict(width=4, color="rgba(34,197,94,1)"),
    fill="tozeroy",
    fillcolor="rgba(34,197,94,0.2)"
))

fig.update_layout(
    height=320, margin=dict(l=10,r=10,t=10,b=10),
    xaxis_title="Date", yaxis_title="USD",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
