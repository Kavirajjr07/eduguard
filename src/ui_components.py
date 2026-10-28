from __future__ import annotations

from pathlib import Path

import plotly.express as px
import streamlit as st


def ensure_model_available() -> bool:
    model_path = Path(__file__).resolve().parents[1] / "models" / "dropout_model.joblib"
    if not model_path.exists():
        st.warning("The trained model is missing. Run `python -m src.train_model` before using prediction features.")
        return False
    return True


def render_metric_card(label: str, value: str | int | float, delta: str | None = None) -> None:
    with st.container():
        st.markdown(
            f"""
            <div style="padding: 1rem; border-radius: 12px; background: linear-gradient(135deg, #eff9ff, #dff3ff); border: 1px solid #dbeafe; margin-bottom: 0.75rem;">
                <div style="font-size: 0.82rem; color: #3b82f6;">{label}</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #0f172a; margin-top: 0.28rem;">{value}</div>
                {f'<div style="font-size: 0.72rem; color: #0f766e; margin-top: 0.2rem;">{delta}</div>' if delta else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_demo_notice() -> None:
    st.info("This application uses synthetic student data for demonstration. Predictions are estimates and not evidence of real-world outcomes.")


def risk_color(probability: float) -> str:
    if probability < 0.30:
        return "#22c55e"
    if probability < 0.60:
        return "#f59e0b"
    return "#ef4444"


def build_risk_chart(df):
    count_df = df.groupby("risk_level").size().reset_index(name="count")
    fig = px.bar(count_df, x="risk_level", y="count", color="risk_level", color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"})
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=20, b=10),
        height=280,
    )
    return fig
