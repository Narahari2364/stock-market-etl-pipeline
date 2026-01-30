import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text


st.set_page_config(page_title="📈 Stock Market Dashboard (Fast)", layout="wide")


@st.cache_data(ttl=600)
def load_all_data():
    """Load all data from DB once (cached)."""
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            st.error("DATABASE_URL is not set")
            return None
        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT symbol, date, close FROM stock_data ORDER BY date ASC"), conn)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return None


def main():
    st.title("📈 Stock Market Dashboard (Fast)")

    with st.spinner("Loading data..."):
        full_df = load_all_data()

    if full_df is None or full_df.empty:
        st.error("No data available")
        st.stop()

    all_symbols = sorted(full_df["symbol"].dropna().unique().tolist())
    default_symbols = all_symbols[:2]

    st.sidebar.header("Filters")
    selected_symbols = st.sidebar.multiselect("Symbols", all_symbols, default=default_symbols)
    if not selected_symbols:
        st.warning("Select at least one symbol.")
        st.stop()

    df = full_df[full_df["symbol"].isin(selected_symbols)]
    if df.empty:
        st.warning("No rows match your filter.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Symbols", len(selected_symbols))
    with c3:
        st.metric("Date Range", f"{df['date'].min().date()} → {df['date'].max().date()}")

    fig = go.Figure()
    for sym in selected_symbols:
        s = df[df["symbol"] == sym]
        fig.add_trace(
            go.Scattergl(
                x=s["date"].values,
                y=s["close"].values,
                mode="lines",
                name=sym,
            )
        )
    fig.update_layout(template="plotly_white", hovermode="x unified", height=520, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(title="Close ($)")
    fig.update_xaxes(title="Date")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


if __name__ == "__main__":
    main()



