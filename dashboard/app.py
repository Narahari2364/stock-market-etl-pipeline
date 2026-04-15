"""
Stock Market Dashboard - Streamlit Application

Interactive dashboard for visualizing and analyzing stock market data
extracted from Alpha Vantage API and stored in PostgreSQL.
"""

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get DATABASE_URL from Streamlit secrets or environment
try:
    DATABASE_URL = st.secrets.get("DATABASE_URL", os.getenv("DATABASE_URL"))
except Exception:
    DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    st.error("❌ DATABASE_URL environment variable is not set")
    st.info("Please configure DATABASE_URL in Streamlit Cloud Settings → Secrets")
    st.stop()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import time
import sys
sys.path.append('src')
from predictions import generate_predictions_for_all, get_trading_signals, get_top_predictions

# Page configuration
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


def generate_sample_data():
    """Generate sample data when database is unavailable"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    # Generate realistic dates
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=100, freq="D")
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

    data = []
    base_prices = {"AAPL": 180, "MSFT": 380, "GOOGL": 140, "TSLA": 250, "NVDA": 500}

    for symbol in symbols:
        base = base_prices[symbol]
        for date in dates:
            price = base + np.random.randn() * 10
            data.append(
                {
                    "symbol": symbol,
                    "date": date,  # Timestamp / datetime-like from date_range
                    "open": abs(price),
                    "high": abs(price + np.random.randn() * 5),
                    "low": abs(price - np.random.randn() * 5),
                    "close": abs(price + np.random.randn() * 3),
                    "volume": int(abs(50000000 + np.random.randn() * 10000000)),
                    "daily_change": np.random.randn() * 2,
                    "daily_change_percent": np.random.randn() * 1.5,
                    "ma_5": abs(price),
                    "ma_20": abs(price - 2),
                    "price_range": abs(np.random.randn() * 5),
                    "volatility_indicator": abs(np.random.randn() * 3),
                    "volatility_category": np.random.choice(["Low", "Medium", "High"]),
                    "volume_category": np.random.choice(["Low", "Medium", "High"]),
                    "is_positive_day": np.random.choice([True, False]),
                    "year": int(date.year),
                    "month": int(date.month),
                    "quarter": int(date.quarter),
                    "day_of_week": date.day_name(),
                    "price_vs_ma5": np.random.randn() * 2,
                    "price_vs_ma20": np.random.randn() * 3,
                }
            )

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])  # Ensure datetime type
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


@st.cache_data(ttl=600)
def load_data():
    """Load data with retry logic for sleeping databases."""
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            engine = create_engine(DATABASE_URL, pool_pre_ping=True)
            with engine.connect() as conn:
                query = text(
                    "SELECT * FROM stock_data ORDER BY date DESC LIMIT 2000"
                )
                df = pd.read_sql(query, conn)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])

            if "symbol" in df.columns and "date" in df.columns:
                df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

            return df, True
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return generate_sample_data(), False

    return pd.DataFrame(), False


def format_number(num, decimals=2):
    """Format number with commas and decimals."""
    if pd.isna(num):
        return "N/A"
    return f"{num:,.{decimals}f}"


def format_currency(num):
    """Format number as currency."""
    if pd.isna(num):
        return "N/A"
    return f"${num:,.2f}"


def format_percent(num):
    """Format number as percentage."""
    if pd.isna(num):
        return "N/A"
    return f"{num:.2f}%"


def filter_data(df, selected_symbols, date_range):
    """
    Filter DataFrame in memory based on selected symbols and date range.
    Ultra-optimized for maximum speed - avoids unnecessary copies.
    
    Args:
        df: Full DataFrame with all data
        selected_symbols: List of selected stock symbols
        date_range: Tuple of (start_date, end_date) or None
    
    Returns:
        Filtered DataFrame view (no copy unless necessary)
    """
    # Ultra-fast vectorized filtering - single pass, no intermediate copies
    # Convert symbols to set for O(1) lookup instead of O(n)
    symbol_set = set(selected_symbols)
    
    # Fast symbol filtering using vectorized isin (already optimized)
    mask = df['symbol'].isin(symbol_set)
    
    # Fast date filtering if needed
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        # Use numpy datetime64 for fastest comparison
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        date_mask = (df['date'].values >= start_ts) & (df['date'].values <= end_ts)
        mask = mask & date_mask
    
    # Return view if possible, copy only if needed for safety
    # Using .loc with boolean mask is fastest
    return df.loc[mask]


def sample_data_for_chart(df, max_points=1000):
    """
    Sample data for chart rendering if dataset is too large.
    Keeps all data for calculations, but reduces chart points for performance.
    Cached for faster repeated access.
    
    Args:
        df: DataFrame to sample
        max_points: Maximum number of points to show in chart
    
    Returns:
        Sampled DataFrame (or original if small enough)
    """
    if len(df) <= max_points:
        return df
    
    # Calculate sampling rate
    sample_rate = len(df) // max_points
    
    # Sample every Nth row (use iloc for speed, no copy needed)
    df_sampled = df.iloc[::sample_rate]
    
    return df_sampled


def should_use_scattergl(df):
    """
    Determine if scattergl should be used instead of scatter for better performance.
    
    Args:
        df: DataFrame to check
    
    Returns:
        True if scattergl should be used (for large datasets)
    """
    return len(df) > 500


def prepare_chart_data(df_filtered, selected_symbols):
    """
    Prepare chart data efficiently.
    Uses views instead of copies when possible.
    
    Args:
        df_filtered: Filtered DataFrame
        selected_symbols: List of selected symbols
    
    Returns:
        Dictionary with prepared chart dataframes
    """
    # Extract only needed columns (use view, no copy)
    chart_df = df_filtered[['symbol', 'date', 'close', 'ma_5', 'ma_20']]
    chart_df_sampled = sample_data_for_chart(chart_df, max_points=500)
    
    volume_df = df_filtered[['symbol', 'date', 'volume']]
    volume_df_sampled = sample_data_for_chart(volume_df, max_points=500)
    
    returns_df = df_filtered[['symbol', 'date', 'daily_change_percent']]
    returns_df_sampled = sample_data_for_chart(returns_df, max_points=500)
    
    return {
        'price': chart_df_sampled,
        'volume': volume_df_sampled,
        'returns': returns_df_sampled
    }


# Main application
def main():
    """Main dashboard application."""

    # ============================================
    # HEADER
    # ============================================
    st.title("📈 Stock Market Dashboard")

    # Load data ONCE from database (cached, with retries)
    with st.spinner("Loading data from database..."):
        full_df, db_connected = load_data()

    if db_connected:
        st.success("📊 Connected to database")
    else:
        st.error("❌ Database connection failed after 3 attempts")
        st.info("💡 The free database may be sleeping. Wait 30 seconds and refresh.")
        st.warning("Showing sample data so the dashboard remains usable.")

    if full_df is None or full_df.empty:
        st.error("No data available")
        st.stop()
    
    # Get last update time
    if 'extracted_at' in full_df.columns and not full_df['extracted_at'].isna().all():
        last_update = pd.to_datetime(full_df['extracted_at']).max()
        st.markdown(f"*Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}*")
    else:
        st.markdown(f"*Data loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    st.divider()
    
    # ============================================
    # SIDEBAR
    # ============================================
    st.sidebar.header("⚙️ Filters & Controls")

    # Get unique symbols from the loaded DataFrame (no DB query)
    all_symbols = sorted(full_df["symbol"].dropna().unique().tolist())

    # Default symbols: first 2 stocks only (fast initial load)
    default_symbols = all_symbols[:2]

    # Initialize session state
    if "filtered_df" not in st.session_state:
        st.session_state.filtered_df = None
    if "filters_key" not in st.session_state:
        st.session_state.filters_key = None
    if "reset_filters" not in st.session_state:
        st.session_state.reset_filters = False

    # Default date range: last 30 days only (fast initial load)
    min_date = full_df["date"].min().date()
    max_date = full_df["date"].max().date()
    max_ts = full_df["date"].max()
    default_end_date = max_date
    default_start_date = (max_ts - timedelta(days=30)).date()
    if default_start_date < min_date:
        default_start_date = min_date

    # Initialize widget defaults in session state if not set or if reset was clicked
    if "symbols_multiselect" not in st.session_state or st.session_state.reset_filters:
        st.session_state.symbols_multiselect = default_symbols
    if "date_range_input" not in st.session_state or st.session_state.reset_filters:
        st.session_state.date_range_input = (default_start_date, default_end_date)
    
    # Reset the reset flag after using it
    if st.session_state.reset_filters:
        st.session_state.reset_filters = False

    # Use a form so changing widgets doesn't re-render charts until Apply is clicked
    with st.sidebar.form("filters_form", clear_on_submit=False):
        selected_symbols = st.multiselect(
            "Select Stock Symbols",
            options=all_symbols,
            default=st.session_state.symbols_multiselect,
            help="Choose one or more stocks to analyze",
            key="symbols_multiselect",
        )

        if not selected_symbols:
            st.warning("⚠️ Please select at least one stock symbol.")

        date_range = st.date_input(
            "Date Range",
            value=st.session_state.date_range_input,
            min_value=min_date,
            max_value=max_date,
            help="Select date range for analysis",
            key="date_range_input",
        )

        c_apply, c_reset = st.columns(2)
        submitted = c_apply.form_submit_button("✅ Apply Filters", type="primary")
        reset_clicked = c_reset.form_submit_button("🔁 Reset Filters")

    # Handle reset button
    if reset_clicked:
        # Reset to defaults (first 2 stocks, last 30 days)
        st.session_state.reset_filters = True
        st.session_state.filtered_df = None
        st.session_state.filters_key = None
        st.rerun()

    # Compute filtered df only when Apply is clicked, otherwise reuse last
    filters_key = (
        tuple(sorted(selected_symbols)) if selected_symbols else tuple(),
        tuple(date_range) if isinstance(date_range, tuple) else (date_range,),
    )

    if submitted or st.session_state.filtered_df is None or st.session_state.filters_key != filters_key:
        if not selected_symbols:
            st.stop()

        # Filter in memory from full_df (no DB queries)
        filtered_df = full_df[full_df["symbol"].isin(selected_symbols)].copy()

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_dt = pd.to_datetime(date_range[0])
            end_dt = pd.to_datetime(date_range[1])
            filtered_df = filtered_df[(filtered_df["date"] >= start_dt) & (filtered_df["date"] <= end_dt)]

        # Persist
        st.session_state.filtered_df = filtered_df
        st.session_state.filters_key = filters_key

    df_filtered = st.session_state.filtered_df

    if df_filtered is None or df_filtered.empty:
        st.warning("⚠️ No data matches the selected filters.")
        st.stop()
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.filtered_df = None
        st.session_state.filters_key = None
        st.rerun()
    
    st.sidebar.divider()
    
    # Sidebar info
    st.sidebar.info("""
    **Data Source:** Alpha Vantage API
    
    **Database:** PostgreSQL
    
    Use the filters above to customize your analysis.
    """)
    
    # ============================================
    # TOP METRICS ROW
    # ============================================
    total_records = len(df_filtered)
    unique_symbols = df_filtered["symbol"].nunique()
    date_range_str = f"{df_filtered['date'].min().date()} → {df_filtered['date'].max().date()}"
    avg_daily_change = df_filtered["daily_change_percent"].mean() if "daily_change_percent" in df_filtered.columns else None

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("📊 Total Records", f"{total_records:,}")
    with m2:
        st.metric("🏢 Unique Symbols", f"{unique_symbols}")
    with m3:
        st.metric("📅 Date Range", date_range_str)
    with m4:
        st.metric(
            "📈 Avg Daily Change %",
            format_percent(avg_daily_change),
            delta=format_percent(avg_daily_change) if avg_daily_change is not None else None,
        )
    
    st.divider()
    
    # ============================================
    # A) Stock Price with Moving Averages
    # ============================================
    st.header("📈 Stock Price Trends with Moving Averages")
    
    # Prepare chart data (cached)
    chart_data = prepare_chart_data(df_filtered, selected_symbols)
    chart_df_sampled = chart_data['price']
    
    # Determine if we should use scattergl for better performance
    # Always use Scattergl for best performance on time series
    fig_price = go.Figure()
    ScatterClass = go.Scattergl
    
    # Pre-compute color indices for faster access
    color_indices = {sym: i for i, sym in enumerate(selected_symbols)}
    
    # Add close price lines for each symbol (optimized loop)
    for symbol in selected_symbols:
        symbol_mask = chart_df_sampled['symbol'] == symbol
        symbol_data = chart_df_sampled[symbol_mask]
        
        if len(symbol_data) == 0:
            continue
        
        # Get color index once
        idx = color_indices[symbol]
        
        base_color = px.colors.qualitative.Set1[idx % len(px.colors.qualitative.Set1)]

        # Close price line
        fig_price.add_trace(ScatterClass(
            x=symbol_data['date'].values,  # Use .values for faster access
            y=symbol_data['close'].values,
            mode='lines',
            name=f'{symbol} Close',
            line=dict(width=2, color=base_color),
            hovertemplate=(
                f"<b>{symbol}</b><br>"
                "Date: %{x}<br>"
                "Close: $%{y:.2f}<br>"
                "<extra></extra>"
            ),
        ))

        # MA5 / MA20 lines (always shown if available)
        if "ma_5" in symbol_data.columns and not symbol_data["ma_5"].isna().all():
            fig_price.add_trace(ScatterClass(
                x=symbol_data['date'].values,
                y=symbol_data['ma_5'].values,
                mode='lines',
                name=f'{symbol} MA5',
                line=dict(dash='dash', width=1, color=base_color),
                opacity=0.7,
                hovertemplate=(
                    f"<b>{symbol}</b><br>"
                    "Date: %{x}<br>"
                    "MA5: $%{y:.2f}<br>"
                    "<extra></extra>"
                ),
            ))

        if "ma_20" in symbol_data.columns and not symbol_data["ma_20"].isna().all():
            fig_price.add_trace(ScatterClass(
                x=symbol_data['date'].values,
                y=symbol_data['ma_20'].values,
                mode='lines',
                name=f'{symbol} MA20',
                line=dict(dash='dash', width=1, color=base_color),
                opacity=0.5,
                hovertemplate=(
                    f"<b>{symbol}</b><br>"
                    "Date: %{x}<br>"
                    "MA20: $%{y:.2f}<br>"
                    "<extra></extra>"
                ),
            ))
    
    # Update layout with optimized settings
    fig_price.update_layout(
        title="📈 Stock Price Trends with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white',
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    
    st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})
    
    st.divider()
    
    # ============================================
    # B) Trading Volume Analysis
    # ============================================
    st.header("📊 Trading Volume")

    volume_df_sampled = chart_data["volume"].copy()
    if "volume" in volume_df_sampled.columns:
        volume_df_sampled["volume_m"] = volume_df_sampled["volume"] / 1_000_000.0
    else:
        volume_df_sampled["volume_m"] = None

    fig_volume = px.bar(
        volume_df_sampled,
        x="date",
        y="volume_m",
        color="symbol",
        barmode="group",
        title="📊 Trading Volume",
        labels={"volume_m": "Volume (Millions)", "date": "Date", "symbol": "Symbol"},
        template="plotly_white",
    )
    fig_volume.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig_volume, use_container_width=True, config={"displayModeBar": False})
    
    st.divider()
    
    # ============================================
    # C) Daily Returns
    # ============================================
    st.header("💹 Daily Price Changes (%)")

    returns_df_sampled = chart_data["returns"]
    fig_returns = go.Figure()

    for symbol in selected_symbols:
        s = returns_df_sampled[returns_df_sampled["symbol"] == symbol]
        if s.empty:
            continue

        x = s["date"].values
        y = s["daily_change_percent"].values

        # Base line
        fig_returns.add_trace(go.Scattergl(
            x=x,
            y=y,
            mode="lines",
            name=symbol,
            line=dict(width=1.5),
            hovertemplate=f"<b>{symbol}</b><br>Date: %{{x}}<br>Change: %{{y:.2f}}%<extra></extra>",
        ))

        # Filled positive/negative areas
        y_pos = y.copy()
        y_pos[y_pos < 0] = None
        y_neg = y.copy()
        y_neg[y_neg >= 0] = None

        fig_returns.add_trace(go.Scattergl(
            x=x,
            y=y_pos,
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(0,180,0,0.15)",
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_returns.add_trace(go.Scattergl(
            x=x,
            y=y_neg,
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(220,0,0,0.15)",
            showlegend=False,
            hoverinfo="skip",
        ))

    fig_returns.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_returns.update_layout(
        title="💹 Daily Price Changes (%)",
        xaxis_title="Date",
        yaxis_title="Daily Change (%)",
        hovermode="x unified",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_returns, use_container_width=True, config={"displayModeBar": False})
    
    st.divider()
    
    # ============================================
    # D) Performance Summary Table
    # ============================================
    st.header("📋 Performance Metrics")

    perf_rows = []
    for symbol in selected_symbols:
        s = df_filtered[df_filtered["symbol"] == symbol].sort_values("date")
        if s.empty:
            continue

        latest_price = s["close"].iloc[-1]
        period_avg = s["close"].mean()
        min_price = s["close"].min()
        max_price = s["close"].max()
        first_price = s["close"].iloc[0]
        total_return = ((latest_price - first_price) / first_price) * 100 if first_price else None
        avg_dc = s["daily_change_percent"].mean() if "daily_change_percent" in s.columns else None
        total_vol_m = (s["volume"].sum() / 1_000_000.0) if "volume" in s.columns else None

        perf_rows.append({
            "Symbol": symbol,
            "Latest Price": latest_price,
            "Period Average": period_avg,
            "Min Price": min_price,
            "Max Price": max_price,
            "Total Return %": total_return,
            "Avg Daily Change %": avg_dc,
            "Total Volume (M)": total_vol_m,
        })

    perf_df = pd.DataFrame(perf_rows)
    if perf_df.empty:
        st.info("No performance data available.")
    else:
        display_df = perf_df.copy()
        display_df["Latest Price"] = display_df["Latest Price"].apply(format_currency)
        display_df["Period Average"] = display_df["Period Average"].apply(format_currency)
        display_df["Min Price"] = display_df["Min Price"].apply(format_currency)
        display_df["Max Price"] = display_df["Max Price"].apply(format_currency)
        display_df["Total Return %"] = display_df["Total Return %"].apply(format_percent)
        display_df["Avg Daily Change %"] = display_df["Avg Daily Change %"].apply(format_percent)
        display_df["Total Volume (M)"] = display_df["Total Volume (M)"].apply(lambda x: f"{x:.2f}M" if pd.notna(x) else "N/A")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # ============================================
    # ML Predictions Section
    # ============================================
    st.header("🤖 ML Price Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Top Predicted Gainers")
        try:
            gainers = get_top_predictions(df_filtered, top_n=5, prediction_type='gainers')
            if not gainers.empty:
                # Format for display
                gainers_display = gainers[['symbol', 'current_price', 'predicted_price',
                                           'predicted_change_percent', 'confidence', 'trend']].copy()

                # Format numeric columns
                gainers_display['current_price'] = gainers_display['current_price'].apply(lambda x: f"${x:.2f}")
                gainers_display['predicted_price'] = gainers_display['predicted_price'].apply(lambda x: f"${x:.2f}")
                gainers_display['predicted_change_percent'] = gainers_display['predicted_change_percent'].apply(
                    lambda x: f"🟢 +{x:.2f}%" if x > 0 else f"{x:.2f}%"
                )

                # Rename columns for display
                gainers_display.columns = ['Symbol', 'Current Price', 'Predicted Price', 'Change %', 'Confidence', 'Trend']

                st.dataframe(
                    gainers_display,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Not enough data for predictions")
        except Exception as e:
            st.error(f"Error: {e}")

    with col2:
        st.subheader("📉 Top Predicted Losers")
        try:
            losers = get_top_predictions(df_filtered, top_n=5, prediction_type='losers')
            if not losers.empty:
                # Format for display
                losers_display = losers[['symbol', 'current_price', 'predicted_price',
                                         'predicted_change_percent', 'confidence', 'trend']].copy()

                # Format numeric columns
                losers_display['current_price'] = losers_display['current_price'].apply(lambda x: f"${x:.2f}")
                losers_display['predicted_price'] = losers_display['predicted_price'].apply(lambda x: f"${x:.2f}")
                losers_display['predicted_change_percent'] = losers_display['predicted_change_percent'].apply(
                    lambda x: f"🔴 {x:.2f}%"
                )

                # Rename columns for display
                losers_display.columns = ['Symbol', 'Current Price', 'Predicted Price', 'Change %', 'Confidence', 'Trend']

                st.dataframe(
                    losers_display,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Not enough data for predictions")
        except Exception as e:
            st.error(f"Error: {e}")

    # Trading Signals
    st.subheader("🎯 Trading Signals (MA Crossover)")
    try:
        signals = get_trading_signals(df_filtered)
        if not signals.empty:
            signals_display = signals[['symbol', 'signal', 'signal_type', 'date', 'price', 'days_ago']].copy()

            # Format columns
            signals_display['price'] = signals_display['price'].apply(lambda x: f"${x:.2f}")
            signals_display['signal'] = signals_display['signal'].apply(
                lambda x: f"🟢 {x}" if x == 'BUY' else f"🔴 {x}"
            )

            # Rename columns
            signals_display.columns = ['Symbol', 'Signal', 'Type', 'Date', 'Price', 'Days Ago']

            st.dataframe(
                signals_display,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No recent MA crossover signals detected. Check back after market movements!")
    except Exception as e:
        st.error(f"Error: {e}")

    # Prediction Details Expander
    with st.expander("ℹ️ How Predictions Work"):
        st.markdown("""
        ### Prediction Methodology

        Our ML predictions use **Moving Average Crossover Strategy**:

        **Bullish Trend (Price Increase Expected):**
        - MA5 (5-day moving average) > MA20 (20-day moving average)
        - Recent positive momentum

        **Bearish Trend (Price Decrease Expected):**
        - MA5 < MA20
        - Recent negative momentum

        **Trading Signals:**
        - **Golden Cross (BUY):** MA5 crosses above MA20
        - **Death Cross (SELL):** MA5 crosses below MA20

        **Confidence Levels:**
        - **HIGH:** Strong trend with low volatility
        - **MEDIUM:** Moderate trend with moderate volatility
        - **LOW:** Weak trend or high volatility

        ⚠️ **Disclaimer:** These are educational predictions based on historical data.
        Not financial advice. Always do your own research before investing.
        """)

    st.divider()

    # ============================================
    # E) Price Distribution (Volatility)
    # ============================================
    with st.expander("📦 Price Distribution & Volatility", expanded=False):
        box_df = df_filtered[["symbol", "close"]].copy()
        fig_box = px.box(
            box_df,
            x="symbol",
            y="close",
            color="symbol",
            title="📦 Price Distribution & Volatility",
            labels={"close": "Close Price ($)", "symbol": "Symbol"},
            template="plotly_white",
        )
        fig_box.update_layout(height=420, hovermode="x unified", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ============================================
    # FOOTER
    # ============================================
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        st.caption("Data source: Alpha Vantage API")
    with f2:
        if 'extracted_at' in full_df.columns and not full_df['extracted_at'].isna().all():
            st.caption(f"Last updated: {pd.to_datetime(full_df['extracted_at']).max().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with f3:
        if st.button("🔄 Refresh Data", key="refresh_data_button", use_container_width=True):
            st.cache_data.clear()
            st.session_state.filtered_df = None
            st.session_state.filters_key = None
            st.rerun()


if __name__ == "__main__":
    main()

