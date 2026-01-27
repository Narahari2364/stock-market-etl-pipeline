"""
Stock Market Dashboard - Streamlit Application

Interactive dashboard for visualizing and analyzing stock market data
extracted from Alpha Vantage API and stored in PostgreSQL.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_data():
    """
    Load ALL stock data from PostgreSQL database.
    This loads everything once, then filtering happens in memory.
    
    Returns:
        DataFrame with stock data or None if error occurs
    """
    try:
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            st.error("❌ DATABASE_URL environment variable is not set")
            return None
        
        # Create database engine
        engine = create_engine(database_url, pool_pre_ping=True)
        
        # Load ALL data from database
        query = text("""
            SELECT 
                symbol, date, open, high, low, close, volume,
                daily_change, daily_change_percent,
                price_range, price_range_percent,
                volatility_indicator, volatility_category,
                volume_category, is_positive_day, is_negative_day,
                ma_5, ma_20, price_vs_ma5, price_vs_ma20,
                year, month, quarter, day_of_week, week_of_year,
                extracted_at, data_source
            FROM stock_data
            ORDER BY symbol, date DESC
        """)
        
        df = pd.read_sql(query, engine)
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date ascending for proper chart display
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        engine.dispose()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data from database: {str(e)}")
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour (symbols don't change often)
def load_symbols():
    """
    Load only unique symbols from PostgreSQL database.
    This is faster than loading all data just to get symbols.
    
    Returns:
        List of unique stock symbols or None if error occurs
    """
    try:
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            return None
        
        # Create database engine
        engine = create_engine(database_url, pool_pre_ping=True)
        
        # Load only unique symbols
        query = text("SELECT DISTINCT symbol FROM stock_data ORDER BY symbol")
        
        result = engine.connect().execute(query)
        symbols = [row[0] for row in result.fetchall()]
        
        engine.dispose()
        
        return symbols
        
    except Exception as e:
        return None


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
    Optimized for speed with vectorized operations.
    
    Args:
        df: Full DataFrame with all data
        selected_symbols: List of selected stock symbols
        date_range: Tuple of (start_date, end_date) or None
    
    Returns:
        Filtered DataFrame
    """
    # Use vectorized operations for faster filtering
    # Filter by symbols first (faster than date filtering)
    mask = df['symbol'].isin(selected_symbols)
    
    # Filter by date range if provided
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        # Convert to datetime for comparison (faster than dt.date)
        date_mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
        mask = mask & date_mask
    
    # Apply combined mask (single pass)
    df_filtered = df[mask].copy()
    
    return df_filtered


def sample_data_for_chart(df, max_points=1000):
    """
    Sample data for chart rendering if dataset is too large.
    Keeps all data for calculations, but reduces chart points for performance.
    
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
    
    # Sample every Nth row
    df_sampled = df.iloc[::sample_rate].copy()
    
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


# Main application
def main():
    """Main dashboard application."""
    
    # Initialize session state for caching filtered data
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = None
    if 'last_symbol_selection' not in st.session_state:
        st.session_state.last_symbol_selection = None
    if 'last_date_range' not in st.session_state:
        st.session_state.last_date_range = None
    if 'load_all_data' not in st.session_state:
        st.session_state.load_all_data = False
    if 'cached_min_date' not in st.session_state:
        st.session_state.cached_min_date = None
    if 'cached_max_date' not in st.session_state:
        st.session_state.cached_max_date = None
    
    # ============================================
    # HEADER
    # ============================================
    st.title("📈 Stock Market Dashboard")
    
    # Load data ONCE from database
    with st.spinner("🔄 Loading data from database..."):
        df = load_data()
    
    if df is None or df.empty:
        st.warning("⚠️ No data found in database. Please run the ETL pipeline first.")
        st.info("💡 Run: `python -m src.pipeline` to populate the database.")
        return
    
    # Get last update time
    if 'extracted_at' in df.columns and not df['extracted_at'].isna().all():
        last_update = pd.to_datetime(df['extracted_at']).max()
        st.markdown(f"*Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}*")
    else:
        st.markdown(f"*Data loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    st.divider()
    
    # ============================================
    # SIDEBAR
    # ============================================
    st.sidebar.header("⚙️ Filters & Controls")
    
    # Load symbols (cached separately for faster loading)
    all_symbols = load_symbols()
    if not all_symbols:
        # Fallback: get symbols from full dataframe if cache miss
        all_symbols = sorted(df['symbol'].unique().tolist())
    
    # Default to 1-2 stocks for faster initial load
    default_symbols = ['AAPL', 'MSFT'] if all(x in all_symbols for x in ['AAPL', 'MSFT']) else all_symbols[:2] if len(all_symbols) >= 2 else all_symbols
    
    # Stock symbol multiselect
    selected_symbols = st.sidebar.multiselect(
        "Select Stock Symbols",
        options=all_symbols,
        default=default_symbols if not st.session_state.load_all_data else all_symbols,
        help="Choose one or more stocks to analyze"
    )
    
    if not selected_symbols:
        st.warning("⚠️ Please select at least one stock symbol.")
        return
    
    # Get date range from filtered data (by symbols only, for date picker)
    # Cache min/max dates for faster response
    symbols_key = tuple(sorted(selected_symbols))
    if (st.session_state.cached_min_date is None or 
        st.session_state.cached_max_date is None or
        st.session_state.last_symbol_selection != symbols_key):
        # Fast vectorized operation to get date range
        mask = df['symbol'].isin(selected_symbols)
        df_symbol_filtered = df[mask]
        st.session_state.cached_min_date = df_symbol_filtered['date'].min().date()
        st.session_state.cached_max_date = df_symbol_filtered['date'].max().date()
    
    min_date = st.session_state.cached_min_date
    max_date = st.session_state.cached_max_date
    
    # Default to last 30 days for faster initial load
    default_start_date = (max_date - timedelta(days=30)) if not st.session_state.load_all_data else min_date
    default_end_date = max_date
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_start_date, default_end_date) if not st.session_state.load_all_data else (min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Select date range for analysis"
    )
    
    # Load All Data button
    if st.sidebar.button("📊 Load All Data", use_container_width=True):
        st.session_state.load_all_data = True
        st.session_state.filtered_data = None
        st.session_state.last_symbol_selection = None
        st.session_state.last_date_range = None
        st.session_state.cached_min_date = None
        st.session_state.cached_max_date = None
        st.rerun()
    
    if st.session_state.load_all_data:
        st.sidebar.success("✓ Showing all data")
    
    # Check if filters have changed
    symbol_selection_changed = (
        st.session_state.last_symbol_selection != tuple(sorted(selected_symbols))
    )
    date_range_changed = (
        st.session_state.last_date_range != date_range
    )
    
    # Only recompute filtered data if filters changed
    # Use optimized filtering for immediate response
    if (symbol_selection_changed or date_range_changed or 
        st.session_state.filtered_data is None):
        
        # Filter data in memory (optimized pandas operation - very fast)
        df_filtered = filter_data(df, selected_symbols, date_range)
        
        # Store in session state
        st.session_state.filtered_data = df_filtered
        st.session_state.last_symbol_selection = tuple(sorted(selected_symbols))
        st.session_state.last_date_range = date_range
    else:
        # Use cached filtered data
        df_filtered = st.session_state.filtered_data
    
    # Advanced charts checkbox
    show_advanced = st.sidebar.checkbox(
        "Show Advanced Charts",
        value=False,
        help="Enable heavy charts like volatility box plots"
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.filtered_data = None
        st.session_state.last_symbol_selection = None
        st.session_state.last_date_range = None
        st.rerun()
    
    st.sidebar.divider()
    
    # Sidebar info
    st.sidebar.info("""
    **Data Source:** Alpha Vantage API
    
    **Database:** PostgreSQL
    
    Use the filters above to customize your analysis.
    """)
    
    # ============================================
    # QUICK STATS SUMMARY
    # ============================================
    st.header("⚡ Quick Stats Summary")
    
    # Calculate quick stats before rendering charts
    total_records = len(df_filtered)
    unique_stocks = df_filtered['symbol'].nunique()
    date_range_str = f"{df_filtered['date'].min().strftime('%Y-%m-%d')} to {df_filtered['date'].max().strftime('%Y-%m-%d')}"
    avg_daily_change = df_filtered['daily_change_percent'].mean()
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("Total Records", f"{total_records:,}")
    with stats_col2:
        st.metric("Unique Stocks", unique_stocks)
    with stats_col3:
        st.metric("Date Range", date_range_str[:20] + "..." if len(date_range_str) > 20 else date_range_str)
    with stats_col4:
        st.metric("Avg Daily Change", format_percent(avg_daily_change))
    
    st.info(f"📈 Showing data for: {', '.join(selected_symbols)} | {total_records:,} records | {date_range_str}")
    
    st.divider()
    
    # ============================================
    # TOP METRICS
    # ============================================
    st.header("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Records",
            f"{len(df_filtered):,}",
            help="Total number of records in selected range"
        )
    
    with col2:
        st.metric(
            "Unique Stocks",
            len(selected_symbols),
            help="Number of selected stock symbols"
        )
    
    with col3:
        st.metric(
            "Date Range",
            date_range_str,
            help="Earliest to latest date in dataset"
        )
    
    with col4:
        avg_daily_change = df_filtered['daily_change_percent'].mean()
        st.metric(
            "Avg Daily Change %",
            format_percent(avg_daily_change),
            delta=format_percent(avg_daily_change),
            help="Average daily percentage change"
        )
    
    st.divider()
    
    # ============================================
    # SECTION 1 - Stock Price Chart
    # ============================================
    st.header("📈 Stock Price Chart")
    
    # Prepare data for chart (sample if large)
    chart_df = df_filtered[['symbol', 'date', 'close', 'ma_5', 'ma_20', 'open', 'high', 'low']].copy()
    chart_df_sampled = sample_data_for_chart(chart_df, max_points=1000)
    
    # Determine if we should use scattergl for better performance
    use_scattergl = should_use_scattergl(chart_df)
    
    # Create Plotly figure
    fig_price = go.Figure()
    
    # Use Scattergl or Scatter based on dataset size
    ScatterClass = go.Scattergl if use_scattergl else go.Scatter
    
    # Add close price lines for each symbol
    for symbol in selected_symbols:
        symbol_data = chart_df_sampled[chart_df_sampled['symbol'] == symbol]
        
        if symbol_data.empty:
            continue
        
        # Close price line
        fig_price.add_trace(ScatterClass(
            x=symbol_data['date'],
            y=symbol_data['close'],
            mode='lines',
            name=f'{symbol} Close',
            line=dict(width=2),
            hovertemplate=(
                f'<b>{symbol}</b><br>' +
                'Date: %{x}<br>' +
                'Close: $%{y:.2f}<br>' +
                '<extra></extra>'
            )
        ))
        
        # MA5 line
        if 'ma_5' in symbol_data.columns and not symbol_data['ma_5'].isna().all():
            fig_price.add_trace(ScatterClass(
                x=symbol_data['date'],
                y=symbol_data['ma_5'],
                mode='lines',
                name=f'{symbol} MA5',
                line=dict(dash='dash', width=1, color=px.colors.qualitative.Set1[selected_symbols.index(symbol) % len(px.colors.qualitative.Set1)]),
                opacity=0.6,
                showlegend=True
            ))
        
        # MA20 line
        if 'ma_20' in symbol_data.columns and not symbol_data['ma_20'].isna().all():
            fig_price.add_trace(ScatterClass(
                x=symbol_data['date'],
                y=symbol_data['ma_20'],
                mode='lines',
                name=f'{symbol} MA20',
                line=dict(dash='dot', width=1, color=px.colors.qualitative.Set2[selected_symbols.index(symbol) % len(px.colors.qualitative.Set2)]),
                opacity=0.6,
                showlegend=True
            ))
    
    # Update layout with optimized settings
    fig_price.update_layout(
        title="Stock Prices with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',  # Faster hover performance
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_price, use_container_width=True)
    
    st.divider()
    
    # ============================================
    # SECTION 2 - Volume Analysis
    # ============================================
    with st.expander("📊 Volume Analysis", expanded=True):
        # Prepare volume data (sample if large)
        volume_df = df_filtered[['symbol', 'date', 'volume']].copy()
        volume_df_sampled = sample_data_for_chart(volume_df, max_points=1000)
        
        # Create bar chart
        fig_volume = px.bar(
            volume_df_sampled,
            x='date',
            y='volume',
            color='symbol',
            title="Trading Volume Over Time",
            labels={'volume': 'Volume (Millions)', 'date': 'Date'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # Format y-axis in millions
        fig_volume.update_layout(
            yaxis=dict(
                tickformat='.0f',
                tickmode='linear',
                title='Volume (Millions)'
            ),
            template='plotly_white',
            height=400,
            hovermode='x unified'  # Faster hover performance
        )
        
        # Convert to millions for display
        fig_volume.update_traces(
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Date: %{x}<br>' +
                          'Volume: %{y:,.0f}<br>' +
                          '<extra></extra>'
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
    
    st.divider()
    
    # ============================================
    # SECTION 3 - Daily Returns
    # ============================================
    with st.expander("📉 Daily Returns", expanded=True):
        # Prepare returns data (sample if large)
        returns_df = df_filtered[['symbol', 'date', 'daily_change_percent']].copy()
        returns_df_sampled = sample_data_for_chart(returns_df, max_points=1000)
        
        # Determine if we should use scattergl
        use_scattergl_returns = should_use_scattergl(returns_df)
        ScatterClassReturns = go.Scattergl if use_scattergl_returns else go.Scatter
        
        # Create line chart
        fig_returns = go.Figure()
        
        for symbol in selected_symbols:
            symbol_data = returns_df_sampled[returns_df_sampled['symbol'] == symbol]
            
            if symbol_data.empty:
                continue
            
            fig_returns.add_trace(ScatterClassReturns(
                x=symbol_data['date'],
                y=symbol_data['daily_change_percent'],
                mode='lines+markers',
                name=symbol,
                line=dict(width=1.5),
                marker=dict(size=3),  # Reduced marker size for better performance
                hovertemplate=(
                    f'<b>{symbol}</b><br>' +
                    'Date: %{x}<br>' +
                    'Daily Return: %{y:.2f}%<br>' +
                    '<extra></extra>'
                )
            ))
        
        # Add horizontal line at y=0
        fig_returns.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Zero Line",
            annotation_position="right"
        )
        
        # Update layout with optimized settings
        fig_returns.update_layout(
            title="Daily Percentage Returns",
            xaxis_title="Date",
            yaxis_title="Daily Return (%)",
            hovermode='x unified',  # Faster hover performance
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
    
    st.divider()
    
    # ============================================
    # SECTION 4 - Performance Table
    # ============================================
    st.header("💼 Performance Summary")
    
    # Calculate metrics for each symbol
    performance_data = []
    
    for symbol in selected_symbols:
        symbol_data = df_filtered[df_filtered['symbol'] == symbol].sort_values('date')
        
        if symbol_data.empty:
            continue
        
        # Calculate metrics
        current_price = symbol_data['close'].iloc[-1]
        avg_price = symbol_data['close'].mean()
        min_price = symbol_data['close'].min()
        max_price = symbol_data['close'].max()
        
        # Total return (first to last)
        first_price = symbol_data['close'].iloc[0]
        total_return = ((current_price - first_price) / first_price) * 100 if first_price > 0 else 0
        
        # Average daily change
        avg_daily_change = symbol_data['daily_change_percent'].mean()
        
        # Total volume
        total_volume = symbol_data['volume'].sum()
        
        performance_data.append({
            'Symbol': symbol,
            'Current Price': current_price,
            'Average Price': avg_price,
            'Min Price': min_price,
            'Max Price': max_price,
            'Total Return %': total_return,
            'Avg Daily Change %': avg_daily_change,
            'Total Volume': total_volume
        })
    
    # Create DataFrame
    perf_df = pd.DataFrame(performance_data)
    
    # Format columns
    if not perf_df.empty:
        # Format numeric columns
        perf_df['Current Price'] = perf_df['Current Price'].apply(lambda x: format_currency(x))
        perf_df['Average Price'] = perf_df['Average Price'].apply(lambda x: format_currency(x))
        perf_df['Min Price'] = perf_df['Min Price'].apply(lambda x: format_currency(x))
        perf_df['Max Price'] = perf_df['Max Price'].apply(lambda x: format_currency(x))
        perf_df['Total Return %'] = perf_df['Total Return %'].apply(lambda x: format_percent(x))
        perf_df['Avg Daily Change %'] = perf_df['Avg Daily Change %'].apply(lambda x: format_percent(x))
        perf_df['Total Volume'] = perf_df['Total Volume'].apply(lambda x: format_number(x, 0))
        
        # Display as dataframe
        st.dataframe(
            perf_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No performance data available.")
    
    st.divider()
    
    # ============================================
    # SECTION 5 - Volatility Box Plot (Advanced)
    # ============================================
    if show_advanced:
        st.header("📊 Price Distribution (Box Plot)")
        
        # Prepare data for box plot
        box_df = df_filtered[['symbol', 'close']].copy()
        
        # Create box plot
        fig_box = px.box(
            box_df,
            x='symbol',
            y='close',
            color='symbol',
            title="Price Distribution by Symbol",
            labels={'close': 'Close Price ($)', 'symbol': 'Symbol'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig_box.update_layout(
            template='plotly_white',
            height=400,
            showlegend=False,
            hovermode='x unified'  # Faster hover performance
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.divider()
    
    # ============================================
    # FOOTER
    # ============================================
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("**Data Source**")
        st.markdown("Alpha Vantage API")
    
    with footer_col2:
        st.markdown("**Database**")
        st.markdown("PostgreSQL")
    
    with footer_col3:
        st.markdown("**GitHub**")
        st.markdown("[View Repository](#)")
    
    st.markdown(
        "<div style='text-align: center; color: gray; padding: 20px;'>"
        "Powered by Alpha Vantage API | Stock Market ETL Pipeline"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

