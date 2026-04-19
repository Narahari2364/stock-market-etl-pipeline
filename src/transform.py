"""
Transform module for cleaning and transforming raw Alpha Vantage stock market data.

This module provides functions to transform raw stock data into a clean,
analysis-ready pandas DataFrame with derived features and data quality checks.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


def categorize_volatility(volatility_percent: float) -> str:
    """
    Categorize volatility into 5 levels based on percentage.
    
    Args:
        volatility_percent: Volatility as a percentage (e.g., 2.5 for 2.5%)
    
    Returns:
        Volatility category: 'Very Low', 'Low', 'Medium', 'High', 'Very High'
    """
    if pd.isna(volatility_percent):
        return 'Unknown'
    
    if volatility_percent < 1.0:
        return 'Very Low'
    elif volatility_percent < 2.0:
        return 'Low'
    elif volatility_percent < 3.5:
        return 'Medium'
    elif volatility_percent < 5.0:
        return 'High'
    else:
        return 'Very High'


def transform_stock_data(raw_stock_data_list: List[Dict]) -> pd.DataFrame:
    """
    Transform raw Alpha Vantage stock data into a clean, feature-rich DataFrame.
    
    This function:
    - Converts list of raw stock dictionaries to pandas DataFrame
    - Extracts OHLC data from time series
    - Adds derived columns (daily changes, date components, indicators)
    - Calculates moving averages per symbol
    - Performs data quality checks
    - Returns clean DataFrame with proper column ordering
    
    Args:
        raw_stock_data_list: List of dictionaries from fetch_stock_data() or fetch_multiple_stocks()
                            Each dict should have 'symbol' and 'time_series' keys
    
    Returns:
        Clean pandas DataFrame with columns:
        - symbol, date, open, high, low, close, volume
        - daily_change, daily_change_percent
        - price_range, price_range_percent
        - year, month, quarter, day_of_week, week_of_year
        - is_positive_day, is_negative_day
        - volume_category
        - volatility_indicator, volatility_category
        - ma_5, ma_20
        - price_vs_ma5, price_vs_ma20
    
    Raises:
        ValueError: If input is empty or invalid
        KeyError: If required keys are missing from input data
    """
    print(f"\n{'='*60}")
    print("Starting Stock Data Transformation")
    print(f"{'='*60}")
    
    if not raw_stock_data_list:
        error_msg = "raw_stock_data_list cannot be empty"
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    # Filter out None values (failed fetches)
    valid_data = [data for data in raw_stock_data_list if data is not None]
    
    if not valid_data:
        error_msg = "No valid stock data found in input list"
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    print(f"Processing {len(valid_data)} stock symbols...")
    
    all_dataframes = []
    
    for stock_data in valid_data:
        symbol = stock_data.get('symbol', 'UNKNOWN')
        time_series = stock_data.get('time_series', {})
        
        if not time_series:
            print(f"⚠ WARNING: No time series data for {symbol}, skipping...")
            continue
        
        print(f"  Processing {symbol}...")
        
        try:
            # Convert time series dictionary to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns from Alpha Vantage format
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            
            # Only rename columns that exist
            existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
            df.rename(columns=existing_mapping, inplace=True)
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            
            # Reset index to make date a column
            df = df.reset_index()
            
            # Convert OHLCV columns to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            all_dataframes.append(df)
            print(f"    ✓ {symbol}: {len(df)} rows extracted")
            
        except Exception as e:
            print(f"    ✗ Error processing {symbol}: {str(e)}")
            continue
    
    if not all_dataframes:
        error_msg = "No dataframes were successfully created"
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    # Concatenate all dataframes
    print(f"\nCombining data from {len(all_dataframes)} symbols...")
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    print(f"  Total rows before transformation: {len(df_combined)}")
    
    # ============================================
    # DATA QUALITY CHECKS
    # ============================================
    print(f"\nPerforming data quality checks...")
    initial_rows = len(df_combined)
    
    # Remove rows with missing critical values
    critical_columns = ['open', 'high', 'low', 'close', 'volume', 'date', 'symbol']
    missing_before = df_combined[critical_columns].isnull().sum().sum()
    df_combined = df_combined.dropna(subset=critical_columns)
    print(f"  Removed rows with missing values: {initial_rows - len(df_combined)}")
    
    # Check logical consistency: high >= low, high >= open, high >= close, low <= open, low <= close
    invalid_logic = (
        (df_combined['high'] < df_combined['low']) |
        (df_combined['high'] < df_combined['open']) |
        (df_combined['high'] < df_combined['close']) |
        (df_combined['low'] > df_combined['open']) |
        (df_combined['low'] > df_combined['close'])
    )
    invalid_count = invalid_logic.sum()
    if invalid_count > 0:
        print(f"  ⚠ WARNING: Found {invalid_count} rows with invalid price logic, removing...")
        df_combined = df_combined[~invalid_logic]
    
    # Remove invalid prices (negative or zero)
    invalid_prices = (
        (df_combined['open'] <= 0) |
        (df_combined['high'] <= 0) |
        (df_combined['low'] <= 0) |
        (df_combined['close'] <= 0) |
        (df_combined['volume'] < 0)
    )
    invalid_price_count = invalid_prices.sum()
    if invalid_price_count > 0:
        print(f"  Removed rows with invalid prices: {invalid_price_count}")
        df_combined = df_combined[~invalid_prices]
    
    # Remove duplicates (same symbol and date)
    duplicates_before = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['symbol', 'date'], keep='last')
    duplicates_removed = duplicates_before - len(df_combined)
    if duplicates_removed > 0:
        print(f"  Removed duplicate rows: {duplicates_removed}")
    
    print(f"  ✓ Data quality checks complete")
    print(f"  Final rows: {len(df_combined)} (removed {initial_rows - len(df_combined)} total)")
    
    # ============================================
    # ADD DERIVED COLUMNS
    # ============================================
    print(f"\nAdding derived columns...")
    
    # Sort by symbol and date for proper calculations
    df_combined = df_combined.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Daily change and daily change percent
    df_combined['daily_change'] = df_combined.groupby('symbol')['close'].diff()
    df_combined['daily_change_percent'] = df_combined.groupby('symbol')['close'].pct_change() * 100
    
    # Price range (high - low) and price range percent
    df_combined['price_range'] = df_combined['high'] - df_combined['low']
    df_combined['price_range_percent'] = (df_combined['price_range'] / df_combined['close']) * 100
    
    # Date components
    df_combined['year'] = df_combined['date'].dt.year
    df_combined['month'] = df_combined['date'].dt.month
    df_combined['quarter'] = df_combined['date'].dt.quarter
    df_combined['day_of_week'] = df_combined['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df_combined['week_of_year'] = df_combined['date'].dt.isocalendar().week
    
    # Boolean indicators
    df_combined['is_positive_day'] = df_combined['daily_change'] > 0
    df_combined['is_negative_day'] = df_combined['daily_change'] < 0
    
    # Volume category using quartiles per symbol
    print(f"  Calculating volume categories...")
    df_combined['volume_category'] = 'Unknown'
    for symbol in df_combined['symbol'].unique():
        symbol_mask = df_combined['symbol'] == symbol
        symbol_volumes = df_combined.loc[symbol_mask, 'volume']
        
        if len(symbol_volumes) > 0:
            q1 = symbol_volumes.quantile(0.25)
            q2 = symbol_volumes.quantile(0.50)
            q3 = symbol_volumes.quantile(0.75)
            
            volume_conditions = [
                (df_combined['symbol'] == symbol) & (df_combined['volume'] <= q1),
                (df_combined['symbol'] == symbol) & (df_combined['volume'] > q1) & (df_combined['volume'] <= q2),
                (df_combined['symbol'] == symbol) & (df_combined['volume'] > q2) & (df_combined['volume'] <= q3),
                (df_combined['symbol'] == symbol) & (df_combined['volume'] > q3)
            ]
            volume_choices = ['Low', 'Medium', 'High', 'Very High']
            
            for condition, choice in zip(volume_conditions, volume_choices):
                df_combined.loc[condition, 'volume_category'] = choice
    
    # Volatility indicator (rolling standard deviation of daily_change_percent)
    print(f"  Calculating volatility indicators...")
    df_combined['volatility_indicator'] = (
        df_combined.groupby('symbol')['daily_change_percent']
        .rolling(window=20, min_periods=1)
        .std()
        .reset_index(0, drop=True)
    )
    
    # Volatility category
    df_combined['volatility_category'] = df_combined['volatility_indicator'].apply(categorize_volatility)
    
    # Moving averages per symbol
    print(f"  Calculating moving averages...")
    df_combined['ma_5'] = df_combined.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    df_combined['ma_20'] = df_combined.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    
    # Price vs moving averages (percentage difference)
    df_combined['price_vs_ma5'] = ((df_combined['close'] - df_combined['ma_5']) / df_combined['ma_5']) * 100
    df_combined['price_vs_ma20'] = ((df_combined['close'] - df_combined['ma_20']) / df_combined['ma_20']) * 100
    
    print(f"  ✓ All derived columns added")
    
    # ============================================
    # FINAL CLEANUP AND COLUMN ORDERING
    # ============================================
    print(f"\nFinalizing DataFrame structure...")
    
    # Define proper column order
    column_order = [
        # Core columns
        'symbol', 'date',
        # OHLCV
        'open', 'high', 'low', 'close', 'volume',
        # Daily changes
        'daily_change', 'daily_change_percent',
        # Price range
        'price_range', 'price_range_percent',
        # Date components
        'year', 'month', 'quarter', 'day_of_week', 'week_of_year',
        # Boolean indicators
        'is_positive_day', 'is_negative_day',
        # Volume
        'volume_category',
        # Volatility
        'volatility_indicator', 'volatility_category',
        # Moving averages
        'ma_5', 'ma_20',
        # Price vs MA
        'price_vs_ma5', 'price_vs_ma20'
    ]
    
    # Get columns that exist in DataFrame
    existing_columns = [col for col in column_order if col in df_combined.columns]
    # Add any remaining columns that weren't in the order
    remaining_columns = [col for col in df_combined.columns if col not in existing_columns]
    
    # Reorder columns
    df_combined = df_combined[existing_columns + remaining_columns]
    
    # Set date as index (optional, but useful for time series analysis)
    # df_combined = df_combined.set_index('date')
    
    print(f"  ✓ DataFrame finalized")
    print(f"\n{'='*60}")
    print("Transformation Complete!")
    print(f"{'='*60}")
    print(f"Final DataFrame shape: {df_combined.shape}")
    print(f"Columns: {len(df_combined.columns)}")
    print(f"Symbols: {df_combined['symbol'].nunique()}")
    print(f"Date range: {df_combined['date'].min()} to {df_combined['date'].max()}")
    print(f"{'='*60}\n")
    
    return df_combined


if __name__ == "__main__":
    """
    Test section for the transform module.
    Fetches data for AAPL and MSFT, transforms it, displays statistics, and saves to CSV.
    """
    print("\n" + "="*60)
    print("Stock Data Transformation - Test Suite")
    print("="*60)
    
    # Import extract module functions
    try:
        from extract import fetch_multiple_stocks
        print("✓ Successfully imported extract module\n")
    except ImportError as e:
        print(f"✗ ERROR: Could not import extract module: {str(e)}")
        print("Make sure extract.py is in the same directory (src/)")
        exit(1)
    
    # Check if API key is set
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("⚠ WARNING: ALPHA_VANTAGE_API_KEY environment variable is not set!")
        print("Please set it before running tests:")
        print("  export ALPHA_VANTAGE_API_KEY='your_api_key_here'")
        print("\nContinuing with tests (will fail if API key is required)...\n")
    else:
        print(f"✓ API Key found: {api_key[:8]}...{api_key[-4:]}\n")
    
    # Test symbols
    test_symbols = ["AAPL"]
    print(f"Test symbols: {', '.join(test_symbols)}")
    print(f"This will make {len(test_symbols)} API calls with 12-second delays.")
    print(f"Estimated time: ~{len(test_symbols) * 12} seconds\n")
    
    user_input = input("Proceed with fetching and transforming stock data? (y/n): ").strip().lower()
    
    if user_input != 'y' and user_input != 'yes':
        print("\n⚠ Test SKIPPED: User chose not to proceed")
        exit(0)
    
    # Fetch data
    print("\n" + "-"*60)
    print("STEP 1: Fetching stock data")
    print("-"*60)
    try:
        raw_data_list = fetch_multiple_stocks(test_symbols, interval="daily", delay=12)
        successful_fetches = [d for d in raw_data_list if d is not None]
        print(f"\n✓ Successfully fetched {len(successful_fetches)}/{len(test_symbols)} stocks")
    except Exception as e:
        print(f"\n✗ Failed to fetch data: {str(e)}")
        exit(1)
    
    if not successful_fetches:
        print("\n✗ No data was successfully fetched. Exiting...")
        exit(1)
    
    # Transform data
    print("\n" + "-"*60)
    print("STEP 2: Transforming stock data")
    print("-"*60)
    try:
        df_transformed = transform_stock_data(raw_data_list)
        print(f"\n✓ Successfully transformed data")
    except Exception as e:
        print(f"\n✗ Failed to transform data: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Display transformation results with statistics
    print("\n" + "-"*60)
    print("STEP 3: Transformation Statistics")
    print("-"*60)
    
    print(f"\nDataFrame Info:")
    print(f"  Shape: {df_transformed.shape}")
    print(f"  Memory usage: {df_transformed.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nSymbol Summary:")
    for symbol in df_transformed['symbol'].unique():
        symbol_df = df_transformed[df_transformed['symbol'] == symbol]
        print(f"\n  {symbol}:")
        print(f"    Rows: {len(symbol_df)}")
        print(f"    Date range: {symbol_df['date'].min()} to {symbol_df['date'].max()}")
        print(f"    Avg daily change: {symbol_df['daily_change_percent'].mean():.2f}%")
        print(f"    Avg volatility: {symbol_df['volatility_indicator'].mean():.2f}%")
        print(f"    Positive days: {symbol_df['is_positive_day'].sum()} ({symbol_df['is_positive_day'].sum()/len(symbol_df)*100:.1f}%)")
        print(f"    Negative days: {symbol_df['is_negative_day'].sum()} ({symbol_df['is_negative_day'].sum()/len(symbol_df)*100:.1f}%)")
    
    print(f"\nColumn Statistics:")
    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
    print(df_transformed[numeric_cols].describe())
    
    print(f"\nVolume Category Distribution:")
    print(df_transformed['volume_category'].value_counts())
    
    print(f"\nVolatility Category Distribution:")
    print(df_transformed['volatility_category'].value_counts())
    
    # Save to CSV
    print("\n" + "-"*60)
    print("STEP 4: Saving transformed data to CSV")
    print("-"*60)
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        output_file = 'data/transformed_stock_data.csv'
        df_transformed.to_csv(output_file, index=False)
        
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"✓ Successfully saved transformed data to {output_file}")
        print(f"  File size: {file_size:.2f} KB")
        print(f"  Rows: {len(df_transformed)}")
        print(f"  Columns: {len(df_transformed.columns)}")
        
    except Exception as e:
        print(f"✗ Failed to save data: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print("\n" + "="*60)
    print("Test Suite Complete")
    print("="*60)
    print(f"✓ Data fetched: {len(successful_fetches)}/{len(test_symbols)} symbols")
    print(f"✓ Data transformed: {df_transformed.shape[0]} rows, {df_transformed.shape[1]} columns")
    print(f"✓ Data saved: data/transformed_stock_data.csv")
    print("="*60 + "\n")
