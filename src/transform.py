"""
Transform module for cleaning and transforming stock market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class StockDataTransformer:
    """Transforms raw stock market data into a clean, standardized format."""
    
    def __init__(self):
        """Initialize the transformer."""
        pass
    
    def transform_api_data(self, raw_data: Dict, symbol: str) -> pd.DataFrame:
        """
        Transform API response data into DataFrame.
        
        Args:
            raw_data: Raw API response dictionary
            symbol: Stock symbol
            
        Returns:
            Transformed DataFrame
        """
        try:
            # Extract time series data (adjust based on API structure)
            time_series_key = None
            for key in raw_data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                logger.warning(f"No time series data found for {symbol}")
                return pd.DataFrame()
            
            time_series = raw_data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            
            # Rename columns to standard format
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            df.rename(columns=column_mapping, inplace=True)
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Add calculated fields
            df = self._add_calculated_fields(df)
            
            logger.info(f"Transformed {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error transforming data for {symbol}: {str(e)}")
            raise
    
    def _add_calculated_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional calculated fields
        """
        # Daily returns
        df['daily_return'] = df['close'].pct_change()
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Volatility (rolling standard deviation of returns)
        df['volatility'] = df['daily_return'].rolling(window=20).std()
        
        # High-Low spread
        df['hl_spread'] = df['high'] - df['low']
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame by removing nulls and outliers.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        initial_rows = len(df)
        
        # Remove rows with null values in critical columns
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Remove outliers (prices that are more than 3 standard deviations away)
        for col in ['open', 'high', 'low', 'close']:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
        
        # Ensure volume is positive
        df = df[df['volume'] > 0]
        
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"Cleaned data: removed {removed_rows} rows")
        
        return df
    
    def standardize_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize DataFrame format and column order.
        
        Args:
            df: DataFrame to standardize
            
        Returns:
            Standardized DataFrame
        """
        # Ensure date is the index
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        
        # Standard column order
        standard_columns = ['symbol', 'open', 'high', 'low', 'close', 'volume']
        existing_columns = [col for col in standard_columns if col in df.columns]
        other_columns = [col for col in df.columns if col not in standard_columns]
        
        df = df[existing_columns + other_columns]
        
        return df
    
    def transform(self, raw_data: Dict, symbol: str) -> pd.DataFrame:
        """
        Complete transformation pipeline.
        
        Args:
            raw_data: Raw data dictionary
            symbol: Stock symbol
            
        Returns:
            Fully transformed and cleaned DataFrame
        """
        df = self.transform_api_data(raw_data, symbol)
        df = self.clean_data(df)
        df = self.standardize_format(df)
        
        return df

