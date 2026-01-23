"""
Extract module for fetching stock market data from various sources.
"""

import os
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class StockDataExtractor:
    """Extracts stock market data from APIs or files."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the extractor.
        
        Args:
            api_key: API key for stock data provider (e.g., Alpha Vantage, Yahoo Finance)
        """
        self.api_key = api_key or os.getenv('STOCK_API_KEY')
        self.base_url = os.getenv('STOCK_API_URL', 'https://www.alphavantage.co/query')
    
    def extract_from_api(self, symbol: str, function: str = 'TIME_SERIES_DAILY') -> Dict:
        """
        Extract stock data from API.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            function: API function type
            
        Returns:
            Dictionary containing stock data
        """
        try:
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            logger.info(f"Extracting data for symbol: {symbol}")
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully extracted data for {symbol}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error extracting data for {symbol}: {str(e)}")
            raise
    
    def extract_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Extract stock data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame containing stock data
        """
        try:
            logger.info(f"Extracting data from CSV: {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Successfully extracted {len(df)} rows from CSV")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting data from CSV: {str(e)}")
            raise
    
    def extract_multiple_symbols(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Extract data for multiple stock symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to their data
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.extract_from_api(symbol)
            except Exception as e:
                logger.error(f"Failed to extract data for {symbol}: {str(e)}")
                results[symbol] = None
        
        return results

