"""
Extract module for fetching stock market data from Alpha Vantage API.

This module provides functions to extract stock market data including:
- Daily time series data
- Company fundamental data
- Batch processing with rate limit handling
"""

import os
import json
import time
import requests
from typing import Dict, List, Optional
from datetime import datetime


def fetch_stock_data(symbol: str, interval: str = "daily") -> Dict:
    """
    Fetch stock market time series data from Alpha Vantage API.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        interval: Time interval for data ('daily', 'weekly', 'monthly')
                  Note: Alpha Vantage TIME_SERIES_DAILY is used regardless of interval
                  for simplicity, but interval parameter is kept for future extensibility
    
    Returns:
        Dictionary containing:
            - symbol: Stock symbol
            - time_series: Dictionary of date -> OHLCV data
            - metadata: Dictionary with API metadata
            - last_refreshed: Last update timestamp
            - timezone: Data timezone
    
    Raises:
        ValueError: If API key is missing or symbol is invalid
        requests.RequestException: If network error occurs
        KeyError: If API response format is unexpected
    """
    print(f"\n{'='*60}")
    print(f"Fetching stock data for symbol: {symbol.upper()}")
    print(f"{'='*60}")
    
    # Get API key from environment variable
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        error_msg = "ALPHA_VANTAGE_API_KEY environment variable is not set"
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    print(f"API Key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Alpha Vantage API endpoint
    base_url = "https://www.alphavantage.co/query"
    
    # Prepare API parameters
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol.upper(),
        'apikey': api_key,
        'outputsize': 'full'  # 'compact' for last 100 data points, 'full' for all available
    }
    
    print(f"Requesting data from Alpha Vantage API...")
    print(f"Parameters: function={params['function']}, symbol={params['symbol']}, outputsize={params['outputsize']}")
    
    try:
        # Make API request
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        print(f"API Response Status: {response.status_code}")
        
        # Parse JSON response
        data = response.json()
        
        # Print actual response data for debugging
        print(f"API Response Data: {data}")
        
        # Check for API errors
        if "Information" in data:
            print(f"API Message: {data['Information']}")
            return None
        
        if 'Error Message' in data:
            error_msg = f"API Error: {data['Error Message']}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        if 'Note' in data:
            error_msg = f"API Rate Limit: {data['Note']}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        if 'Invalid API call' in str(data):
            error_msg = f"Invalid API call. Check symbol: {symbol}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Extract time series data
        time_series_key = None
        for key in data.keys():
            if 'Time Series (Daily)' in key:
                time_series_key = key
                break
        
        if not time_series_key:
            error_msg = f"No time series data found in API response for {symbol}"
            print(f"ERROR: {error_msg}")
            print(f"Available keys in response: {list(data.keys())}")
            raise KeyError(error_msg)
        
        time_series = data[time_series_key]
        metadata = data.get('Meta Data', {})
        
        # Structure the response
        result = {
            'symbol': symbol.upper(),
            'time_series': time_series,
            'metadata': metadata,
            'last_refreshed': metadata.get('3. Last Refreshed', ''),
            'timezone': metadata.get('5. Time Zone', ''),
            'data_points': len(time_series)
        }
        
        print(f"✓ Successfully fetched data for {symbol.upper()}")
        print(f"  - Data points: {result['data_points']}")
        print(f"  - Last refreshed: {result['last_refreshed']}")
        print(f"  - Timezone: {result['timezone']}")
        
        return result
        
    except requests.exceptions.Timeout:
        error_msg = "Request timed out. Please try again later."
        print(f"ERROR: {error_msg}")
        raise requests.RequestException(error_msg)
    
    except requests.exceptions.ConnectionError:
        error_msg = "Network connection error. Please check your internet connection."
        print(f"ERROR: {error_msg}")
        raise requests.RequestException(error_msg)
    
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise requests.RequestException(error_msg)
    
    except json.JSONDecodeError:
        error_msg = "Invalid JSON response from API"
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise


def fetch_company_overview(symbol: str) -> Dict:
    """
    Fetch company fundamental data (overview) from Alpha Vantage API.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
    
    Returns:
        Dictionary containing company overview data including:
            - Symbol, Name, Description
            - Sector, Industry
            - Market Capitalization, P/E Ratio
            - 52 Week High/Low
            - Dividend information
            - And more fundamental metrics
    
    Raises:
        ValueError: If API key is missing or symbol is invalid
        requests.RequestException: If network error occurs
    """
    print(f"\n{'='*60}")
    print(f"Fetching company overview for symbol: {symbol.upper()}")
    print(f"{'='*60}")
    
    # Get API key from environment variable
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        error_msg = "ALPHA_VANTAGE_API_KEY environment variable is not set"
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    print(f"API Key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Alpha Vantage API endpoint
    base_url = "https://www.alphavantage.co/query"
    
    # Prepare API parameters
    params = {
        'function': 'OVERVIEW',
        'symbol': symbol.upper(),
        'apikey': api_key
    }
    
    print(f"Requesting company overview from Alpha Vantage API...")
    
    try:
        # Make API request
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        print(f"API Response Status: {response.status_code}")
        
        # Parse JSON response
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            error_msg = f"API Error: {data['Error Message']}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        if 'Note' in data:
            error_msg = f"API Rate Limit: {data['Note']}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Check if data is empty (invalid symbol)
        if not data or len(data) == 0:
            error_msg = f"No data found for symbol: {symbol}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Check if symbol field exists (indicates valid response)
        if 'Symbol' not in data:
            error_msg = f"Invalid response format for symbol: {symbol}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        print(f"✓ Successfully fetched company overview for {symbol.upper()}")
        print(f"  - Company Name: {data.get('Name', 'N/A')}")
        print(f"  - Sector: {data.get('Sector', 'N/A')}")
        print(f"  - Industry: {data.get('Industry', 'N/A')}")
        print(f"  - Market Cap: {data.get('MarketCapitalization', 'N/A')}")
        print(f"  - P/E Ratio: {data.get('PERatio', 'N/A')}")
        
        return data
        
    except requests.exceptions.Timeout:
        error_msg = "Request timed out. Please try again later."
        print(f"ERROR: {error_msg}")
        raise requests.RequestException(error_msg)
    
    except requests.exceptions.ConnectionError:
        error_msg = "Network connection error. Please check your internet connection."
        print(f"ERROR: {error_msg}")
        raise requests.RequestException(error_msg)
    
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise requests.RequestException(error_msg)
    
    except json.JSONDecodeError:
        error_msg = "Invalid JSON response from API"
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise


def fetch_multiple_stocks(symbols: List[str], interval: str = "daily", delay: int = 12) -> List[Dict]:
    """
    Fetch stock data for multiple symbols with rate limit handling.
    
    Alpha Vantage API has a rate limit of 5 API calls per minute for free tier.
    This function implements delays between requests to respect rate limits.
    
    Args:
        symbols: List of stock symbols to fetch (e.g., ['AAPL', 'MSFT', 'GOOGL'])
        interval: Time interval for data ('daily', 'weekly', 'monthly')
        delay: Delay in seconds between API calls (default: 12 seconds to stay under 5/min limit)
    
    Returns:
        List of dictionaries, each containing stock data from fetch_stock_data()
        Failed symbols will have None in the list (maintains order)
    
    Raises:
        ValueError: If symbols list is empty
    """
    if not symbols:
        error_msg = "Symbols list cannot be empty"
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    print(f"\n{'='*60}")
    print(f"Fetching data for {len(symbols)} stock symbols")
    print(f"Rate limit delay: {delay} seconds between requests")
    print(f"{'='*60}")
    
    results = []
    total_symbols = len(symbols)
    
    for index, symbol in enumerate(symbols, 1):
        print(f"\n[{index}/{total_symbols}] Processing: {symbol.upper()}")
        
        try:
            stock_data = fetch_stock_data(symbol, interval)
            results.append(stock_data)
            print(f"✓ Successfully processed {symbol.upper()}")
            
        except Exception as e:
            print(f"✗ Failed to fetch data for {symbol.upper()}: {str(e)}")
            results.append(None)
        
        # Add delay between requests (except for the last one)
        if index < total_symbols:
            print(f"Waiting {delay} seconds before next request (rate limit protection)...")
            time.sleep(delay)
    
    # Summary
    successful = sum(1 for r in results if r is not None)
    failed = total_symbols - successful
    
    print(f"\n{'='*60}")
    print(f"Batch Processing Summary")
    print(f"{'='*60}")
    print(f"Total symbols: {total_symbols}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    """
    Test section for the extract module.
    Tests single stock fetch, company overview, and multiple stocks with user confirmation.
    """
    print("\n" + "="*60)
    print("Alpha Vantage Stock Data Extraction - Test Suite")
    print("="*60)
    
    # Check if API key is set
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("\n⚠ WARNING: ALPHA_VANTAGE_API_KEY environment variable is not set!")
        print("Please set it before running tests:")
        print("  export ALPHA_VANTAGE_API_KEY='your_api_key_here'")
        print("\nContinuing with tests (will fail if API key is required)...\n")
    else:
        print(f"✓ API Key found: {api_key[:8]}...{api_key[-4:]}\n")
    
    # Test 1: Single stock fetch (AAPL)
    print("\n" + "-"*60)
    print("TEST 1: Fetching single stock data (AAPL)")
    print("-"*60)
    try:
        aapl_data = fetch_stock_data("AAPL", interval="daily")
        print(f"\n✓ Test 1 PASSED: Successfully fetched AAPL data")
        print(f"  Sample dates: {list(aapl_data['time_series'].keys())[:5]}")
    except Exception as e:
        print(f"\n✗ Test 1 FAILED: {str(e)}")
        aapl_data = None
    
    # Test 2: Company overview
    print("\n" + "-"*60)
    print("TEST 2: Fetching company overview (AAPL)")
    print("-"*60)
    try:
        aapl_overview = fetch_company_overview("AAPL")
        print(f"\n✓ Test 2 PASSED: Successfully fetched AAPL company overview")
        print(f"  Company: {aapl_overview.get('Name', 'N/A')}")
        print(f"  Description: {aapl_overview.get('Description', 'N/A')[:100]}...")
    except Exception as e:
        print(f"\n✗ Test 2 FAILED: {str(e)}")
        aapl_overview = None
    
    # Test 3: Multiple stocks (with user confirmation)
    print("\n" + "-"*60)
    print("TEST 3: Fetching multiple stocks")
    print("-"*60)
    
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    print(f"\nTest symbols: {', '.join(test_symbols)}")
    print(f"This will make {len(test_symbols)} API calls with 12-second delays.")
    print(f"Estimated time: ~{len(test_symbols) * 12} seconds")
    
    user_input = input("\nProceed with multiple stocks test? (y/n): ").strip().lower()
    
    if user_input == 'y' or user_input == 'yes':
        try:
            multiple_data = fetch_multiple_stocks(test_symbols, interval="daily", delay=12)
            successful_fetches = [d for d in multiple_data if d is not None]
            print(f"\n✓ Test 3 PASSED: Successfully fetched {len(successful_fetches)}/{len(test_symbols)} stocks")
        except Exception as e:
            print(f"\n✗ Test 3 FAILED: {str(e)}")
            multiple_data = []
    else:
        print("\n⚠ Test 3 SKIPPED: User chose not to proceed")
        multiple_data = []
    
    # Save sample data to JSON file
    print("\n" + "-"*60)
    print("Saving sample data to data/sample_stock_data.json")
    print("-"*60)
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Prepare sample data (limit time series to first 5 dates for file size)
        sample_data = {
            'test_timestamp': datetime.now().isoformat(),
            'single_stock_test': None,
            'company_overview_test': None,
            'multiple_stocks_test': []
        }
        
        if aapl_data:
            # Limit time series data to first 5 entries for sample
            limited_time_series = dict(list(aapl_data['time_series'].items())[:5])
            sample_data['single_stock_test'] = {
                'symbol': aapl_data['symbol'],
                'metadata': aapl_data['metadata'],
                'last_refreshed': aapl_data['last_refreshed'],
                'timezone': aapl_data['timezone'],
                'data_points': aapl_data['data_points'],
                'time_series_sample': limited_time_series
            }
        
        if aapl_overview:
            sample_data['company_overview_test'] = aapl_overview
        
        if multiple_data:
            for stock_data in multiple_data:
                if stock_data:
                    limited_time_series = dict(list(stock_data['time_series'].items())[:5])
                    sample_data['multiple_stocks_test'].append({
                        'symbol': stock_data['symbol'],
                        'metadata': stock_data['metadata'],
                        'last_refreshed': stock_data['last_refreshed'],
                        'data_points': stock_data['data_points'],
                        'time_series_sample': limited_time_series
                    })
        
        # Save to JSON file
        output_file = 'data/sample_stock_data.json'
        with open(output_file, 'w') as f:
            json.dump(sample_data, f, indent=2, default=str)
        
        print(f"✓ Successfully saved sample data to {output_file}")
        print(f"  File size: {os.path.getsize(output_file)} bytes")
        
    except Exception as e:
        print(f"✗ Failed to save sample data: {str(e)}")
    
    # Final summary
    print("\n" + "="*60)
    print("Test Suite Complete")
    print("="*60)
    print(f"Test 1 (Single Stock): {'PASSED' if aapl_data else 'FAILED'}")
    print(f"Test 2 (Company Overview): {'PASSED' if aapl_overview else 'FAILED'}")
    print(f"Test 3 (Multiple Stocks): {'PASSED' if multiple_data else 'SKIPPED/FAILED'}")
    print("="*60 + "\n")
