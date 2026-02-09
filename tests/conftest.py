import pytest
import pandas as pd
from datetime import datetime, timedelta
import os


@pytest.fixture
def mock_api_response():
    """Mock Alpha Vantage API response with realistic data"""
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(20)]
    time_series = {}

    base_price = 150.0
    for i, date in enumerate(dates):
        price = base_price + i
        time_series[date] = {
            '1. open': str(price),
            '2. high': str(price + 5),
            '3. low': str(price - 2),
            '4. close': str(price + 3),
            '5. volume': str(50000000 + i * 1000000)
        }

    return {
        'Meta Data': {
            '1. Information': 'Daily Prices',
            '2. Symbol': 'AAPL',
            '3. Last Refreshed': dates[0],
            '4. Output Size': 'Compact',
            '5. Time Zone': 'US/Eastern'
        },
        'Time Series (Daily)': time_series
    }


@pytest.fixture
def sample_raw_stock_data():
    """Sample raw stock data list for testing transform"""
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(20)]
    time_series = {}

    base_price = 150.0
    for i, date in enumerate(dates):
        price = base_price + i
        time_series[date] = {
            '1. open': str(price),
            '2. high': str(price + 5),
            '3. low': str(price - 2),
            '4. close': str(price + 3),
            '5. volume': str(50000000 + i * 1000000)
        }

    return [{
        'symbol': 'AAPL',
        'time_series': time_series,
        'extracted_at': datetime.now().isoformat(),
        'data_source': 'Alpha Vantage'
    }]


@pytest.fixture
def sample_transformed_df():
    """Sample transformed DataFrame with all required columns"""
    dates = pd.date_range(end=datetime.now(), periods=20, freq='D')

    df = pd.DataFrame({
        'symbol': ['AAPL'] * 20,
        'date': dates,
        'open': [150.0 + i for i in range(20)],
        'high': [155.0 + i for i in range(20)],
        'low': [149.0 + i for i in range(20)],
        'close': [154.0 + i for i in range(20)],
        'volume': [50000000 + i * 1000000 for i in range(20)],
        'daily_change': [4.0] * 20,
        'daily_change_percent': [2.67] * 20,
        'price_range': [6.0] * 20,
        'price_range_percent': [4.03] * 20,
        'year': [2026] * 20,
        'month': [2] * 20,
        'quarter': [1] * 20,
        'day_of_week': [5] * 20,   # 0=Monday, 6=Sunday (5=Saturday)
        'week_of_year': [6] * 20,
        'is_positive_day': [True] * 20,
        'is_negative_day': [False] * 20,
        'volume_category': ['High'] * 20,
        'volatility_indicator': [4.0] * 20,
        'volatility_category': ['High'] * 20,
        'ma_5': [152.0 + i for i in range(20)],
        'ma_20': [150.0 + i for i in range(20)],
        'price_vs_ma5': [1.32] * 20,
        'price_vs_ma20': [2.67] * 20,
        'extracted_at': [datetime.now()] * 20,
        'data_source': ['Alpha Vantage'] * 20
    })

    return df


@pytest.fixture
def test_env(monkeypatch):
    """Set test environment variables"""
    monkeypatch.setenv('ALPHA_VANTAGE_API_KEY', 'test_api_key_12345')
    monkeypatch.setenv('DATABASE_URL', 'postgresql://dataengineer:password123@localhost:5432/stock_db')
    return True
