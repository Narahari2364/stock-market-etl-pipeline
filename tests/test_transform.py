import pytest
import pandas as pd
from src.transform import transform_stock_data, categorize_volatility


class TestTransformStockData:
    """Tests for transform_stock_data function"""

    def test_transform_creates_required_columns(self, sample_raw_stock_data):
        """Test that transformation creates all required columns"""
        result = transform_stock_data(sample_raw_stock_data)

        required_columns = [
            'symbol', 'date', 'open', 'high', 'low', 'close', 'volume',
            'daily_change', 'daily_change_percent',
            'price_range', 'price_range_percent',
            'ma_5', 'ma_20', 'price_vs_ma5', 'price_vs_ma20',
            'volatility_indicator', 'volatility_category',
            'year', 'month', 'quarter', 'day_of_week', 'week_of_year',
            'is_positive_day', 'is_negative_day',
            'volume_category'
        ]

        for col in required_columns:
            assert col in result.columns, f"Missing required column: {col}"

    def test_moving_average_calculation(self, sample_raw_stock_data):
        """Test moving average calculations"""
        result = transform_stock_data(sample_raw_stock_data)

        # Check MA columns exist and have values
        assert 'ma_5' in result.columns
        assert 'ma_20' in result.columns
        assert not result['ma_5'].isna().all()

        # MA5 should be calculated after 5 data points
        assert result.iloc[-1]['ma_5'] > 0

    def test_daily_change_calculation(self, sample_raw_stock_data):
        """Test daily change calculations (day-over-day close diff)"""
        result = transform_stock_data(sample_raw_stock_data)

        # Transform uses close.diff() per symbol: daily_change = current close - previous close
        result_sorted = result.sort_values(['symbol', 'date']).reset_index(drop=True)
        for idx in range(1, len(result_sorted)):
            prev_close = result_sorted.iloc[idx - 1]['close']
            curr_close = result_sorted.iloc[idx]['close']
            expected_change = curr_close - prev_close
            actual_change = result_sorted.iloc[idx]['daily_change']
            if pd.notna(actual_change):
                assert abs(actual_change - expected_change) < 0.01
                if prev_close > 0:
                    expected_pct = (curr_close - prev_close) / prev_close * 100
                    assert abs(result_sorted.iloc[idx]['daily_change_percent'] - expected_pct) < 0.1

    def test_price_range_calculation(self, sample_raw_stock_data):
        """Test price range calculations"""
        result = transform_stock_data(sample_raw_stock_data)

        # Price range should be high - low
        for idx, row in result.iterrows():
            expected_range = row['high'] - row['low']
            assert abs(row['price_range'] - expected_range) < 0.01

    def test_data_quality_checks(self):
        """Test that invalid data is removed"""
        invalid_data = [{
            'symbol': 'TEST',
            'time_series': {
                '2026-01-23': {
                    '1. open': '-10',  # Invalid negative
                    '2. high': '100',
                    '3. low': '110',   # Invalid: low > high
                    '4. close': '105',
                    '5. volume': '1000'
                },
                '2026-01-22': {
                    '1. open': '100',
                    '2. high': '105',
                    '3. low': '95',
                    '4. close': '103',
                    '5. volume': '2000'
                }
            },
            'extracted_at': '2026-01-23'
        }]

        result = transform_stock_data(invalid_data)

        # Should remove rows with invalid data
        assert all(result['open'] > 0)
        assert all(result['high'] >= result['low'])
        assert all(result['high'] >= result['close'])

    def test_date_components(self, sample_raw_stock_data):
        """Test date component extraction"""
        result = transform_stock_data(sample_raw_stock_data)

        # Check date components exist
        assert 'year' in result.columns
        assert 'month' in result.columns
        assert 'quarter' in result.columns
        assert 'day_of_week' in result.columns
        assert 'week_of_year' in result.columns

        # Check values are reasonable
        assert all(result['year'] >= 2020)
        assert all((result['month'] >= 1) & (result['month'] <= 12))
        assert all((result['quarter'] >= 1) & (result['quarter'] <= 4))


class TestCategorizeVolatility:
    """Tests for volatility categorization"""

    def test_volatility_very_low(self):
        assert categorize_volatility(0.5) == 'Very Low'

    def test_volatility_low(self):
        assert categorize_volatility(1.5) == 'Low'

    def test_volatility_moderate(self):
        # Implementation uses 'Medium' not 'Moderate'
        assert categorize_volatility(2.5) == 'Medium'

    def test_volatility_high(self):
        assert categorize_volatility(4.0) == 'High'

    def test_volatility_very_high(self):
        assert categorize_volatility(6.0) == 'Very High'

    def test_volatility_boundary_cases(self):
        """Test boundary values"""
        assert categorize_volatility(1.0) == 'Low'
        assert categorize_volatility(2.0) == 'Medium'
        assert categorize_volatility(3.0) == 'Medium'
        assert categorize_volatility(5.0) == 'Very High'

    def test_volatility_unknown_for_nan(self):
        """Test that NaN returns Unknown"""
        assert categorize_volatility(float('nan')) == 'Unknown'
