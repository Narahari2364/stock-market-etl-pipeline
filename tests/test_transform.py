import pytest
import pandas as pd
from src.transform import transform_stock_data, categorize_volatility


def _make_raw(symbol='AAPL', dates_prices=None, extra_time_series=None):
    """Build minimal raw stock dict for transform tests."""
    if dates_prices is None:
        dates_prices = [
            ('2026-01-20', 100.0),
            ('2026-01-21', 101.0),
            ('2026-01-22', 102.0),
        ]
    time_series = {}
    for date_str, price in dates_prices:
        time_series[date_str] = {
            '1. open': str(price),
            '2. high': str(price + 2),
            '3. low': str(price - 1),
            '4. close': str(price + 1),
            '5. volume': str(1_000_000),
        }
    if extra_time_series:
        time_series.update(extra_time_series)
    return {
        'symbol': symbol,
        'time_series': time_series,
        'extracted_at': '2026-01-23',
    }


class TestTransformStockData:
    """Tests for transform_stock_data function"""

    def test_transform_creates_required_columns(self, sample_raw_stock_data):
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
        result = transform_stock_data(sample_raw_stock_data)

        assert 'ma_5' in result.columns
        assert 'ma_20' in result.columns
        assert not result['ma_5'].isna().all()
        assert result.iloc[-1]['ma_5'] > 0

    def test_daily_change_calculation(self, sample_raw_stock_data):
        result = transform_stock_data(sample_raw_stock_data)
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
        result = transform_stock_data(sample_raw_stock_data)
        for _, row in result.iterrows():
            expected_range = row['high'] - row['low']
            assert abs(row['price_range'] - expected_range) < 0.01

    def test_data_quality_checks(self):
        invalid_data = [{
            'symbol': 'TEST',
            'time_series': {
                '2026-01-23': {
                    '1. open': '-10',
                    '2. high': '100',
                    '3. low': '110',
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
        assert all(result['open'] > 0)
        assert all(result['high'] >= result['low'])
        assert all(result['high'] >= result['close'])

    def test_date_components(self, sample_raw_stock_data):
        result = transform_stock_data(sample_raw_stock_data)
        assert all(result['year'] >= 2020)
        assert all((result['month'] >= 1) & (result['month'] <= 12))
        assert all((result['quarter'] >= 1) & (result['quarter'] <= 4))

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            transform_stock_data([])

    def test_all_none_entries_raises(self):
        with pytest.raises(ValueError, match="No valid stock data"):
            transform_stock_data([None, None])

    def test_empty_time_series_skipped_then_raises(self):
        with pytest.raises(ValueError, match="No dataframes were successfully created"):
            transform_stock_data([{'symbol': 'EMPTY', 'time_series': {}}])

    def test_single_row_input(self):
        result = transform_stock_data([_make_raw(dates_prices=[('2026-01-22', 150.0)])])
        assert len(result) == 1
        assert result.iloc[0]['symbol'] == 'AAPL'
        assert pd.isna(result.iloc[0]['daily_change'])

    def test_duplicate_symbol_date_removed(self):
        raw = _make_raw(extra_time_series={
            '2026-01-22': {
                '1. open': '200',
                '2. high': '205',
                '3. low': '195',
                '4. close': '203',
                '5. volume': '5000000',
            }
        })
        result = transform_stock_data([raw])
        assert len(result[result['date'] == pd.Timestamp('2026-01-22')]) == 1
        assert result[result['date'] == pd.Timestamp('2026-01-22')].iloc[0]['close'] == 203

    def test_null_critical_columns_dropped(self):
        raw = [{
            'symbol': 'NULLS',
            'time_series': {
                '2026-01-20': {
                    '1. open': '100',
                    '2. high': '105',
                    '3. low': '95',
                    '4. close': '103',
                    '5. volume': '1000',
                },
                '2026-01-21': {
                    '1. open': None,
                    '2. high': '105',
                    '3. low': '95',
                    '4. close': '103',
                    '5. volume': '1000',
                },
            },
            'extracted_at': '2026-01-23',
        }]
        result = transform_stock_data(raw)
        assert len(result) == 1

    def test_malformed_time_series_shape(self):
        raw = [{
            'symbol': 'BAD',
            'time_series': 'not-a-valid-dict',
            'extracted_at': '2026-01-23',
        }]
        with pytest.raises(ValueError, match="No dataframes were successfully created"):
            transform_stock_data(raw)

    def test_filters_none_entries_keeps_valid(self):
        valid = _make_raw('MSFT')
        result = transform_stock_data([None, valid])
        assert len(result) >= 1
        assert 'MSFT' in result['symbol'].values

    def test_multiple_symbols(self):
        result = transform_stock_data([
            _make_raw('AAPL'),
            _make_raw('MSFT'),
        ])
        assert result['symbol'].nunique() == 2

    def test_negative_volume_removed(self):
        raw = [{
            'symbol': 'VOL',
            'time_series': {
                '2026-01-20': {
                    '1. open': '10', '2. high': '12', '3. low': '9',
                    '4. close': '11', '5. volume': '-1',
                },
            },
            'extracted_at': '2026-01-23',
        }]
        result = transform_stock_data(raw)
        assert len(result) == 0


class TestCategorizeVolatility:
    """Tests for volatility categorization"""

    def test_volatility_very_low(self):
        assert categorize_volatility(0.5) == 'Very Low'

    def test_volatility_low(self):
        assert categorize_volatility(1.5) == 'Low'

    def test_volatility_moderate(self):
        assert categorize_volatility(2.5) == 'Medium'

    def test_volatility_high(self):
        assert categorize_volatility(4.0) == 'High'

    def test_volatility_very_high(self):
        assert categorize_volatility(6.0) == 'Very High'

    def test_volatility_boundary_cases(self):
        assert categorize_volatility(1.0) == 'Low'
        assert categorize_volatility(2.0) == 'Medium'
        assert categorize_volatility(3.0) == 'Medium'
        assert categorize_volatility(3.49) == 'Medium'
        assert categorize_volatility(5.0) == 'Very High'
        assert categorize_volatility(4.99) == 'High'

    def test_volatility_unknown_for_nan(self):
        assert categorize_volatility(float('nan')) == 'Unknown'

    def test_volatility_zero(self):
        assert categorize_volatility(0.0) == 'Very Low'
