import json
import pytest
from unittest.mock import patch, Mock
from src.extract import fetch_stock_data, fetch_multiple_stocks, fetch_company_overview
import requests


class TestFetchStockData:
    """Tests for fetch_stock_data function"""

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_success(self, mock_get, mock_api_response, test_env):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = fetch_stock_data('AAPL')

        assert result is not None
        assert result['symbol'] == 'AAPL'
        assert 'time_series' in result
        assert len(result['time_series']) > 0
        assert 'data_points' in result

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_invalid_symbol(self, mock_get, test_env):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'Error Message': 'Invalid API call. Please check your symbol.'
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError):
            fetch_stock_data('INVALID')

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_rate_limit_information(self, mock_get, test_env):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'Information': 'Thank you for using Alpha Vantage! Our standard API rate limit is 5 requests per minute.'
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = fetch_stock_data('AAPL')
        assert result is None

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_rate_limit_note_raises(self, mock_get, test_env):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'Note': 'Thank you for using Alpha Vantage! Please consider premium.'
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Rate Limit"):
            fetch_stock_data('AAPL')

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_network_error(self, mock_get, test_env):
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        with pytest.raises(requests.RequestException):
            fetch_stock_data('AAPL')

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_timeout(self, mock_get, test_env):
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")

        with pytest.raises(requests.RequestException):
            fetch_stock_data('AAPL')

    def test_fetch_stock_data_missing_api_key(self, monkeypatch):
        monkeypatch.delenv('ALPHA_VANTAGE_API_KEY', raising=False)
        with pytest.raises(ValueError, match="ALPHA_VANTAGE_API_KEY"):
            fetch_stock_data('AAPL')

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_malformed_json(self, mock_get, test_env):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Invalid JSON"):
            fetch_stock_data('AAPL')

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_empty_response_no_time_series(self, mock_get, test_env):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'Meta Data': {'2. Symbol': 'AAPL'}}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(KeyError, match="No time series data"):
            fetch_stock_data('AAPL')

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_invalid_api_call_string(self, mock_get, test_env):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'Invalid API call': True, 'symbol': 'BAD'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Invalid API call"):
            fetch_stock_data('BAD')

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_http_error(self, mock_get, test_env):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = mock_response

        with pytest.raises(requests.RequestException, match="HTTP Error"):
            fetch_stock_data('AAPL')

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_uppercases_symbol(self, mock_get, mock_api_response, test_env):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = fetch_stock_data('aapl')
        assert result['symbol'] == 'AAPL'


class TestFetchCompanyOverview:
    @patch('src.extract.requests.get')
    def test_fetch_company_overview_success(self, mock_get, test_env):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'Symbol': 'AAPL',
            'Name': 'Apple Inc',
            'Sector': 'Technology',
            'Industry': 'Consumer Electronics',
            'MarketCapitalization': '3000000000000',
            'PERatio': '28.5',
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = fetch_company_overview('AAPL')
        assert result['Symbol'] == 'AAPL'
        assert result['Name'] == 'Apple Inc'

    @patch('src.extract.requests.get')
    def test_fetch_company_overview_error_message(self, mock_get, test_env):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'Error Message': 'Invalid symbol'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="API Error"):
            fetch_company_overview('BAD')

    @patch('src.extract.requests.get')
    def test_fetch_company_overview_rate_limit_note(self, mock_get, test_env):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'Note': 'rate limit'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Rate Limit"):
            fetch_company_overview('AAPL')

    @patch('src.extract.requests.get')
    def test_fetch_company_overview_empty_data(self, mock_get, test_env):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="No data found"):
            fetch_company_overview('EMPTY')

    @patch('src.extract.requests.get')
    def test_fetch_company_overview_invalid_format(self, mock_get, test_env):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'Name': 'No Symbol Key'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Invalid response format"):
            fetch_company_overview('AAPL')

    def test_fetch_company_overview_missing_api_key(self, monkeypatch):
        monkeypatch.delenv('ALPHA_VANTAGE_API_KEY', raising=False)
        with pytest.raises(ValueError, match="ALPHA_VANTAGE_API_KEY"):
            fetch_company_overview('AAPL')

    @patch('src.extract.requests.get')
    def test_fetch_company_overview_timeout(self, mock_get, test_env):
        mock_get.side_effect = requests.exceptions.Timeout("timeout")
        with pytest.raises(requests.RequestException):
            fetch_company_overview('AAPL')

    @patch('src.extract.requests.get')
    def test_fetch_company_overview_connection_error(self, mock_get, test_env):
        mock_get.side_effect = requests.exceptions.ConnectionError("offline")
        with pytest.raises(requests.RequestException):
            fetch_company_overview('AAPL')

    @patch('src.extract.requests.get')
    def test_fetch_company_overview_malformed_json(self, mock_get, test_env):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("err", "", 0)
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Invalid JSON"):
            fetch_company_overview('AAPL')


class TestFetchMultipleStocks:
    """Tests for fetch_multiple_stocks function"""

    @patch('src.extract.fetch_stock_data')
    @patch('src.extract.time.sleep')
    def test_fetch_multiple_stocks_success(self, mock_sleep, mock_fetch):
        mock_fetch.return_value = {
            'symbol': 'AAPL',
            'time_series': {'2026-01-23': {}},
            'data_points': 1
        }

        symbols = ['AAPL', 'MSFT', 'GOOGL']
        result = fetch_multiple_stocks(symbols, delay=0)

        assert len(result) == 3
        assert mock_fetch.call_count == 3

    @patch('src.extract.fetch_stock_data')
    @patch('src.extract.time.sleep')
    def test_fetch_multiple_stocks_with_failures(self, mock_sleep, mock_fetch):
        mock_fetch.side_effect = [
            {'symbol': 'AAPL', 'time_series': {}, 'data_points': 0},
            None,
            {'symbol': 'GOOGL', 'time_series': {}, 'data_points': 0}
        ]

        symbols = ['AAPL', 'MSFT', 'GOOGL']
        result = fetch_multiple_stocks(symbols, delay=0)

        assert len(result) == 3
        assert result[0] is not None
        assert result[1] is None
        assert result[2] is not None

    @patch('src.extract.fetch_stock_data')
    @patch('src.extract.time.sleep')
    def test_fetch_multiple_stocks_applies_delay(self, mock_sleep, mock_fetch):
        mock_fetch.return_value = {'symbol': 'AAPL', 'time_series': {}, 'data_points': 0}
        fetch_multiple_stocks(['AAPL', 'MSFT'], delay=12)
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(12)

    def test_fetch_multiple_stocks_empty_list_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            fetch_multiple_stocks([])

    @patch('src.extract.fetch_stock_data', side_effect=ValueError("bad symbol"))
    @patch('src.extract.time.sleep')
    def test_fetch_multiple_stocks_catches_exception(self, mock_sleep, mock_fetch):
        result = fetch_multiple_stocks(['BAD'], delay=0)
        assert result == [None]
