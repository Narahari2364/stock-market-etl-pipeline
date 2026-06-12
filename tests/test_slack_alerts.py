from unittest.mock import patch, MagicMock
from src.slack_alerts import (
    send_slack_message,
    send_pipeline_success_slack,
    send_pipeline_failure_slack,
    send_data_quality_warning_slack,
)


class TestSlackAlerts:
    def test_send_slack_not_configured(self, monkeypatch):
        monkeypatch.delenv('SLACK_WEBHOOK_URL', raising=False)
        assert send_slack_message('hello') is False

    @patch('src.slack_alerts.requests.post')
    def test_send_slack_success(self, mock_post, monkeypatch):
        monkeypatch.setenv('SLACK_WEBHOOK_URL', 'https://hooks.slack.com/test')
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        assert send_slack_message('Pipeline ok', status='success', details={'Records': '10'}) is True
        mock_post.assert_called_once()

    @patch('src.slack_alerts.requests.post', side_effect=Exception("network"))
    def test_send_slack_failure(self, mock_post, monkeypatch):
        monkeypatch.setenv('SLACK_WEBHOOK_URL', 'https://hooks.slack.com/test')
        assert send_slack_message('fail') is False

    @patch('src.slack_alerts.send_slack_message')
    def test_pipeline_success_slack(self, mock_send):
        send_pipeline_success_slack(500, 5, ['AAPL', 'MSFT'])
        mock_send.assert_called_once()
        assert mock_send.call_args[1]['status'] == 'success'

    @patch('src.slack_alerts.send_slack_message')
    def test_pipeline_failure_slack(self, mock_send):
        send_pipeline_failure_slack('error', step='EXTRACT')
        mock_send.assert_called_once()
        assert mock_send.call_args[1]['status'] == 'error'

    @patch('src.slack_alerts.send_slack_message')
    def test_data_quality_warning_slack(self, mock_send):
        send_data_quality_warning_slack('bad data')
        mock_send.assert_called_once()
        assert mock_send.call_args[1]['status'] == 'warning'
