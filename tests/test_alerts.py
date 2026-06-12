from unittest.mock import patch, MagicMock
from src.alerts import (
    send_email_alert,
    send_pipeline_success_email,
    send_pipeline_failure_email,
    send_data_quality_warning_email,
)


class TestEmailAlerts:
    def test_send_email_alert_not_configured(self, monkeypatch):
        monkeypatch.delenv('ALERT_EMAIL', raising=False)
        monkeypatch.delenv('ALERT_EMAIL_PASSWORD', raising=False)
        monkeypatch.delenv('ALERT_RECIPIENT_EMAIL', raising=False)
        assert send_email_alert('Test', 'Body') is False

    @patch('src.alerts.smtplib.SMTP')
    def test_send_email_alert_success(self, mock_smtp, monkeypatch):
        monkeypatch.setenv('ALERT_EMAIL', 'sender@test.com')
        monkeypatch.setenv('ALERT_EMAIL_PASSWORD', 'secret')
        monkeypatch.setenv('ALERT_RECIPIENT_EMAIL', 'recipient@test.com')

        server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=server)
        mock_smtp.return_value = server

        result = send_email_alert('Subject', 'Message', status='success')
        assert result is True
        server.starttls.assert_called_once()
        server.login.assert_called_once()
        server.send_message.assert_called_once()
        server.quit.assert_called_once()

    @patch('src.alerts.smtplib.SMTP', side_effect=Exception("smtp fail"))
    def test_send_email_alert_failure(self, mock_smtp, monkeypatch):
        monkeypatch.setenv('ALERT_EMAIL', 'sender@test.com')
        monkeypatch.setenv('ALERT_EMAIL_PASSWORD', 'secret')
        monkeypatch.setenv('ALERT_RECIPIENT_EMAIL', 'recipient@test.com')
        assert send_email_alert('Subject', 'Message') is False

    @patch('src.alerts.send_email_alert')
    def test_pipeline_success_email(self, mock_send):
        mock_send.return_value = True
        send_pipeline_success_email(100, 3, ['AAPL', 'MSFT', 'GOOGL'])
        mock_send.assert_called_once()
        assert 'success' in mock_send.call_args.kwargs.get('status', mock_send.call_args[1].get('status', ''))

    @patch('src.alerts.send_email_alert')
    def test_pipeline_failure_email(self, mock_send):
        send_pipeline_failure_email('boom', step='LOAD')
        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args[1]
        assert call_kwargs['status'] == 'error'

    @patch('src.alerts.send_email_alert')
    def test_data_quality_warning_email(self, mock_send):
        send_data_quality_warning_email('null close values')
        mock_send.assert_called_once()
        assert mock_send.call_args[1]['status'] == 'warning'
