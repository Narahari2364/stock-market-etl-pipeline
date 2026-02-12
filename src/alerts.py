import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def send_email_alert(subject, message, status='info'):
    """
    Send email alert using Gmail SMTP

    Args:
        subject (str): Email subject
        message (str): Email body
        status (str): Alert status (success, error, warning, info)

    Returns:
        bool: True if sent successfully, False otherwise
    """

    # Get credentials from environment
    sender_email = os.getenv('ALERT_EMAIL')
    sender_password = os.getenv('ALERT_EMAIL_PASSWORD')
    recipient_email = os.getenv('ALERT_RECIPIENT_EMAIL')

    if not all([sender_email, sender_password, recipient_email]):
        print("⚠️  Email alerts not configured. Skipping...")
        return False

    try:
        # Status emojis
        emoji_map = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        }
        emoji = emoji_map.get(status, '📧')

        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"{emoji} [{status.upper()}] {subject}"

        # Email body with HTML formatting
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: {'#28a745' if status == 'success' else '#dc3545' if status == 'error' else '#ffc107'};">
                {emoji} Stock Market ETL Pipeline Alert
            </h2>

            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <p><strong>Status:</strong> {status.upper()}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div style="margin: 20px 0;">
                <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; white-space: pre-wrap;">
{message}
                </pre>
            </div>

            <hr style="margin: 30px 0;">
            <p style="color: #6c757d; font-size: 12px;">
                This is an automated message from your Stock Market ETL Pipeline.
                <br>Dashboard: <a href="https://your-dashboard-url.streamlit.app">View Dashboard</a>
            </p>
        </body>
        </html>
        """

        msg.attach(MIMEText(html_body, 'html'))

        # Send email via Gmail SMTP
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()

        print(f"✅ Email alert sent to {recipient_email}")
        return True

    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False


def send_pipeline_success_email(records_loaded, symbols_count, symbols_list):
    """Send success notification email"""

    message = f"""
Pipeline executed successfully!

📊 Summary:
    • Records Loaded: {records_loaded}
    • Symbols Processed: {symbols_count}
    • Symbols: {', '.join(symbols_list[:10])}{'...' if len(symbols_list) > 10 else ''}

All data has been successfully extracted, transformed, and loaded to the database.

Database is up to date with the latest stock market data.
    """

    send_email_alert(
        subject="Pipeline Completed Successfully",
        message=message,
        status='success'
    )


def send_pipeline_failure_email(error_message, step='Unknown'):
    """Send failure notification email"""

    message = f"""
Pipeline failed during execution!

❌ Error Details:
    • Failed Step: {step}
    • Error Message: {error_message}

Please check the logs for more detailed information:
    • Log location: logs/pipeline_*.log
    • Check Streamlit dashboard for data freshness

Action Required: Review the error and re-run the pipeline.
    """

    send_email_alert(
        subject="Pipeline Failed",
        message=message,
        status='error'
    )


def send_data_quality_warning_email(issues):
    """Send data quality warning email"""

    message = f"""
Data quality issues detected!

⚠️  Issues Found:
{issues}

The pipeline continued with data loading, but please review the data quality.
    """

    send_email_alert(
        subject="Data Quality Warning",
        message=message,
        status='warning'
    )
