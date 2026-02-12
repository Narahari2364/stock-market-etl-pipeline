import requests
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def send_slack_message(message, status='info', details=None):
    """
    Send message to Slack webhook

    Args:
        message (str): Main message
        status (str): Alert status (success, error, warning, info)
        details (dict): Additional details to display

    Returns:
        bool: True if sent successfully, False otherwise
    """

    webhook_url = os.getenv('SLACK_WEBHOOK_URL')

    if not webhook_url:
        print("⚠️  Slack webhook not configured. Skipping...")
        return False

    # Status emoji and color mapping
    status_config = {
        'success': {'emoji': ':white_check_mark:', 'color': '#28a745'},
        'error': {'emoji': ':x:', 'color': '#dc3545'},
        'warning': {'emoji': ':warning:', 'color': '#ffc107'},
        'info': {'emoji': ':information_source:', 'color': '#17a2b8'}
    }

    config = status_config.get(status, status_config['info'])
    emoji = config['emoji']
    color = config['color']

    # Build fields for details
    fields = [
        {
            "type": "mrkdwn",
            "text": f"*Status:*\n{status.upper()}"
        },
        {
            "type": "mrkdwn",
            "text": f"*Time:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        }
    ]

    if details:
        for key, value in details.items():
            fields.append({
                "type": "mrkdwn",
                "text": f"*{key}:*\n{value}"
            })

    # Slack message format with blocks
    slack_data = {
        "text": f"{emoji} Stock Market ETL Pipeline - {status.upper()}",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Stock Market ETL Pipeline"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message
                }
            },
            {
                "type": "section",
                "fields": fields
            },
            {
                "type": "divider"
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Automated alert from Stock Market ETL Pipeline | <https://your-dashboard.streamlit.app|View Dashboard>"
                    }
                ]
            }
        ],
        "attachments": [
            {
                "color": color,
                "text": ""
            }
        ]
    }

    try:
        response = requests.post(webhook_url, json=slack_data)
        response.raise_for_status()
        print("✅ Slack notification sent")
        return True
    except Exception as e:
        print(f"❌ Failed to send Slack notification: {e}")
        return False


def send_pipeline_success_slack(records, symbols_count, symbols_list):
    """Send success notification to Slack"""

    message = f"*Pipeline executed successfully!* :tada:"

    details = {
        "Records Loaded": f"`{records:,}`",
        "Symbols Processed": f"`{symbols_count}`",
        "Top Symbols": f"`{', '.join(symbols_list[:5])}`"
    }

    send_slack_message(message, status='success', details=details)


def send_pipeline_failure_slack(error, step='Unknown'):
    """Send failure notification to Slack"""

    message = f"*Pipeline failed!* :rotating_light:\n\nThe ETL pipeline encountered an error and could not complete."

    details = {
        "Failed Step": f"`{step}`",
        "Error": f"```{error[:500]}```"  # Limit error length
    }

    send_slack_message(message, status='error', details=details)


def send_data_quality_warning_slack(issues):
    """Send data quality warning to Slack"""

    message = f"*Data quality issues detected!* :warning:\n\nSome data quality checks failed."

    details = {
        "Issues": f"```{issues[:500]}```"
    }

    send_slack_message(message, status='warning', details=details)
