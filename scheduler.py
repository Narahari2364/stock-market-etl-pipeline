import schedule
import time
import subprocess
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import alert modules
from src.alerts import send_pipeline_success_email, send_pipeline_failure_email
from src.slack_alerts import send_pipeline_success_slack, send_pipeline_failure_slack

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/scheduler.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def run_pipeline():
    """Run the ETL pipeline"""
    logging.info("=" * 70)
    logging.info(f"🚀 Scheduler triggered pipeline run at {datetime.now()}")
    logging.info("=" * 70)
    
    try:
        # Run the pipeline
        result = subprocess.run(
            ['python3', 'src/pipeline.py'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Log the output
        if result.stdout:
            logging.info("Pipeline output:")
            logging.info(result.stdout)
        
        # Check result
        if result.returncode == 0:
            logging.info("✅ Pipeline completed successfully")
            
            # Send success alerts
            try:
                # Try to extract info from stdout (records, symbols)
                # Default values if parsing fails
                records_loaded = 0
                symbols_count = 17  # Default from pipeline
                symbols_list = []
                
                # Try to parse records from stdout
                if result.stdout:
                    import re
                    records_match = re.search(r'Records loaded: ([\d,]+)', result.stdout)
                    if records_match:
                        records_loaded = int(records_match.group(1).replace(',', ''))
                    
                    symbols_match = re.search(r'Symbols processed: (\d+)', result.stdout)
                    if symbols_match:
                        symbols_count = int(symbols_match.group(1))
                
                send_pipeline_success_email(
                    records_loaded=records_loaded,
                    symbols_count=symbols_count,
                    symbols_list=symbols_list if symbols_list else ['N/A']
                )
                send_pipeline_success_slack(
                    records=records_loaded,
                    symbols_count=symbols_count,
                    symbols_list=symbols_list if symbols_list else ['N/A']
                )
            except Exception as alert_error:
                logging.warning(f"Failed to send success alerts: {alert_error}")
        else:
            logging.error(f"❌ Pipeline failed with exit code: {result.returncode}")
            error_message = result.stderr if result.stderr else f"Pipeline failed with exit code {result.returncode}"
            if result.stderr:
                logging.error(f"Error details: {result.stderr}")
            
            # Send failure alerts
            try:
                send_pipeline_failure_email(error_message, step='Scheduled Pipeline Run')
                send_pipeline_failure_slack(error_message, step='Scheduled Pipeline Run')
            except Exception as alert_error:
                logging.warning(f"Failed to send failure alerts: {alert_error}")
                
    except Exception as e:
        logging.error(f"❌ Exception while running pipeline: {str(e)}")
        
        # Send failure alerts for exceptions
        try:
            send_pipeline_failure_email(str(e), step='Scheduled Pipeline Run (Exception)')
            send_pipeline_failure_slack(str(e), step='Scheduled Pipeline Run (Exception)')
        except Exception as alert_error:
            logging.warning(f"Failed to send failure alerts: {alert_error}")
    
    logging.info("=" * 70)

# Schedule the job - runs every day at 9:00 AM
schedule.every().day.at("09:00").do(run_pipeline)

# For testing - run every 2 minutes
# schedule.every(2).minutes.do(run_pipeline)

if __name__ == "__main__":
    logging.info("=" * 70)
    logging.info("🚀 Stock Market ETL Scheduler Started")
    logging.info("=" * 70)
    logging.info(f"📅 Current time: {datetime.now()}")
    logging.info(f"⏰ Pipeline scheduled to run daily at 9:00 AM")
    logging.info(f"📈 Pipeline default: 25 stocks (configurable in pipeline)")
    logging.info(f"📝 Logs: logs/scheduler.log")
    logging.info(f"📊 Pipeline logs: logs/pipeline_*.log")
    logging.info("=" * 70)
    logging.info("Press Ctrl+C to stop the scheduler")
    logging.info("")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logging.info("")
        logging.info("=" * 70)
        logging.info("⏸️  Scheduler stopped by user")
        logging.info("=" * 70)
