import schedule
import time
import subprocess
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        else:
            logging.error(f"❌ Pipeline failed with exit code: {result.returncode}")
            if result.stderr:
                logging.error(f"Error details: {result.stderr}")
                
    except Exception as e:
        logging.error(f"❌ Exception while running pipeline: {str(e)}")
    
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
