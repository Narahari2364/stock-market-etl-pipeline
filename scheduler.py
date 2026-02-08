#!/usr/bin/env python3
"""
Stock Market ETL Pipeline - Scheduler

Runs the ETL pipeline daily at 9:00 AM. Start in background with:
    python3 scheduler.py &

Avoids macOS launchd permission issues by using a simple Python scheduler.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Load environment before other imports
from dotenv import load_dotenv
load_dotenv()

import schedule
import time

# Project root (directory containing this script)
PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "scheduler.log"

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("scheduler")


def run_pipeline():
    """Run the ETL pipeline (non-interactive: default symbols, auto-confirm)."""
    logger.info("Starting scheduled ETL pipeline run")
    start = datetime.now()

    # Use default symbols and auto-confirm: Enter, then y
    pipeline_input = "\ny\n"

    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.pipeline"],
            cwd=PROJECT_ROOT,
            input=pipeline_input.encode(),
            capture_output=True,
            text=False,
            timeout=600,  # 10 minutes max
            env=os.environ.copy(),
        )

        elapsed = (datetime.now() - start).total_seconds()
        if result.returncode == 0:
            logger.info("Pipeline completed successfully in %.1f seconds", elapsed)
        else:
            logger.error(
                "Pipeline failed with exit code %s after %.1f seconds",
                result.returncode,
                elapsed,
            )
            if result.stderr:
                logger.error("stderr: %s", result.stderr.decode(errors="replace"))

    except subprocess.TimeoutExpired:
        logger.error("Pipeline run timed out after 10 minutes")
    except Exception as e:
        logger.exception("Pipeline run failed: %s", e)


def main():
    logger.info("Scheduler started. Next run: daily at 09:00")
    schedule.every().day.at("09:00").do(run_pipeline)

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    main()
