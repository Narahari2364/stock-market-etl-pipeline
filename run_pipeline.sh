#!/bin/bash

# Stock Market ETL Pipeline - Automated Run Script
# This script activates the virtual environment and runs the pipeline

# Change to project directory
cd /Users/narah/Documents/stock-market-etl-pipeline

# Activate virtual environment
source venv/bin/activate

# Set environment variables (load from .env)
export $(cat .env | xargs)

# Run the pipeline with timestamp
echo "=========================================="
echo "Starting ETL Pipeline: $(date)"
echo "=========================================="

python3 src/pipeline.py

# Capture exit code
EXIT_CODE=$?

echo "=========================================="
echo "Pipeline finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Optional: Send notification on failure
if [ $EXIT_CODE -ne 0 ]; then
    echo "⚠️ Pipeline failed with exit code $EXIT_CODE"
    # Add email/slack notification here if needed
fi

exit $EXIT_CODE

