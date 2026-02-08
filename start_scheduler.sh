#!/bin/bash
cd /Users/narah/Documents/stock-market-etl-pipeline
source venv/bin/activate
nohup python3 scheduler.py > logs/scheduler_console.log 2>&1 &
echo "✅ Scheduler started in background"
echo "📝 Check status: ps aux | grep scheduler.py"
echo "📋 View logs: tail -f logs/scheduler.log"
