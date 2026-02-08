#!/bin/bash
PID=$(ps aux | grep '[p]ython3 scheduler.py' | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ Scheduler is NOT running"
else
    echo "✅ Scheduler is running (PID: $PID)"
    echo ""
    echo "Recent scheduler logs:"
    tail -10 logs/scheduler.log
fi
