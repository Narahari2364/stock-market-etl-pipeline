#!/bin/bash
PID=$(ps aux | grep '[p]ython3 scheduler.py' | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ Scheduler not running"
else
    kill $PID
    echo "✅ Scheduler stopped (PID: $PID)"
fi
