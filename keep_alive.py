import time
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


def ping_database():
    """Ping database to keep it awake"""
    try:
        engine = create_engine(os.getenv('DATABASE_URL'))
        with engine.connect() as conn:
            result = conn.execute(text('SELECT 1'))
            result.scalar()
        print(f"✅ {datetime.now()}: Database ping successful")
        return True
    except Exception as e:
        print(f"❌ {datetime.now()}: Database ping failed: {e}")
        return False


if __name__ == "__main__":
    print("🔔 Starting database keep-alive service...")
    print("Pinging every 10 minutes to prevent sleep")
    print("Press Ctrl+C to stop")

    while True:
        ping_database()
        time.sleep(600)  # Sleep for 10 minutes (600 seconds)
