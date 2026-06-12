# 📈 Stock Market ETL Pipeline

![Tests](https://github.com/Narahari2364/stock-market-etl-pipeline/workflows/Tests/badge.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Coverage](https://img.shields.io/badge/coverage-62%25-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A production-ready ETL (Extract, Transform, Load) pipeline for processing stock market data from Alpha Vantage API into PostgreSQL database. This project demonstrates end-to-end data engineering practices including API integration, data transformation, database operations, and comprehensive error handling.

## 📊 Project Stats

- **2900+** stock records across **17 symbols**
- **106** unit tests with **62%** code coverage
- **100%** data quality validation pass rate
- **Streamlit dashboard** with charts, filters, and ML predictions
- **Daily automated** data updates via scheduler

## 📋 Project Overview

This project demonstrates:

- **API Integration**: Automated data extraction from Alpha Vantage API with rate limit handling
- **Data Transformation**: Comprehensive data cleaning, validation, and feature engineering
- **Database Operations**: SQLAlchemy ORM models and efficient batch loading to PostgreSQL
- **Pipeline Orchestration**: Complete ETL workflow with logging and error handling
- **Data Quality**: Automated data validation, outlier detection, and duplicate removal
- **Technical Indicators**: Calculation of moving averages, volatility metrics, and price analysis
- **Modular Architecture**: Clean separation of concerns with extract, transform, and load modules
- **Production Features**: Environment variable management, comprehensive logging, and Docker support
- **Error Resilience**: Robust error handling with detailed logging and graceful failure recovery
- **Scalability**: Batch processing with configurable chunk sizes for large datasets

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.9+ | Core programming language |
| **Pandas** | 2.0+ | Data manipulation and analysis |
| **NumPy** | 1.24+ | Numerical computations |
| **SQLAlchemy** | 2.0+ | ORM and database operations |
| **PostgreSQL** | 15+ | Relational database |
| **Requests** | 2.31+ | HTTP API calls |
| **python-dotenv** | 1.0+ | Environment variable management |
| **Docker** | Latest | Containerization |
| **PyArrow** | 12.0+ | Parquet file support |

## 🧪 Testing

This project includes comprehensive unit tests with 62% code coverage:

- **106 test cases** covering extraction, transformation, loading, data quality, alerts, and predictions
- **Mocked external dependencies** (API calls, database)
- **Automated CI/CD** with GitHub Actions
- **Coverage reports** generated with pytest-cov

### Run Tests Locally
```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage report in browser
open htmlcov/index.html

# Run specific test file
pytest tests/test_extract.py -v
```

## 🔍 Data Quality

This project includes automated data quality checks using Great Expectations:

### Quality Checks

- **Column Validation**: Ensures all required columns exist
- **Null Checks**: Validates critical fields have no missing data
- **Range Validation**: Prices between $0-$10,000, volumes > 0
- **Logical Consistency**: High >= Low, unique symbol+date combinations
- **Outlier Detection**: Daily changes within reasonable bounds (-50% to +50%)

### Validation Reports

Data quality reports are automatically generated in `logs/data_quality_*.txt`

## ⚡ Performance

The pipeline instruments each stage with `perf_counter` (and optional cProfile) and writes throughput reports to `logs/perf_*.log`. LOAD batch size is tunable via `LOAD_BATCH_SIZE` (default **2000**).

**Measured LOAD benchmark** (2,500 rows, median of 3 runs, local Docker PostgreSQL — see `logs/benchmark_batch_size_20260611_195824.txt`):

| Batch size | Median LOAD time | Throughput |
|------------|------------------|------------|
| 500 | 0.959s | 2,606 rec/s |
| 1000 | 0.953s | 2,623 rec/s |
| **2000** (default) | **0.987s** | **2,534 rec/s** |
| 5000 | 0.939s | 2,662 rec/s |

Peak throughput at this scale was chunksize **5000** (0.939s). Default **2000** was selected to balance throughput with fewer DB round-trips on larger production loads (2,900+ rows). vs. the prior **1000**-row default, median LOAD time differs by **3.6%** at 2,500 rows — within run variance; the larger win is reducing round-trips as data grows. Full bottleneck analysis, raw run timings, and resume bullet in **[PERFORMANCE.md](PERFORMANCE.md)**.

```bash
source venv/bin/activate
docker compose up -d                        # Postgres must be running

python scripts/benchmark_batch_size.py      # writes to logs/
python src/pipeline.py                      # writes perf log to logs/
ENABLE_CPROFILE=1 python src/pipeline.py    # also writes profile report to logs/

ls -t logs/
```

## 📊 Data Features

### Extracted Data
- **OHLCV Data**: Open, High, Low, Close prices and Volume
- **Time Series**: Daily stock price data with full historical records
- **Metadata**: Company information, last refreshed timestamps, timezone data

### Calculated Features
- **Price Metrics**: Daily change, daily change percentage, price range
- **Technical Indicators**: 5-day and 20-day moving averages
- **Volatility Analysis**: Rolling volatility indicators with categorization (Very Low to Very High)
- **Volume Analysis**: Volume categorization (Low, Medium, High, Very High) using quartiles
- **Date Components**: Year, month, quarter, day of week, week of year
- **Boolean Indicators**: Positive/negative day flags
- **Price vs MA**: Percentage difference between current price and moving averages

## 📂 Project Structure

```
stock-market-etl-pipeline/
├── src/
│   ├── extract.py               # Alpha Vantage API data extraction
│   ├── transform.py             # Data cleaning and transformation
│   ├── load.py                  # PostgreSQL database operations
│   ├── pipeline.py              # Main ETL pipeline orchestrator
│   ├── data_quality.py          # Great Expectations validation
│   ├── predictions.py           # MA-based price predictions & signals
│   ├── alerts.py                # Email notifications
│   ├── slack_alerts.py          # Slack notifications
│   └── performance.py           # Stage timing & throughput metrics
├── scripts/
│   └── benchmark_batch_size.py  # LOAD batch-size benchmark
├── dashboard/
│   ├── app.py                   # Streamlit dashboard (main)
│   └── requirements.txt         # Dashboard dependencies
├── tests/                       # Unit tests (extract, transform, load)
├── .github/workflows/
│   └── tests.yml                # GitHub Actions CI
├── logs/                        # Pipeline, scheduler, and DQ reports
├── scheduler.py                 # Daily 9:00 AM pipeline scheduler
├── keep_alive.py                # Database keep-alive pings
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment variables
├── docker-compose.yml           # PostgreSQL via Docker
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- PostgreSQL 15 or higher (or Docker for containerized setup)
- Alpha Vantage API key ([Get one here](https://www.alphavantage.co/support/#api-key))
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Narahari2364/stock-market-etl-pipeline.git
   cd stock-market-etl-pipeline
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file with your credentials:
   ```env
   ALPHA_VANTAGE_API_KEY=your_api_key_here
   DATABASE_URL=postgresql://dataengineer:password123@localhost:5432/stock_db
   ```

5. **Start PostgreSQL database (using Docker)**
   ```bash
   docker-compose up -d
   ```

### Running the Dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Set `DATABASE_URL` in `.env` (local) or Streamlit Cloud **Secrets** for live data. The dashboard falls back to sample data if the database is unavailable.

### Running the Pipeline

**Interactive Mode** (Recommended for first-time users):
```bash
python src/pipeline.py
```

The pipeline will:
- Display API rate limit warnings
- Allow you to customize stock symbols
- Show estimated completion time
- Execute the complete ETL process
- Display database summary statistics

**Example Output:**
```
⚠️  API RATE LIMIT WARNING
Alpha Vantage API has a rate limit of 5 calls per minute.
For 17+ default stocks: Estimated time: ~5-6 minutes

Enter stock symbols (comma-separated) or press Enter for default: 
Proceed with ETL pipeline? (y/n): y

🚀 Starting ETL pipeline...
```

**Default Stocks**: The pipeline processes 17+ major stocks by default. See **Stock Coverage** below for the full list.

## ⚠️ API Rate Limits

Alpha Vantage free tier has the following rate limits:
- **5 API calls per minute**
- **500 API calls per day**

The pipeline automatically handles rate limits by:
- Implementing 12-second delays between API calls
- Providing clear warnings about estimated completion times
- Gracefully handling rate limit errors with informative messages

**Recommendations:**
- Use the default 17+ stocks for testing
- For production, consider upgrading to a premium API key
- Monitor your daily API call usage
- Implement caching for frequently accessed data

## 📊 Database Schema

The pipeline creates a `stock_data` table with the following key columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `symbol` | VARCHAR(10) | Stock symbol (indexed) |
| `date` | DATE | Trading date (indexed) |
| `open` | FLOAT | Opening price |
| `high` | FLOAT | Highest price |
| `low` | FLOAT | Lowest price |
| `close` | FLOAT | Closing price |
| `volume` | BIGINT | Trading volume |
| `daily_change` | FLOAT | Absolute daily price change |
| `daily_change_percent` | FLOAT | Percentage daily change |
| `price_range` | FLOAT | High - Low price range |
| `price_range_percent` | FLOAT | Price range as percentage |
| `volatility_indicator` | FLOAT | Rolling volatility metric |
| `volatility_category` | VARCHAR(20) | Volatility category (Very Low to Very High) |
| `volume_category` | VARCHAR(20) | Volume category (Low to Very High) |
| `ma_5` | FLOAT | 5-day moving average |
| `ma_20` | FLOAT | 20-day moving average |
| `price_vs_ma5` | FLOAT | Price vs MA5 percentage difference |
| `price_vs_ma20` | FLOAT | Price vs MA20 percentage difference |
| `is_positive_day` | BOOLEAN | True if price increased |
| `is_negative_day` | BOOLEAN | True if price decreased |
| `year`, `month`, `quarter` | INTEGER | Date components |
| `day_of_week`, `week_of_year` | INTEGER | Time components |
| `extracted_at` | DATETIME | Data extraction timestamp |
| `data_source` | VARCHAR(50) | Data source identifier |

**Indexes:**
- Primary key on `id`
- Index on `symbol`
- Index on `date`
- Composite index on `(symbol, date)` for query optimization — dashboard and prediction queries filter by symbol and date range together, so this index avoids full-table scans on the largest queries

## 🔄 Pipeline Flow

The ETL pipeline follows these 5 main steps:

1. **📥 EXTRACT**
   - Fetches stock data from Alpha Vantage API
   - Handles rate limits with configurable delays
   - Validates API responses and handles errors
   - Returns structured data dictionaries

2. **🔄 TRANSFORM**
   - Converts raw API data to pandas DataFrame
   - Performs data quality checks (nulls, outliers, duplicates)
   - Validates price logic (high >= low, etc.)
   - Calculates derived features and technical indicators
   - Standardizes data format and column ordering

3. **🔍 VALIDATE**
   - Runs Great Expectations checks on transformed data
   - Saves validation reports to `logs/data_quality_*.txt`
   - Sends email/Slack alerts when quality falls below threshold

4. **💾 LOAD**
   - Creates database tables if they don't exist
   - Loads transformed data to PostgreSQL in batches
   - Uses efficient batch processing (default **2000** records per chunk via `LOAD_BATCH_SIZE`)
   - Tracks record counts and loading progress

5. **📊 SUMMARY**
   - Retrieves database statistics
   - Displays total records, unique symbols, date ranges
   - Shows average metrics (close price, volume, daily change)
   - Lists all symbols in the database

## 📈 Stock Coverage

The pipeline tracks **17+ major stocks** across different sectors (symbols that successfully load):

**Technology** (5 stocks)
- AAPL (Apple), MSFT (Microsoft), GOOGL (Google), AMZN (Amazon), NVDA (NVIDIA)

**Financial Services** (3 stocks)
- JPM (JP Morgan), V (Visa), MA (Mastercard)

**Healthcare** (3 stocks)
- JNJ (Johnson & Johnson), UNH (UnitedHealth), PFE (Pfizer)

**Consumer Goods** (3 stocks)
- WMT (Walmart), PG (Procter & Gamble), KO (Coca-Cola)

**Energy** (2 stocks)
- XOM (Exxon Mobil), CVX (Chevron)

**Industrials** (1 stock)
- CAT (Caterpillar)

You can customize the stock list when running the pipeline interactively. With default symbols and ~100 days of data per symbol, expect **2900+ records** in the database.

## ⏰ Automated Scheduling

The pipeline runs automatically every day at 9:00 AM using a Python-based scheduler.

### Start the Scheduler
```bash
# Option 1: Run in foreground (keep terminal open)
python3 scheduler.py

# Option 2: Run in background
./start_scheduler.sh

# Option 3: Run with screen (recommended for servers)
screen -S etl-scheduler
python3 scheduler.py
# Press Ctrl+A then D to detach
```

### Manage Scheduler
```bash
# Check status
./check_scheduler.sh

# View logs
tail -f logs/scheduler.log

# Stop scheduler
./stop_scheduler.sh
```

### Customize Schedule

Edit `scheduler.py` to change the schedule:
- Daily at specific time: `schedule.every().day.at("09:00").do(run_pipeline)`
- Every X hours: `schedule.every(6).hours.do(run_pipeline)`
- Specific days: `schedule.every().monday.at("09:00").do(run_pipeline)`

## 🎯 Future Enhancements

- [ ] **Apache Airflow Integration**: Schedule and orchestrate pipeline runs with DAGs
- [ ] **Kafka Integration**: Real-time data streaming with Kafka producers/consumers
- [ ] **Advanced ML models**: Beyond MA crossover (e.g. time-series forecasting)
- [ ] **Cloud Deployment**: Deploy pipeline and dashboard to AWS/GCP/Azure
- [ ] **Multi-Source Integration**: Additional data sources (Yahoo Finance, IEX Cloud)
- [ ] **Incremental Loading**: Delta/incremental data loading strategies
- [ ] **Data Lineage Tracking**: Track data transformations and dependencies
## 👤 Author

**Narahari Bheemaganapalli**
- GitHub: [@Narahari2364](https://github.com/Narahari2364)
- Email: [naraharibn2364@gmail.com](mailto:naraharibn2364@gmail.com)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This project is for educational and portfolio purposes. Always respect API terms of service and rate limits when working with financial data APIs.
