# 📈 Stock Market ETL Pipeline

![Tests](https://github.com/YOUR_USERNAME/stock-market-etl-pipeline/workflows/Tests/badge.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Coverage](https://img.shields.io/badge/coverage-41%25-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

*(Replace YOUR_USERNAME with your actual GitHub username)*

A production-ready ETL (Extract, Transform, Load) pipeline for processing stock market data from Alpha Vantage API into PostgreSQL database. This project demonstrates end-to-end data engineering practices including API integration, data transformation, database operations, and comprehensive error handling.

## 📊 Project Stats

- **2900+** stock records across **17 symbols**
- **100%** data quality validation pass rate
- **25** comprehensive unit tests
- **41%** code coverage
- **5** automated alerts/notifications
- **Live dashboard** deployed on Streamlit Cloud
- **Daily automated** data updates

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

This project includes comprehensive unit tests with 41% code coverage:

- **25 test cases** covering extraction, transformation, and loading
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
│   ├── __init__.py              # Package initialization
│   ├── extract.py               # Alpha Vantage API data extraction
│   ├── transform.py             # Data cleaning and transformation
│   ├── load.py                  # PostgreSQL database operations
│   └── pipeline.py              # Main ETL pipeline orchestrator
├── data/                        # Output data files (CSV, Parquet, JSON)
├── logs/                        # Pipeline execution logs
├── config/                      # Configuration files
├── tests/                       # Unit and integration tests
├── dashboard/                   # Dashboard files (future)
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (gitignored)
├── .env.example                 # Example environment variables
├── .gitignore                   # Git ignore rules
├── docker-compose.yml           # Docker Compose configuration
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
   git clone <repository-url>
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

### Running the Pipeline

**Interactive Mode** (Recommended for first-time users):
```bash
python -m src.pipeline
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
For 5 default stocks: Estimated time: ~60 seconds

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
- Composite index on `(symbol, date)` for query optimization

## 🔄 Pipeline Flow

The ETL pipeline follows these 4 main steps:

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

3. **💾 LOAD**
   - Creates database tables if they don't exist
   - Loads transformed data to PostgreSQL in batches
   - Uses efficient batch processing (1000 records per chunk)
   - Tracks record counts and loading progress

4. **📊 SUMMARY**
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
- [ ] **Streamlit Dashboard**: Interactive web dashboard for data visualization and analysis
- [ ] **Kafka Integration**: Real-time data streaming with Kafka producers/consumers
- [ ] **Machine Learning Models**: Predictive models for stock price forecasting
- [ ] **Cloud Deployment**: Deploy to AWS/GCP/Azure with serverless functions
- [ ] **Comprehensive Testing**: Unit tests, integration tests, and end-to-end test suite
- [ ] **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
- [ ] **Data Validation Framework**: Great Expectations or similar for data quality checks
- [ ] **Multi-Source Integration**: Support for additional data sources (Yahoo Finance, IEX Cloud)
- [ ] **Incremental Loading**: Delta/incremental data loading strategies
- [ ] **Data Lineage Tracking**: Track data transformations and dependencies
- [ ] **Performance Monitoring**: Metrics and alerting for pipeline health

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This project is for educational and portfolio purposes. Always respect API terms of service and rate limits when working with financial data APIs.
