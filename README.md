# Stock Market ETL Pipeline

A complete ETL (Extract, Transform, Load) pipeline for processing stock market data from various sources.

## Features

- **Extract**: Fetch stock market data from APIs (e.g., Alpha Vantage) or CSV files
- **Transform**: Clean, standardize, and enrich data with technical indicators
- **Load**: Store data in databases (PostgreSQL) or file formats (CSV, Parquet)
- **Pipeline**: Orchestrated ETL workflow with logging and error handling

## Project Structure

```
stock-market-etl-pipeline/
├── src/
│   ├── __init__.py
│   ├── extract.py      # Data extraction module
│   ├── transform.py    # Data transformation module
│   ├── load.py         # Data loading module
│   └── pipeline.py     # Main ETL pipeline orchestrator
├── data/               # Output data files
├── logs/               # Pipeline logs
├── config/             # Configuration files
├── tests/              # Unit tests
├── dashboard/          # Dashboard files (future)
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in git)
├── .env.example       # Example environment variables
├── .gitignore         # Git ignore rules
├── docker-compose.yml  # Docker configuration
└── README.md          # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock-market-etl-pipeline
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and database credentials
```

## Configuration

Edit the `.env` file with your configuration:

- `STOCK_API_KEY`: Your API key for stock data provider (e.g., Alpha Vantage)
- `STOCK_API_URL`: API endpoint URL
- `DATABASE_URL`: PostgreSQL connection string (format: `postgresql://user:password@host:port/database`)

## Usage

### Command Line

Run the pipeline for specific stock symbols:

```bash
python -m src.pipeline --symbols AAPL MSFT GOOGL --format csv
```

Options:
- `--symbols`: One or more stock symbols (required)
- `--format`: Output format (`csv` or `parquet`, default: `csv`)
- `--no-db`: Skip database loading

### Python API

```python
from src.pipeline import StockETLPipeline

# Initialize pipeline
pipeline = StockETLPipeline(
    api_key='your_api_key',
    database_url='postgresql://user:pass@localhost/db'
)

# Run pipeline
results = pipeline.run(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    output_format='csv',
    load_to_db=True
)
```

## Data Processing

The pipeline performs the following transformations:

1. **Data Cleaning**: Removes null values and outliers
2. **Standardization**: Normalizes column names and formats
3. **Enrichment**: Adds calculated fields:
   - Daily returns
   - Simple Moving Averages (SMA 20, SMA 50)
   - Volatility (rolling standard deviation)
   - High-Low spread

## Database Schema

The pipeline creates a table with the following schema:

```sql
CREATE TABLE stock_data (
    date TIMESTAMP NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    daily_return DECIMAL(10, 6),
    sma_20 DECIMAL(10, 2),
    sma_50 DECIMAL(10, 2),
    volatility DECIMAL(10, 6),
    hl_spread DECIMAL(10, 2),
    PRIMARY KEY (date, symbol)
);
```

## Docker

Run the pipeline using Docker Compose:

```bash
docker-compose up
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Logging

Pipeline logs are written to:
- Console output
- `logs/pipeline.log` file

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

