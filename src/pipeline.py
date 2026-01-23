"""
Main ETL pipeline orchestrator.
"""

import os
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from extract import StockDataExtractor
from transform import StockDataTransformer
from load import StockDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class StockETLPipeline:
    """Main ETL pipeline for stock market data."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 database_url: Optional[str] = None,
                 output_dir: str = 'data'):
        """
        Initialize the ETL pipeline.
        
        Args:
            api_key: API key for stock data provider
            database_url: Database connection URL
            output_dir: Directory for output files
        """
        self.extractor = StockDataExtractor(api_key=api_key)
        self.transformer = StockDataTransformer()
        self.loader = StockDataLoader(database_url=database_url)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path('logs').mkdir(parents=True, exist_ok=True)
    
    def run(self, symbols: List[str], 
            output_format: str = 'csv',
            load_to_db: bool = True) -> dict:
        """
        Run the complete ETL pipeline.
        
        Args:
            symbols: List of stock symbols to process
            output_format: Output file format ('csv' or 'parquet')
            load_to_db: Whether to load data to database
            
        Returns:
            Dictionary with pipeline execution results
        """
        results = {
            'start_time': datetime.now().isoformat(),
            'symbols': symbols,
            'successful': [],
            'failed': [],
            'total_rows': 0
        }
        
        logger.info(f"Starting ETL pipeline for {len(symbols)} symbols")
        
        # Create database table if loading to database
        if load_to_db:
            self.loader.create_table_if_not_exists()
        
        for symbol in symbols:
            try:
                logger.info(f"Processing symbol: {symbol}")
                
                # Extract
                raw_data = self.extractor.extract_from_api(symbol)
                
                if not raw_data or 'Error Message' in raw_data:
                    logger.error(f"API error for {symbol}: {raw_data.get('Error Message', 'Unknown error')}")
                    results['failed'].append(symbol)
                    continue
                
                # Transform
                df = self.transformer.transform(raw_data, symbol)
                
                if df.empty:
                    logger.warning(f"No data to load for {symbol}")
                    results['failed'].append(symbol)
                    continue
                
                # Load
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Load to file
                if output_format == 'csv':
                    file_path = f"{self.output_dir}/{symbol}_{timestamp}.csv"
                    self.loader.load_to_csv(df, file_path)
                elif output_format == 'parquet':
                    file_path = f"{self.output_dir}/{symbol}_{timestamp}.parquet"
                    self.loader.load_to_parquet(df, file_path)
                
                # Load to database
                if load_to_db:
                    self.loader.load_to_database(df, if_exists='append')
                
                results['successful'].append(symbol)
                results['total_rows'] += len(df)
                
                logger.info(f"Successfully processed {symbol}: {len(df)} rows")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                results['failed'].append(symbol)
        
        results['end_time'] = datetime.now().isoformat()
        results['duration'] = (
            datetime.fromisoformat(results['end_time']) - 
            datetime.fromisoformat(results['start_time'])
        ).total_seconds()
        
        logger.info(
            f"Pipeline completed: {len(results['successful'])} successful, "
            f"{len(results['failed'])} failed, {results['total_rows']} total rows"
        )
        
        return results


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Market ETL Pipeline')
    parser.add_argument('--symbols', nargs='+', required=True,
                       help='Stock symbols to process (e.g., AAPL MSFT GOOGL)')
    parser.add_argument('--format', choices=['csv', 'parquet'], default='csv',
                       help='Output file format')
    parser.add_argument('--no-db', action='store_true',
                       help='Skip database loading')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = StockETLPipeline(
        api_key=os.getenv('STOCK_API_KEY'),
        database_url=os.getenv('DATABASE_URL')
    )
    
    # Run pipeline
    results = pipeline.run(
        symbols=args.symbols,
        output_format=args.format,
        load_to_db=not args.no_db
    )
    
    # Print summary
    print("\n" + "="*50)
    print("ETL Pipeline Summary")
    print("="*50)
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Total rows: {results['total_rows']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    print("="*50)


if __name__ == '__main__':
    main()

