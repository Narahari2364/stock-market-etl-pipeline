"""
Load module for storing transformed stock market data.
"""

import os
import pandas as pd
from typing import Optional
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class StockDataLoader:
    """Loads transformed stock data into various destinations."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the loader.
        
        Args:
            database_url: Database connection URL (e.g., postgresql://user:pass@host/db)
        """
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.engine = None
        
        if self.database_url:
            try:
                self.engine = create_engine(self.database_url)
                logger.info("Database connection initialized")
            except Exception as e:
                logger.warning(f"Could not initialize database connection: {str(e)}")
    
    def load_to_database(self, df: pd.DataFrame, table_name: str = 'stock_data', 
                        if_exists: str = 'append') -> bool:
        """
        Load DataFrame to database.
        
        Args:
            df: DataFrame to load
            table_name: Target table name
            if_exists: What to do if table exists ('fail', 'replace', 'append')
            
        Returns:
            True if successful, False otherwise
        """
        if self.engine is None:
            logger.error("Database engine not initialized")
            return False
        
        try:
            logger.info(f"Loading {len(df)} rows to table: {table_name}")
            df.to_sql(
                table_name,
                self.engine,
                if_exists=if_exists,
                index=True,
                method='multi',
                chunksize=1000
            )
            logger.info(f"Successfully loaded data to {table_name}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Error loading data to database: {str(e)}")
            return False
    
    def load_to_csv(self, df: pd.DataFrame, file_path: str) -> bool:
        """
        Load DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            logger.info(f"Loading {len(df)} rows to CSV: {file_path}")
            df.to_csv(file_path, index=True)
            logger.info(f"Successfully saved data to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to CSV: {str(e)}")
            return False
    
    def load_to_parquet(self, df: pd.DataFrame, file_path: str) -> bool:
        """
        Load DataFrame to Parquet file.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            logger.info(f"Loading {len(df)} rows to Parquet: {file_path}")
            df.to_parquet(file_path, index=True, engine='pyarrow')
            logger.info(f"Successfully saved data to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to Parquet: {str(e)}")
            return False
    
    def create_table_if_not_exists(self, table_name: str = 'stock_data') -> bool:
        """
        Create database table if it doesn't exist.
        
        Args:
            table_name: Table name to create
            
        Returns:
            True if successful, False otherwise
        """
        if self.engine is None:
            logger.error("Database engine not initialized")
            return False
        
        try:
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
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
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
            
            logger.info(f"Table {table_name} created or already exists")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating table: {str(e)}")
            return False

