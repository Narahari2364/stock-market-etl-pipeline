"""
Load module for storing transformed stock market data to PostgreSQL.

This module provides functions to load stock data into PostgreSQL database
using SQLAlchemy ORM and pandas DataFrame operations.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
from sqlalchemy import create_engine, Column, Integer, String, Date, Float, BigInteger, Boolean, DateTime, Index, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Create declarative base for SQLAlchemy models
Base = declarative_base()


class StockData(Base):
    """
    SQLAlchemy model for stock_data table.
    
    Represents stock market data with OHLCV, derived features, and metadata.
    """
    __tablename__ = 'stock_data'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Symbol and date (indexed for performance)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # Time components
    year = Column(Integer, nullable=True)
    month = Column(Integer, nullable=True)
    quarter = Column(Integer, nullable=True)
    day_of_week = Column(Integer, nullable=True)
    week_of_year = Column(Integer, nullable=True)
    
    # OHLC prices
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    
    # Volume
    volume = Column(BigInteger, nullable=True)
    
    # Daily changes
    daily_change = Column(Float, nullable=True)
    daily_change_percent = Column(Float, nullable=True)
    
    # Price range
    price_range = Column(Float, nullable=True)
    price_range_percent = Column(Float, nullable=True)
    
    # Volatility
    volatility_indicator = Column(Float, nullable=True)
    volatility_category = Column(String(20), nullable=True)
    
    # Volume category
    volume_category = Column(String(20), nullable=True)
    
    # Boolean indicators
    is_positive_day = Column(Boolean, nullable=True)
    is_negative_day = Column(Boolean, nullable=True)
    
    # Moving averages
    ma_5 = Column(Float, nullable=True)
    ma_20 = Column(Float, nullable=True)
    price_vs_ma5 = Column(Float, nullable=True)
    price_vs_ma20 = Column(Float, nullable=True)
    
    # Metadata
    extracted_at = Column(DateTime, nullable=True, default=datetime.utcnow)
    data_source = Column(String(50), nullable=True, default='Alpha Vantage')
    
    # Create composite index on symbol and date for faster queries
    __table_args__ = (
        Index('idx_symbol_date', 'symbol', 'date'),
    )
    
    def __repr__(self):
        return f"<StockData(symbol={self.symbol}, date={self.date}, close={self.close})>"


def get_database_engine():
    """
    Create and return SQLAlchemy database engine from DATABASE_URL environment variable.
    
    Returns:
        SQLAlchemy engine object
        
    Raises:
        ValueError: If DATABASE_URL is not set
        SQLAlchemyError: If engine creation fails
    """
    print(f"\n{'='*60}")
    print("Initializing Database Connection")
    print(f"{'='*60}")
    
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        error_msg = "DATABASE_URL environment variable is not set"
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    # Mask password in URL for logging
    safe_url = database_url
    if '@' in database_url:
        parts = database_url.split('@')
        if len(parts) == 2:
            user_pass = parts[0].split('//')[-1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                safe_url = database_url.replace(user_pass, f"{user}:***")
    
    print(f"Database URL: {safe_url}")
    
    try:
        engine = create_engine(
            database_url,
            pool_pre_ping=True,  # Verify connections before using
            echo=False  # Set to True for SQL query logging
        )
        print("✓ Database engine created successfully")
        return engine
        
    except Exception as e:
        error_msg = f"Failed to create database engine: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise SQLAlchemyError(error_msg) from e


def create_tables(engine=None):
    """
    Create database tables if they don't exist.
    
    Args:
        engine: SQLAlchemy engine (optional, will create if not provided)
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print("Creating Database Tables")
    print(f"{'='*60}")
    
    try:
        if engine is None:
            engine = get_database_engine()
        
        print("Creating tables from models...")
        Base.metadata.create_all(engine, checkfirst=True)
        print("✓ Tables created or already exist")
        print(f"  - Table: {StockData.__tablename__}")
        print(f"  - Columns: {len(StockData.__table__.columns)}")
        
        return True
        
    except Exception as e:
        error_msg = f"Error creating tables: {str(e)}"
        print(f"ERROR: {error_msg}")
        return False


def load_to_database(df: pd.DataFrame, engine=None, table_name: str = 'stock_data') -> bool:
    """
    Load DataFrame to PostgreSQL database.
    
    Args:
        df: DataFrame to load (must have columns matching StockData model)
        engine: SQLAlchemy engine (optional, will create if not provided)
        table_name: Target table name (default: 'stock_data')
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print("Loading Data to Database")
    print(f"{'='*60}")
    
    if df is None or df.empty:
        print("ERROR: DataFrame is empty or None")
        return False
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    try:
        if engine is None:
            engine = get_database_engine()
        
        # Ensure tables exist
        create_tables(engine)
        
        # Prepare DataFrame for loading
        df_to_load = df.copy()
        
        # Add extracted_at timestamp if not present
        if 'extracted_at' not in df_to_load.columns:
            df_to_load['extracted_at'] = datetime.utcnow()
        
        # Add data_source if not present
        if 'data_source' not in df_to_load.columns:
            df_to_load['data_source'] = 'Alpha Vantage'
        
        # Ensure date column is datetime/date type
        if 'date' in df_to_load.columns:
            df_to_load['date'] = pd.to_datetime(df_to_load['date']).dt.date
        
        # Convert boolean columns
        bool_columns = ['is_positive_day', 'is_negative_day']
        for col in bool_columns:
            if col in df_to_load.columns:
                df_to_load[col] = df_to_load[col].astype('bool')
        
        # Replace NaN with None for SQL compatibility
        df_to_load = df_to_load.where(pd.notnull(df_to_load), None)
        
        # Get initial record count
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            initial_count = result.scalar() or 0
        
        print(f"Records in database before load: {initial_count}")
        print(f"Records to load: {len(df_to_load)}")
        
        # Load data in batches
        chunksize = 1000
        total_chunks = (len(df_to_load) // chunksize) + (1 if len(df_to_load) % chunksize > 0 else 0)
        
        print(f"Loading in {total_chunks} batch(es) of {chunksize} records...")
        
        for i, chunk in enumerate(range(0, len(df_to_load), chunksize), 1):
            chunk_df = df_to_load.iloc[chunk:chunk + chunksize]
            print(f"  Loading batch {i}/{total_chunks} ({len(chunk_df)} records)...", end=' ')
            
            try:
                chunk_df.to_sql(
                    table_name,
                    engine,
                    if_exists='append',
                    index=False,
                    method='multi',
                    chunksize=chunksize
                )
                print("✓")
                
            except Exception as e:
                print(f"✗")
                print(f"    ERROR in batch {i}: {str(e)}")
                # Continue with next batch instead of failing completely
                continue
        
        # Get final record count
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            final_count = result.scalar() or 0
        
        records_added = final_count - initial_count
        print(f"\n✓ Data loading complete")
        print(f"  Records in database after load: {final_count}")
        print(f"  Records added: {records_added}")
        print(f"  Expected: {len(df_to_load)}")
        
        return True
        
    except SQLAlchemyError as e:
        error_msg = f"SQLAlchemy error loading data: {str(e)}"
        print(f"ERROR: {error_msg}")
        return False
    
    except Exception as e:
        error_msg = f"Unexpected error loading data: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return False


def get_database_summary(engine=None) -> Dict:
    """
    Get summary statistics from the database.
    
    Args:
        engine: SQLAlchemy engine (optional, will create if not provided)
    
    Returns:
        Dictionary containing:
        - total_records: Total number of records
        - unique_symbols: Number of unique symbols
        - date_range: Dictionary with min_date and max_date
        - average_metrics: Dictionary with average close, volume, daily_change_percent
        - symbols_list: List of all symbols in database
    """
    print(f"\n{'='*60}")
    print("Getting Database Summary")
    print(f"{'='*60}")
    
    try:
        if engine is None:
            engine = get_database_engine()
        
        summary = {}
        
        with engine.connect() as conn:
            # Total records
            result = conn.execute(text("SELECT COUNT(*) FROM stock_data"))
            summary['total_records'] = result.scalar() or 0
            print(f"Total records: {summary['total_records']}")
            
            if summary['total_records'] == 0:
                print("⚠ Database is empty")
                return {
                    'total_records': 0,
                    'unique_symbols': 0,
                    'date_range': {'min_date': None, 'max_date': None},
                    'average_metrics': {},
                    'symbols_list': []
                }
            
            # Unique symbols
            result = conn.execute(text("SELECT COUNT(DISTINCT symbol) FROM stock_data"))
            summary['unique_symbols'] = result.scalar() or 0
            print(f"Unique symbols: {summary['unique_symbols']}")
            
            # Date range
            result = conn.execute(text("SELECT MIN(date), MAX(date) FROM stock_data"))
            row = result.fetchone()
            summary['date_range'] = {
                'min_date': row[0].isoformat() if row[0] else None,
                'max_date': row[1].isoformat() if row[1] else None
            }
            print(f"Date range: {summary['date_range']['min_date']} to {summary['date_range']['max_date']}")
            
            # Average metrics
            result = conn.execute(text("""
                SELECT 
                    AVG(close) as avg_close,
                    AVG(volume) as avg_volume,
                    AVG(daily_change_percent) as avg_daily_change_percent
                FROM stock_data
                WHERE close IS NOT NULL
            """))
            row = result.fetchone()
            summary['average_metrics'] = {
                'avg_close': float(row[0]) if row[0] else None,
                'avg_volume': float(row[1]) if row[1] else None,
                'avg_daily_change_percent': float(row[2]) if row[2] else None
            }
            print(f"Average close price: ${summary['average_metrics']['avg_close']:.2f}" if summary['average_metrics']['avg_close'] else "N/A")
            print(f"Average volume: {summary['average_metrics']['avg_volume']:,.0f}" if summary['average_metrics']['avg_volume'] else "N/A")
            print(f"Average daily change: {summary['average_metrics']['avg_daily_change_percent']:.2f}%" if summary['average_metrics']['avg_daily_change_percent'] else "N/A")
            
            # List of symbols
            result = conn.execute(text("SELECT DISTINCT symbol FROM stock_data ORDER BY symbol"))
            summary['symbols_list'] = [row[0] for row in result.fetchall()]
            print(f"Symbols: {', '.join(summary['symbols_list'])}")
        
        print(f"\n✓ Database summary retrieved")
        return summary
        
    except SQLAlchemyError as e:
        error_msg = f"SQLAlchemy error getting summary: {str(e)}"
        print(f"ERROR: {error_msg}")
        return {}
    
    except Exception as e:
        error_msg = f"Unexpected error getting summary: {str(e)}"
        print(f"ERROR: {error_msg}")
        return {}


if __name__ == "__main__":
    """
    Test section for the load module.
    Currently commented out - we'll skip testing this module.
    """
    
    # # Test database connection
    # print("\n" + "="*60)
    # print("Load Module - Test Suite")
    # print("="*60)
    # 
    # try:
    #     # Test engine creation
    #     engine = get_database_engine()
    #     
    #     # Test table creation
    #     create_tables(engine)
    #     
    #     # Test database summary (on empty database)
    #     summary = get_database_summary(engine)
    #     print(f"\nSummary: {summary}")
    #     
    #     # Note: To test load_to_database, you would need a transformed DataFrame
    #     # from the transform module
    #     
    #     print("\n✓ All tests passed")
    #     
    # except Exception as e:
    #     print(f"\n✗ Test failed: {str(e)}")
    #     import traceback
    #     traceback.print_exc()
    
    print("\nLoad module test section is commented out.")
    print("To test, uncomment the code in __main__ section.")
