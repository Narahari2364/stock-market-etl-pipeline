"""
Main ETL pipeline orchestrator for stock market data.

This module orchestrates the complete ETL pipeline:
- EXTRACT: Fetches stock data from Alpha Vantage API
- TRANSFORM: Cleans and transforms raw data into analysis-ready format
- LOAD: Loads transformed data into PostgreSQL database
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import logging
import time
import cProfile
import pstats
import io
from datetime import datetime
from typing import List, Optional
from pathlib import Path

# Import ETL modules
from extract import fetch_multiple_stocks
from transform import transform_stock_data
from load import load_to_database, get_database_summary, create_tables, get_database_engine

# Import alert modules
from alerts import send_pipeline_success_email, send_pipeline_failure_email
from slack_alerts import send_pipeline_success_slack, send_pipeline_failure_slack

# Import data quality modules
from data_quality import validate_stock_data, save_validation_report

# Import performance instrumentation
from performance import PipelinePerformanceTracker

# Default symbols if none provided
DEFAULT_SYMBOLS = [
    # Tech Giants (7)
    'AAPL',   # Apple
    'MSFT',   # Microsoft
    'GOOGL',  # Alphabet (Google)
    'AMZN',   # Amazon
    'META',   # Meta (Facebook)
    'NVDA',   # NVIDIA
    'TSLA',   # Tesla
    # Financial Services (5)
    'JPM',    # JP Morgan Chase
    'BAC',    # Bank of America
    'GS',     # Goldman Sachs
    'V',      # Visa
    'MA',     # Mastercard
    # Healthcare (4)
    'JNJ',    # Johnson & Johnson
    'UNH',    # UnitedHealth Group
    'PFE',    # Pfizer
    'ABBV',   # AbbVie
    # Consumer Goods (4)
    'WMT',    # Walmart
    'PG',     # Procter & Gamble
    'KO',     # Coca-Cola
    'MCD',    # McDonald's
    # Energy (2)
    'XOM',    # Exxon Mobil
    'CVX',    # Chevron
    # Industrials (2)
    'BA',     # Boeing
    'CAT',    # Caterpillar
    # Entertainment (1)
    'DIS',    # Disney
]


def setup_logging() -> logging.Logger:
    """
    Set up logging to both file and console.
    
    Creates logs/ directory if it doesn't exist and sets up logging with:
    - File handler: logs/pipeline_YYYYMMDD_HHMMSS.log
    - Console handler: stdout
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'pipeline_{timestamp}.log'
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create logger
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"📝 Logging initialized - Log file: {log_file}")
    
    return logger


def _write_profile_report(profiler: cProfile.Profile, perf_timestamp: str, logger: logging.Logger) -> None:
    """Write cProfile cumulative stats to logs/profile_<timestamp>.txt."""
    profile_path = Path("logs") / f"profile_{perf_timestamp}.txt"
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(30)
    with open(profile_path, "w", encoding="utf-8") as handle:
        handle.write(stream.getvalue())
    logger.info(f"🔬 cProfile report saved: {profile_path}")


def run_etl_pipeline(symbols: Optional[List[str]] = None, interval: str = "daily") -> bool:
    """
    Run the complete ETL pipeline: Extract, Transform, Load.
    
    Args:
        symbols: List of stock symbols to process. Default: 25 stocks across sectors (see DEFAULT_SYMBOLS)
        interval: Time interval for data extraction (default: "daily")
    
    Returns:
        True if pipeline completed successfully, False otherwise
    """
    logger = logging.getLogger('pipeline')
    
    # Default symbols if none provided
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    
    # Calculate estimated time (12 seconds per stock + processing time)
    estimated_seconds = len(symbols) * 12 + 30  # 12s per API call + 30s for processing
    estimated_minutes = estimated_seconds // 60
    estimated_secs = estimated_seconds % 60
    
    # Pipeline start banner
    logger.info("")
    logger.info("=" * 80)
    logger.info("🚀 STOCK MARKET ETL PIPELINE - STARTING")
    logger.info("=" * 80)
    logger.info(f"📊 Symbols to process: {', '.join(symbols)}")
    logger.info(f"📈 Total symbols: {len(symbols)}")
    logger.info(f"⏱️  Estimated time: ~{estimated_minutes}m {estimated_secs}s")
    start_time = datetime.now()
    logger.info(f"🕐 Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    logger.info("")

    perf = PipelinePerformanceTracker()
    load_batch_size = int(os.getenv("LOAD_BATCH_SIZE", "2000"))
    logger.info(f"📈 Performance log: {perf.log_file}")
    logger.info(f"📦 Load batch size: {load_batch_size}")

    profile_enabled = os.getenv("ENABLE_CPROFILE", "").lower() in ("1", "true", "yes")
    profiler = cProfile.Profile() if profile_enabled else None
    if profiler:
        profiler.enable()
        logger.info("🔬 cProfile instrumentation enabled (ENABLE_CPROFILE=1)")
    
    try:
        # ============================================
        # STEP 1: EXTRACT
        # ============================================
        logger.info("")
        logger.info("-" * 80)
        logger.info("📥 STEP 1: EXTRACT - Fetching stock data from Alpha Vantage API")
        logger.info("-" * 80)

        extract_start = perf.start_stage()
        try:
            raw_data_list = fetch_multiple_stocks(symbols, interval=interval, delay=12)
            
            # Filter out None values (failed fetches)
            successful_extracts = [data for data in raw_data_list if data is not None]
            failed_count = len(raw_data_list) - len(successful_extracts)
            
            logger.info("")
            logger.info(f"✅ Extraction complete:")
            logger.info(f"   ✓ Successfully extracted: {len(successful_extracts)}/{len(symbols)} symbols")
            if failed_count > 0:
                logger.warning(f"   ✗ Failed extractions: {failed_count} symbols")
            
            if not successful_extracts:
                logger.error("❌ No data was successfully extracted. Pipeline cannot continue.")
                return False
            
            # Log extraction details
            for data in successful_extracts:
                symbol = data.get('symbol', 'UNKNOWN')
                data_points = data.get('data_points', 0)
                logger.info(f"   • {symbol}: {data_points} data points")

            extract_records = sum(
                data.get('data_points', 0) for data in successful_extracts
            )
            perf.end_stage("EXTRACT", extract_start, record_count=extract_records)
            
        except Exception as e:
            logger.error(f"❌ EXTRACT step failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
        # ============================================
        # STEP 2: TRANSFORM
        # ============================================
        logger.info("")
        logger.info("-" * 80)
        logger.info("🔄 STEP 2: TRANSFORM - Cleaning and transforming stock data")
        logger.info("-" * 80)

        transform_start = perf.start_stage()
        try:
            df_transformed = transform_stock_data(raw_data_list)
            
            if df_transformed is None or df_transformed.empty:
                logger.error("❌ Transformation produced empty DataFrame. Pipeline cannot continue.")
                return False
            
            # Log transformation results
            logger.info("")
            logger.info(f"✅ Transformation complete:")
            logger.info(f"   ✓ Rows: {len(df_transformed):,}")
            logger.info(f"   ✓ Columns: {len(df_transformed.columns)}")
            logger.info(f"   ✓ Symbols: {df_transformed['symbol'].nunique()}")
            
            # Date range
            if 'date' in df_transformed.columns:
                min_date = df_transformed['date'].min()
                max_date = df_transformed['date'].max()
                logger.info(f"   ✓ Date range: {min_date} to {max_date}")
            
            # Memory usage
            memory_mb = df_transformed.memory_usage(deep=True).sum() / 1024**2
            logger.info(f"   ✓ Memory usage: {memory_mb:.2f} MB")

            perf.end_stage("TRANSFORM", transform_start, record_count=len(df_transformed))

            logger.info("\n🔍 STEP 2.5: VALIDATE - Running data quality checks...")
            validate_start = perf.start_stage()
            validation_results = validate_stock_data(df_transformed, log_results=True)

            if not validation_results['success']:
                logger.warning(
                    f"⚠️  Data quality below threshold: {validation_results.get('success_rate', 0):.1f}%"
                )

                # Send alert about data quality issues
                try:
                    from alerts import send_data_quality_warning_email
                    from slack_alerts import send_data_quality_warning_slack

                    issues_summary = f"{len(validation_results.get('failed_expectations', []))} checks failed"
                    send_data_quality_warning_email(issues_summary)
                    send_data_quality_warning_slack(issues_summary)
                except Exception:
                    pass

                # Ask user if they want to continue
                logger.warning("⚠️  Data quality issues detected. Pipeline will continue with loading...")
            else:
                logger.info(
                    f"✅ Data quality validation passed: {validation_results.get('success_rate', 0):.1f}%"
                )

            # Save validation report
            save_validation_report(validation_results)
            perf.end_stage(
                "VALIDATE",
                validate_start,
                record_count=len(df_transformed),
                notes=f"success_rate={validation_results.get('success_rate', 0):.1f}%",
            )
            
        except Exception as e:
            logger.error(f"❌ TRANSFORM step failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
        # ============================================
        # STEP 3: LOAD
        # ============================================
        logger.info("")
        logger.info("-" * 80)
        logger.info("💾 STEP 3: LOAD - Loading data to PostgreSQL database")
        logger.info("-" * 80)

        load_start = perf.start_stage()
        try:
            # Ensure database tables exist
            logger.info("Creating database tables if needed...")
            create_tables()
            
            # Load data to database
            logger.info("Loading transformed data to database...")
            load_success = load_to_database(
                df_transformed,
                chunksize=load_batch_size,
                perf_tracker=perf,
            )
            
            if not load_success:
                logger.error("❌ LOAD step failed: Could not load data to database")
                return False

            perf.end_stage(
                "LOAD",
                load_start,
                record_count=len(df_transformed),
                notes=f"chunksize={load_batch_size}",
            )
            
            logger.info("")
            logger.info("✅ Loading complete")
            
        except Exception as e:
            logger.error(f"❌ LOAD step failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
        # ============================================
        # STEP 4: DATABASE SUMMARY
        # ============================================
        logger.info("")
        logger.info("-" * 80)
        logger.info("📊 STEP 4: DATABASE SUMMARY - Retrieving database statistics")
        logger.info("-" * 80)

        summary_start = perf.start_stage()
        try:
            summary = get_database_summary()
            
            if summary:
                logger.info("")
                logger.info("✅ Database Summary:")
                logger.info(f"   📈 Total records: {summary.get('total_records', 0):,}")
                logger.info(f"   🏷️  Unique symbols: {summary.get('unique_symbols', 0)}")
                
                date_range = summary.get('date_range', {})
                if date_range.get('min_date') and date_range.get('max_date'):
                    logger.info(f"   📅 Date range: {date_range['min_date']} to {date_range['max_date']}")
                
                avg_metrics = summary.get('average_metrics', {})
                if avg_metrics:
                    logger.info(f"   💰 Avg close price: ${avg_metrics.get('avg_close', 0):.2f}")
                    logger.info(f"   📊 Avg volume: {avg_metrics.get('avg_volume', 0):,.0f}")
                    logger.info(f"   📉 Avg daily change: {avg_metrics.get('avg_daily_change_percent', 0):.2f}%")
                
                symbols_list = summary.get('symbols_list', [])
                if symbols_list:
                    logger.info(f"   🏢 Symbols in database: {', '.join(symbols_list)}")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not retrieve database summary: {str(e)}")
            # Don't fail the pipeline if summary fails
        finally:
            perf.end_stage("SUMMARY", summary_start, record_count=0)

        perf_summary = perf.finalize(total_records=len(df_transformed))
        logger.info(f"📈 Performance report saved: {perf_summary['log_file']}")
        for stage in perf_summary["stages"]:
            if "pct_of_total" in stage:
                logger.info(
                    f"   ⏱️  {stage['stage']}: {stage['duration_sec']:.2f}s "
                    f"({stage['pct_of_total']:.1f}% of runtime, "
                    f"{stage['records_per_sec']:,.0f} rec/s)"
                )

        # ============================================
        # PIPELINE COMPLETE
        # ============================================
        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()
        duration_minutes = int(duration_seconds // 60)
        duration_secs = int(duration_seconds % 60)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ ETL PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"🕐 End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱️  Total duration: {duration_minutes}m {duration_secs}s")
        logger.info(f"📊 Symbols processed: {len(symbols)}")
        logger.info(f"📈 Records loaded: {len(df_transformed):,}")
        logger.info("=" * 80)
        logger.info("")
        
        # Send success alerts
        try:
            send_pipeline_success_email(
                records_loaded=len(df_transformed),
                symbols_count=len(symbols),
                symbols_list=symbols
            )
            send_pipeline_success_slack(
                records=len(df_transformed),
                symbols_count=len(symbols),
                symbols_list=symbols
            )
        except Exception as e:
            logger.warning(f"Failed to send success alerts: {e}")
        
        return True
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("❌ ETL PIPELINE FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        logger.error("")
        
        # Send failure alerts
        try:
            send_pipeline_failure_email(str(e), step='Pipeline Execution')
            send_pipeline_failure_slack(str(e), step='Pipeline Execution')
        except Exception as alert_error:
            logger.warning(f"Failed to send failure alerts: {alert_error}")
        
        return False
    finally:
        if profiler:
            profiler.disable()
            _write_profile_report(profiler, perf.timestamp, logger)


if __name__ == "__main__":
    """
    Main entry point for the ETL pipeline.
    """
    # Setup logging first
    logger = setup_logging()
    
    # Print warning banner
    print("\n" + "=" * 80)
    print("⚠️  API RATE LIMIT WARNING")
    print("=" * 80)
    print("Alpha Vantage API has a rate limit of 5 calls per minute.")
    print("This pipeline uses a 12-second delay between API calls to respect this limit.")
    print("")
    print("For 25 default stocks:")
    print("  • Estimated time: ~5-6 minutes")
    print("  • Plus additional time for transformation and loading")
    print("")
    print("=" * 80)
    print("")
    
    # Allow customization of symbols
    user_input = input(
        "Enter stock symbols (comma-separated) or press Enter for default (25 stocks): "
    ).strip()

    if user_input:
        # Parse user input
        symbols = [s.strip().upper() for s in user_input.split(',') if s.strip()]
        if not symbols:
            print("⚠️  No valid symbols entered. Using defaults.")
            symbols = DEFAULT_SYMBOLS
        else:
            print(f"✓ Using custom symbols: {', '.join(symbols)}")
    else:
        symbols = DEFAULT_SYMBOLS
        print(f"✓ Using default symbols ({len(symbols)} stocks): {', '.join(symbols)}")
    
    # Calculate and display estimated time
    estimated_seconds = len(symbols) * 12 + 30
    estimated_minutes = estimated_seconds // 60
    estimated_secs = estimated_seconds % 60
    print(f"⏱️  Estimated time: ~{estimated_minutes}m {estimated_secs}s")
    print("")
    
    # Confirm before proceeding
    confirm = input("Proceed with ETL pipeline? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Pipeline cancelled by user.")
        exit(0)
    
    print("")
    print("🚀 Starting ETL pipeline...")
    print("")
    
    # Run the pipeline
    success = run_etl_pipeline(symbols=symbols, interval="daily")
    
    # Exit with appropriate code
    if success:
        print("")
        print("✅ Pipeline completed successfully!")
        exit(0)
    else:
        print("")
        print("❌ Pipeline failed. Check logs for details.")
        exit(1)
