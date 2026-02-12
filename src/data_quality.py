import pandas as pd
import great_expectations as gx
from great_expectations.core import ExpectationSuite
from datetime import datetime
import os


def create_stock_data_expectations():
    """
    Define data quality expectations for stock data
    
    Returns:
        list: List of expectations
    """
    expectations = []
    
    # Column existence checks
    expectations.extend([
        {
            'expectation_type': 'expect_column_to_exist',
            'kwargs': {'column': 'symbol'}
        },
        {
            'expectation_type': 'expect_column_to_exist',
            'kwargs': {'column': 'date'}
        },
        {
            'expectation_type': 'expect_column_to_exist',
            'kwargs': {'column': 'close'}
        },
        {
            'expectation_type': 'expect_column_to_exist',
            'kwargs': {'column': 'volume'}
        }
    ])
    
    # Data type validations
    expectations.extend([
        {
            'expectation_type': 'expect_column_values_to_not_be_null',
            'kwargs': {'column': 'symbol'}
        },
        {
            'expectation_type': 'expect_column_values_to_not_be_null',
            'kwargs': {'column': 'date'}
        },
        {
            'expectation_type': 'expect_column_values_to_not_be_null',
            'kwargs': {'column': 'close'}
        }
    ])
    
    # Value range validations
    expectations.extend([
        {
            'expectation_type': 'expect_column_values_to_be_between',
            'kwargs': {
                'column': 'open',
                'min_value': 0,
                'max_value': 10000,
                'mostly': 1.0
            }
        },
        {
            'expectation_type': 'expect_column_values_to_be_between',
            'kwargs': {
                'column': 'high',
                'min_value': 0,
                'max_value': 10000,
                'mostly': 1.0
            }
        },
        {
            'expectation_type': 'expect_column_values_to_be_between',
            'kwargs': {
                'column': 'low',
                'min_value': 0,
                'max_value': 10000,
                'mostly': 1.0
            }
        },
        {
            'expectation_type': 'expect_column_values_to_be_between',
            'kwargs': {
                'column': 'close',
                'min_value': 0,
                'max_value': 10000,
                'mostly': 1.0
            }
        },
        {
            'expectation_type': 'expect_column_values_to_be_between',
            'kwargs': {
                'column': 'volume',
                'min_value': 0,
                'mostly': 1.0
            }
        },
        {
            'expectation_type': 'expect_column_values_to_be_between',
            'kwargs': {
                'column': 'daily_change_percent',
                'min_value': -50,
                'max_value': 50,
                'mostly': 0.95  # Allow 5% outliers
            }
        }
    ])
    
    # Logical consistency checks
    expectations.extend([
        {
            'expectation_type': 'expect_column_pair_values_A_to_be_greater_than_B',
            'kwargs': {
                'column_A': 'high',
                'column_B': 'low',
                'mostly': 1.0
            }
        },
        {
            'expectation_type': 'expect_compound_columns_to_be_unique',
            'kwargs': {
                'column_list': ['symbol', 'date']
            }
        }
    ])
    
    return expectations


def validate_stock_data(df, log_results=True):
    """
    Validate stock data against quality expectations
    
    Args:
        df (pd.DataFrame): Stock data to validate
        log_results (bool): Whether to print results
        
    Returns:
        dict: Validation results with success rate and failed checks
    """
    
    if log_results:
        print("\n" + "=" * 70)
        print("🔍 RUNNING DATA QUALITY CHECKS")
        print("=" * 70)
    
    try:
        # Create Great Expectations context
        context = gx.get_context()
        
        # Convert DataFrame to GX Batch
        datasource = context.sources.add_or_update_pandas(name="pandas_datasource")
        data_asset = datasource.add_dataframe_asset(name="stock_data")
        batch_request = data_asset.build_batch_request(dataframe=df)
        
        # Create validator
        expectation_suite_name = "stock_data_suite"
        context.add_or_update_expectation_suite(expectation_suite_name=expectation_suite_name)
        
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=expectation_suite_name
        )
        
        # Add expectations
        expectations = create_stock_data_expectations()
        
        for exp in expectations:
            expectation_type = exp['expectation_type']
            kwargs = exp['kwargs']
            
            # Dynamically call the expectation method
            expectation_method = getattr(validator, expectation_type)
            expectation_method(**kwargs)
        
        # Run validation
        results = validator.validate()
        
        # Parse results
        success_count = results.statistics['successful_expectations']
        total_count = results.statistics['evaluated_expectations']
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        failed_expectations = []
        for result in results.results:
            if not result.success:
                failed_expectations.append({
                    'expectation': result.expectation_config.expectation_type,
                    'column': result.expectation_config.kwargs.get('column', 'N/A'),
                    'details': str(result.result)[:200]
                })
        
        # Log results
        if log_results:
            print(f"\n📊 Validation Results:")
            print(f"   Total Checks: {total_count}")
            print(f"   Passed: {success_count}")
            print(f"   Failed: {total_count - success_count}")
            print(f"   Success Rate: {success_rate:.1f}%")
            
            if success_rate >= 95:
                print(f"   ✅ Data quality: EXCELLENT")
            elif success_rate >= 90:
                print(f"   ⚠️  Data quality: GOOD (some issues detected)")
            elif success_rate >= 80:
                print(f"   ⚠️  Data quality: FAIR (multiple issues detected)")
            else:
                print(f"   ❌ Data quality: POOR (significant issues detected)")
            
            if failed_expectations:
                print(f"\n❌ Failed Checks:")
                for i, failure in enumerate(failed_expectations[:5], 1):
                    print(f"   {i}. {failure['expectation']} on column '{failure['column']}'")
                
                if len(failed_expectations) > 5:
                    print(f"   ... and {len(failed_expectations) - 5} more")
            
            print("=" * 70)
        
        return {
            'success': success_rate >= 90,
            'success_rate': success_rate,
            'passed_checks': success_count,
            'total_checks': total_count,
            'failed_expectations': failed_expectations,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        if log_results:
            print(f"\n❌ Data quality validation failed: {e}")
            print("=" * 70)
        
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def save_validation_report(validation_results, output_dir='logs'):
    \"\"\"Save validation results to a report file\"\"\"
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f\"{output_dir}/data_quality_{timestamp}.txt\"
    
    with open(report_file, 'w') as f:
        f.write(\"=\" * 70 + \"\\n\")
        f.write(\"DATA QUALITY VALIDATION REPORT\\n\")
        f.write(\"=\" * 70 + \"\\n\")
        f.write(f\"Timestamp: {validation_results['timestamp']}\\n\")
        f.write(f\"Success Rate: {validation_results.get('success_rate', 0):.1f}%\\n\")
        f.write(f\"Passed Checks: {validation_results.get('passed_checks', 0)}\\n\")
        f.write(f\"Total Checks: {validation_results.get('total_checks', 0)}\\n\")
        f.write(f\"Overall Status: {'PASSED' if validation_results['success'] else 'FAILED'}\\n\")
        f.write(\"\\n\")
        
        if validation_results.get('failed_expectations'):
            f.write(\"FAILED EXPECTATIONS:\\n\")
            f.write(\"-\" * 70 + \"\\n\")
            for failure in validation_results['failed_expectations']:
                f.write(f\"Expectation: {failure['expectation']}\\n\")
                f.write(f\"Column: {failure['column']}\\n\")
                f.write(f\"Details: {failure['details']}\\n\")
                f.write(\"\\n\")
        
        f.write(\"=\" * 70 + \"\\n\")
    
    print(f\"📄 Validation report saved: {report_file}\")
    return report_file


# Test function
if __name__ == \"__main__\":
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print(\"=\" * 70)
    print(\"TESTING DATA QUALITY VALIDATION\")
    print(\"=\" * 70)
    
    # Load sample data
    engine = create_engine(os.getenv('DATABASE_URL'))
    with engine.connect() as conn:
        query = text(\"SELECT * FROM stock_data LIMIT 500\")
        df = pd.read_sql(query, conn)
    
    print(f\"\\n📊 Loaded {len(df)} records for validation\")
    
    # Run validation
    results = validate_stock_data(df, log_results=True)
    
    # Save report
    if results['success']:
        print(\"\\n✅ Data quality validation PASSED!\")
    else:
        print(\"\\n❌ Data quality validation FAILED!\")
    
    save_validation_report(results)
    
    print(\"\\n\" + \"=\" * 70)
