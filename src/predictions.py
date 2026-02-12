import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def predict_next_day_price(df, symbol):
    """
    Predict next day's price using moving average trend analysis

    Strategy:
    - If MA5 > MA20: Bullish trend (predict price increase)
    - If MA5 < MA20: Bearish trend (predict price decrease)
    - Use recent volatility to estimate change magnitude

    Args:
        df (pd.DataFrame): Stock data with moving averages
        symbol (str): Stock symbol to predict

    Returns:
        dict: Prediction details or None if insufficient data
    """

    # Filter for specific symbol and sort by date
    symbol_df = df[df['symbol'] == symbol].sort_values('date', ascending=False)

    if len(symbol_df) < 20:
        return None  # Need at least 20 days for reliable prediction

    # Get latest data
    latest = symbol_df.iloc[0]
    recent_data = symbol_df.head(5)  # Last 5 days

    # Extract key metrics
    current_price = latest['close']
    ma5 = latest['ma_5']
    ma20 = latest['ma_20']
    recent_volatility = recent_data['daily_change_percent'].std()
    avg_daily_change = recent_data['daily_change_percent'].mean()

    # Determine trend
    if ma5 > ma20:
        trend = 'BULLISH'
        trend_strength = ((ma5 - ma20) / ma20 * 100)  # How much MA5 is above MA20
    else:
        trend = 'BEARISH'
        trend_strength = ((ma20 - ma5) / ma20 * 100)  # How much MA5 is below MA20

    # Predict price change based on trend and recent performance
    if trend == 'BULLISH':
        # Predict increase proportional to trend strength and recent performance
        predicted_change_percent = min(trend_strength * 0.3 + avg_daily_change * 0.7, 5.0)  # Cap at 5%
    else:
        # Predict decrease
        predicted_change_percent = max(-trend_strength * 0.3 + avg_daily_change * 0.7, -5.0)  # Cap at -5%

    # Calculate predicted price
    predicted_price = current_price * (1 + predicted_change_percent / 100)

    # Confidence level based on trend strength and volatility
    if trend_strength > 5 and recent_volatility < 2:
        confidence = 'HIGH'
    elif trend_strength > 2 and recent_volatility < 4:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    return {
        'symbol': symbol,
        'current_price': round(current_price, 2),
        'predicted_price': round(predicted_price, 2),
        'predicted_change': round(predicted_price - current_price, 2),
        'predicted_change_percent': round(predicted_change_percent, 2),
        'trend': trend,
        'trend_strength': round(trend_strength, 2),
        'confidence': confidence,
        'ma_5': round(ma5, 2),
        'ma_20': round(ma20, 2),
        'recent_volatility': round(recent_volatility, 2),
        'prediction_date': (latest['date'] + timedelta(days=1)).strftime('%Y-%m-%d'),
        'current_date': latest['date'].strftime('%Y-%m-%d')
    }


def generate_predictions_for_all(df):
    """
    Generate predictions for all symbols in the DataFrame

    Args:
        df (pd.DataFrame): Complete stock data

    Returns:
        pd.DataFrame: Predictions for all symbols
    """
    symbols = df['symbol'].unique()
    predictions = []

    for symbol in symbols:
        pred = predict_next_day_price(df, symbol)
        if pred:
            predictions.append(pred)

    if not predictions:
        return pd.DataFrame()

    pred_df = pd.DataFrame(predictions)

    # Sort by predicted change (biggest movers first)
    pred_df = pred_df.sort_values('predicted_change_percent', ascending=False)

    return pred_df


def get_trading_signals(df):
    """
    Generate buy/sell signals based on moving average crossover

    Golden Cross (BUY): MA5 crosses above MA20
    Death Cross (SELL): MA5 crosses below MA20

    Args:
        df (pd.DataFrame): Stock data with moving averages

    Returns:
        pd.DataFrame: Trading signals
    """
    df = df.sort_values(['symbol', 'date'])

    signals = []

    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].tail(10)  # Last 10 days

        if len(symbol_df) < 2:
            continue

        # Check last 5 days for crossovers
        for i in range(len(symbol_df) - 1):
            prev = symbol_df.iloc[i]
            current = symbol_df.iloc[i + 1]

            signal = None
            signal_type = None

            # Golden Cross - BUY signal
            if prev['ma_5'] <= prev['ma_20'] and current['ma_5'] > current['ma_20']:
                signal = 'BUY'
                signal_type = 'Golden Cross'

            # Death Cross - SELL signal
            elif prev['ma_5'] >= prev['ma_20'] and current['ma_5'] < current['ma_20']:
                signal = 'SELL'
                signal_type = 'Death Cross'

            if signal:
                signals.append({
                    'symbol': symbol,
                    'signal': signal,
                    'signal_type': signal_type,
                    'date': current['date'],
                    'price': current['close'],
                    'ma_5': current['ma_5'],
                    'ma_20': current['ma_20'],
                    'days_ago': (df['date'].max() - current['date']).days
                })

    if not signals:
        return pd.DataFrame()

    signals_df = pd.DataFrame(signals)

    # Sort by most recent
    signals_df = signals_df.sort_values('date', ascending=False)

    return signals_df


def get_top_predictions(df, top_n=5, prediction_type='gainers'):
    """
    Get top predicted gainers or losers

    Args:
        df (pd.DataFrame): Stock data
        top_n (int): Number of top stocks to return
        prediction_type (str): 'gainers' or 'losers'

    Returns:
        pd.DataFrame: Top predictions
    """
    predictions = generate_predictions_for_all(df)

    if predictions.empty:
        return pd.DataFrame()

    if prediction_type == 'gainers':
        return predictions.nlargest(top_n, 'predicted_change_percent')
    else:  # losers
        return predictions.nsmallest(top_n, 'predicted_change_percent')


# Test function
if __name__ == "__main__":
    from sqlalchemy import create_engine, text
    import os
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 70)
    print("TESTING ML PRICE PREDICTIONS")
    print("=" * 70)

    # Load data from database
    engine = create_engine(os.getenv('DATABASE_URL'))
    with engine.connect() as conn:
        query = text("SELECT * FROM stock_data ORDER BY date DESC LIMIT 1000")
        df = pd.read_sql(query, conn)
        df['date'] = pd.to_datetime(df['date'])

    print(f"\n📊 Loaded {len(df)} records for {df['symbol'].nunique()} symbols")

    # Generate predictions
    print("\n🤖 Generating predictions...")
    predictions = generate_predictions_for_all(df)

    if not predictions.empty:
        print(f"\n✅ Generated {len(predictions)} predictions")
        print("\n📈 Top 5 Predicted Gainers:")
        print(predictions.head().to_string(index=False))

        print("\n📉 Top 5 Predicted Losers:")
        print(predictions.tail().to_string(index=False))

    # Get trading signals
    print("\n\n🎯 Checking for trading signals...")
    signals = get_trading_signals(df)

    if not signals.empty:
        print(f"\n✅ Found {len(signals)} trading signals")
        print(signals.to_string(index=False))
    else:
        print("ℹ️  No recent MA crossover signals detected")

    print("\n" + "=" * 70)
