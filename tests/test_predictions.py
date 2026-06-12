import pandas as pd
from datetime import datetime
from src.predictions import (
    predict_next_day_price,
    generate_predictions_for_all,
    get_trading_signals,
    get_top_predictions,
)


def _prediction_df(rows=25, bullish=True):
    dates = pd.date_range(end=datetime(2026, 1, 25), periods=rows, freq='D')
    closes = [150.0 + i for i in range(rows)]
    ma5 = [152.0 + i for i in range(rows)]
    ma20 = [148.0 + i for i in range(rows)] if bullish else [155.0 + i for i in range(rows)]
    return pd.DataFrame({
        'symbol': ['AAPL'] * rows,
        'date': dates,
        'close': closes,
        'ma_5': ma5,
        'ma_20': ma20,
        'daily_change_percent': [0.5] * rows,
    })


class TestPredictions:
    def test_predict_insufficient_data_returns_none(self):
        df = _prediction_df(rows=10)
        assert predict_next_day_price(df, 'AAPL') is None

    def test_predict_bullish_trend(self):
        result = predict_next_day_price(_prediction_df(rows=25, bullish=True), 'AAPL')
        assert result is not None
        assert result['trend'] == 'BULLISH'
        assert result['symbol'] == 'AAPL'
        assert 'predicted_price' in result

    def test_predict_bearish_trend(self):
        result = predict_next_day_price(_prediction_df(rows=25, bullish=False), 'AAPL')
        assert result is not None
        assert result['trend'] == 'BEARISH'

    def test_generate_predictions_for_all(self):
        df = pd.concat([
            _prediction_df(rows=25, bullish=True),
            _prediction_df(rows=10, bullish=True).assign(symbol='SHORT'),
        ], ignore_index=True)
        preds = generate_predictions_for_all(df)
        assert len(preds) == 1
        assert preds.iloc[0]['symbol'] == 'AAPL'

    def test_generate_predictions_empty(self):
        df = pd.DataFrame({'symbol': [], 'date': [], 'close': [], 'ma_5': [], 'ma_20': [], 'daily_change_percent': []})
        assert generate_predictions_for_all(df).empty

    def test_get_trading_signals_golden_cross(self):
        df = _prediction_df(rows=12, bullish=True)
        df.loc[df.index[-2], 'ma_5'] = 140
        df.loc[df.index[-2], 'ma_20'] = 145
        df.loc[df.index[-1], 'ma_5'] = 150
        df.loc[df.index[-1], 'ma_20'] = 145
        signals = get_trading_signals(df)
        assert not signals.empty
        assert signals.iloc[0]['signal'] == 'BUY'

    def test_get_trading_signals_death_cross(self):
        df = _prediction_df(rows=12, bullish=False)
        df.loc[df.index[-2], 'ma_5'] = 150
        df.loc[df.index[-2], 'ma_20'] = 145
        df.loc[df.index[-1], 'ma_5'] = 140
        df.loc[df.index[-1], 'ma_20'] = 145
        signals = get_trading_signals(df)
        assert not signals.empty
        assert signals.iloc[0]['signal'] == 'SELL'

    def test_get_trading_signals_no_cross(self):
        df = _prediction_df(rows=12, bullish=True)
        signals = get_trading_signals(df)
        assert signals.empty

    def test_get_top_predictions_gainers(self):
        df = pd.concat([
            _prediction_df(rows=25).assign(symbol='AAPL'),
            _prediction_df(rows=25, bullish=False).assign(symbol='MSFT'),
        ], ignore_index=True)
        top = get_top_predictions(df, top_n=1, prediction_type='gainers')
        assert len(top) == 1

    def test_get_top_predictions_losers(self):
        df = pd.concat([
            _prediction_df(rows=25).assign(symbol='AAPL'),
            _prediction_df(rows=25, bullish=False).assign(symbol='MSFT'),
        ], ignore_index=True)
        top = get_top_predictions(df, top_n=1, prediction_type='losers')
        assert len(top) == 1

    def test_get_top_predictions_empty(self):
        df = pd.DataFrame({'symbol': [], 'date': [], 'close': [], 'ma_5': [], 'ma_20': [], 'daily_change_percent': []})
        assert get_top_predictions(df).empty
