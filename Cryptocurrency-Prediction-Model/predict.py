from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import ccxt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import uvicorn

app = FastAPI()

def calculate_technical_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
    df['%K'] = 100 * (df['Close'] - df['Low'].rolling(14).min()) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min())
    df['%D'] = df['%K'].rolling(3).mean()
    return df

def get_binance_data(symbol='BTC/USDT', since='2022-10-12', limit=1000):
    exchange = ccxt.binance()
    since_ms = exchange.parse8601(since + 'T00:00:00Z')
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since_ms, limit=limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        since_ms = ohlcv[-1][0] + 24 * 60 * 60 * 1000
        if len(ohlcv) < limit:
            break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    return df.drop(columns=['timestamp'])

model = load_model('best_model.keras')

feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

@app.post("/predict")
def predict(future_days: int = Query(10, ge=1, le=30)):
    btc_data = get_binance_data("BTC/USDT", since='2024-5-12')
    btc_data['Target'] = btc_data['Close']
    btc_data = calculate_technical_indicators(btc_data).dropna()

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal_Line',
                'Log_Return', 'MA5', 'MA20', 'ATR', '%K', '%D']
    scaled_features = feature_scaler.fit_transform(btc_data[features])
    btc_data['Target_Scaled'] = target_scaler.fit_transform(btc_data[['Target']])

    window_size = 90
    scaled_data = np.hstack((scaled_features, btc_data[['Target_Scaled']].values))

    x_test = []
    for i in range(len(scaled_data) - window_size):
        x_test.append(scaled_data[i:i + window_size, :-1])
    x_test = np.array(x_test)

    # 과거 예측 (날짜 리스트)
    past_predictions_scaled = model.predict(x_test, verbose=0)
    past_predictions = target_scaler.inverse_transform(past_predictions_scaled)

    past_dates = btc_data.index[window_size:].to_pydatetime()
    past_result = [{"date": d.strftime('%Y-%m-%d'), "price": float(p)} for d, p in zip(past_dates, past_predictions.flatten())]

    # 미래 예측
    last_window = scaled_data[-window_size:, :-1].copy()
    future_predictions = []
    for _ in range(future_days):
        current_pred = model.predict(last_window.reshape(1, window_size, -1), verbose=0)[0][0]
        future_predictions.append(current_pred)

        new_row = last_window[-1].copy()
        new_row[0] = new_row[3]  # Open = 이전 Close
        new_row[1] = new_row[3] * (1 + np.random.uniform(0.01, 0.03))  # High
        new_row[2] = new_row[3] * (1 - np.random.uniform(0.01, 0.03))  # Low
        new_row[3] = current_pred  # Close

        delta = new_row[3] - new_row[0]
        new_row[5] = max(0, delta) / (abs(delta) + 1e-8) * 100  # 단순 RSI
        new_row[6] = new_row[3] - new_row[2]  # 단순 MACD

        last_window = np.vstack([last_window[1:], new_row])

    future_prices = target_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    last_date = btc_data.index[-1]
    future_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, future_days + 1)]

    future_result = [{"date": d, "price": float(p)} for d, p in zip(future_dates, future_prices.flatten())]

    return {
        "past_predictions": past_result,
        "future_predictions": future_result
    }

if __name__ == "__main__":
    uvicorn.run("filename:app", host="0.0.0.0", port=8000, reload=True)
