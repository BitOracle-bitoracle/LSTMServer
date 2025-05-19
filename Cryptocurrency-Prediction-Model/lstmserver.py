from typing import List
from fastapi import FastAPI, Query,HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import yfinance as yf
import uvicorn
from pydantic import BaseModel

class PredictRequest(BaseModel):
    window_size: int
    start_date: str
    end_date: str
    future_prediction_days: int


app = FastAPI()
class PriceDto(BaseModel):
    date: str
    actual: float
    createdAt: str


@app.get("/bitcoin/history", response_model=List[PriceDto])
def get_bitcoin_history(
    start_date: str = Query(..., description="예: 2024-01-01"),
    end_date: str = Query(..., description="예: 2024-05-01")
):
    try:
        df = yf.download("BTC-KRW", start=start_date, end=end_date)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data for given range")

        df.columns = df.columns.get_level_values(0)  # 멀티인덱스 -> 단일레벨로 변환

        df = df.reset_index()
        df = df[["Date", "Close"]].dropna()
        df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
        df["actual"] = df["Close"].round(2)
        df["createdAt"] = datetime.now().strftime("%Y-%m-%d")

        result = df[["date", "actual", "createdAt"]].to_dict(orient="records")
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 전역 변수 초기화
model = load_model('best_model.keras')
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal_Line',
            'Log_Return', 'MA5', 'MA20', 'ATR', '%K', '%D']

# 보조지표 계산 함수
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
    df['%K'] = 100 * (df['Close'] - df['Low'].rolling(14).min()) / \
               (df['High'].rolling(14).max() - df['Low'].rolling(14).min())
    df['%D'] = df['%K'].rolling(3).mean()
    return df.dropna()

# 데이터 준비 함수
def prepare_data(start_date,end_date, window_size):
    #end_date = datetime.today().strftime('%Y-%m-%d')
    df = yf.download('BTC-KRW', start=start_date, end=end_date)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df['Target'] = df['Close']
    df = calculate_technical_indicators(df)

    # 정규화
    scaled_features = feature_scaler.fit_transform(df[features])
    df['Target_Scaled'] = target_scaler.fit_transform(df[['Target']])

    # 입력 시퀀스 생성
    x, y = [], []
    scaled_data = np.hstack((scaled_features, df[['Target_Scaled']].values))
    for i in range(len(scaled_data) - window_size):
        x.append(scaled_data[i:i + window_size, :-1])
        y.append(scaled_data[i + window_size, -1])
    return np.array(x), np.array(y), df

@app.post("/predict")
def predict_prices(req: PredictRequest):
    window_size = req.window_size
    start_date = req.start_date
    end_date = req.end_date
    future_prediction_days = req.future_prediction_days

    # 데이터 준비
    x, y, df = prepare_data(start_date,end_date, window_size)

    # 과거 예측
    y_pred_scaled = model.predict(x, verbose=0)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_true = target_scaler.inverse_transform(y.reshape(-1, 1))
    dates = df.index[window_size:]

    predictions = [
        {"date": str(date.date()), "predicted": float(pred[0]), "actual": float(true[0])}
        for date, pred, true in zip(dates, y_pred, y_true)
    ]

    # 미래 예측
    last_window = x[-1]
    for i in range(future_prediction_days):
        current_pred_scaled = model.predict(last_window.reshape(1, window_size, -1), verbose=0)[0][0]
        predicted_price = target_scaler.inverse_transform([[current_pred_scaled]])[0][0]
        prediction_date = df.index[-1] + timedelta(days=i + 1)

        predictions.append({
            "date": str(prediction_date.date()),
            "predicted": float(predicted_price),
            "actual": None
        })

        # 다음 시퀀스를 위해 가짜 데이터 생성
        new_row = last_window[-1].copy()
        new_row[0] = new_row[3]
        new_row[1] = new_row[3] * (1 + np.random.uniform(0.01, 0.03))
        new_row[2] = new_row[3] * (1 - np.random.uniform(0.01, 0.03))
        new_row[3] = current_pred_scaled
        delta = new_row[3] - new_row[0]
        new_row[5] = max(0, delta) / (abs(delta) + 1e-8) * 100
        new_row[6] = new_row[3] - new_row[2]
        last_window = np.vstack([last_window[1:], new_row])

    return {
        "predictions": predictions
    }