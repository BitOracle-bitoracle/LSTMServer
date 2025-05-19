import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# 보조지표 계산 함수
def calculate_technical_indicators(df):
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 로그 수익률
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # 이동평균
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # 새로운 보조 지표 추가
    df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()  # Average True Range
    df['%K'] = 100 * (df['Close'] - df['Low'].rolling(14).min()) / (
                df['High'].rolling(14).max() - df['Low'].rolling(14).min())  # Stochastic Oscillator
    df['%D'] = df['%K'].rolling(3).mean()

    return df


# 비트코인 데이터 다운로드
end_date = datetime.today().strftime('%Y-%m-%d')
btc_data = yf.download('BTC-KRW', start='2024-12-01', end=end_date)
btc_data = btc_data[['Open', 'High', 'Low', 'Close', 'Volume']]
btc_data['Target'] = btc_data['Close']

# 보조지표 추가
btc_data = calculate_technical_indicators(btc_data)
btc_data = btc_data.dropna()

# 데이터 정규화
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal_Line',
            'Log_Return', 'MA5', 'MA20', 'ATR', '%K', '%D']
feature_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = feature_scaler.fit_transform(btc_data[features])

# Target 값만 별도로 스케일링
target_scaler = MinMaxScaler(feature_range=(0, 1))
btc_data['Target_Scaled'] = target_scaler.fit_transform(btc_data[['Target']])

# 윈도우 설정 및 학습 데이터 생성
window_size = 90
x, y = [], []
scaled_data = np.hstack((scaled_features, btc_data[['Target_Scaled']].values))
for i in range(len(scaled_data) - window_size):
    x.append(scaled_data[i:i + window_size, :-1])
    y.append(scaled_data[i + window_size, -1])

x, y = np.array(x), np.array(y)

# 학습 및 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# CNN-LSTM-Attention 하이브리드 모델 구성
model = Sequential([
    # CNN 블록 - 간소화
    Conv1D(filters=64, kernel_size=3, activation='relu',
           input_shape=(window_size, x_train.shape[2])),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.1),

    # LSTM 블록
    LSTM(256, activation='tanh', return_sequences=True,
         recurrent_dropout=0.2),
    BatchNormalization(),

    LSTM(128, activation='tanh', return_sequences=False),
    BatchNormalization(),

    # Dense 블록
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# 컴파일 설정
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='huber',  # 단일 손실 함수 사용
    metrics=['mae', 'mape']
)

# 콜백 함수
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,  # 줄어든 patience
        restore_best_weights=True,
        mode='min',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # 더 점진적인 학습률 감소
        patience=5,
        min_lr=0.00001,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
    )
]

# 모델 학습
history = model.fit(
    x_train, y_train,
    epochs=150,
    batch_size=16,  # 작은 배치 사이즈
    validation_split=0.2,
    callbacks=callbacks,
    shuffle=False  # 시계열 데이터 순서 유지
)

# 모델 평가
test_loss, test_mae, test_mape = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}, Test MAPE: {test_mape}")

# 예측 수행
predictions = model.predict(x_test)

# 예측 데이터 복원 (정규화 해제)
predicted_prices = target_scaler.inverse_transform(predictions)

# 실제 데이터 복원 (정규화 해제)
real_prices = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# 미래 데이터 예측 (랜덤성 추가)
future_days = 10
last_window = scaled_data[-window_size:, :-1].copy()
future_predictions = []

for _ in range(future_days):
    current_pred = model.predict(last_window.reshape(1, window_size, -1), verbose=0)[0][0]
    future_predictions.append(current_pred)

    # 랜덤성을 고려한 미래 데이터 업데이트
    new_row = last_window[-1].copy()
    new_row[0] = new_row[3]  # Open = 이전 Close
    new_row[1] = new_row[3] * (1 + np.random.uniform(0.01, 0.03))  # High = 이전 Close + 1~3%
    new_row[2] = new_row[3] * (1 - np.random.uniform(0.01, 0.03))  # Low = 이전 Close - 1~3%
    new_row[3] = current_pred  # Close = 예측값

    # RSI, MACD 갱신
    delta = new_row[3] - new_row[0]
    new_row[5] = max(0, delta) / (abs(delta) + 1e-8) * 100  # 단순 RSI
    new_row[6] = new_row[3] - new_row[2]  # 단순 MACD

    last_window = np.vstack([last_window[1:], new_row])

# 정규화 해제
future_prices = target_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 미래 날짜 생성
last_date = btc_data.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]

# 차트 그리기
plt.figure(figsize=(14, 7))
plt.plot(btc_data.index[-len(real_prices):], real_prices, label='Real Prices', color='blue')
plt.plot(btc_data.index[-len(predicted_prices):], predicted_prices, label='Predicted Prices', color='orange')
plt.plot(future_dates, future_prices, label='Future Predictions', color='green', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Price Prediction with Enhanced LSTM')
plt.legend()
plt.grid()
plt.show()