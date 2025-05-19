import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
import datetime
from datetime import datetime, timedelta

# 비트코인 데이터 다운로드
end_date = datetime.today().strftime('%Y-%m-%d')
btc_data = yf.download('BTC-USD', start='2024-01-01', end=end_date)
btc_data = btc_data[['Open', 'High', 'Low', 'Close', 'Volume']]  # 필요한 열만 선택
btc_data['Target'] = btc_data['Close']  # 예측할 목표 값

# 데이터 정규화
scaler = StandardScaler()
scaled_data = scaler.fit_transform(btc_data)

# 윈도우 크기 설정
window_size = 30  # 30일간의 데이터를 입력으로 사용

x = []
y = []

for i in range(len(scaled_data) - window_size):
    x.append(scaled_data[i:i+window_size, :-1])  # 입력 데이터
    y.append(scaled_data[i+window_size, -1])    # 타겟 값

x = np.array(x)
y = np.array(y)

# 학습 및 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# LSTM 모델 구성
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=False))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 콜백 설정
checkpoint_filepath = './btc_lstm_model.h5'
callbacks = [
    ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_mae', mode='min', save_best_only=True),
    EarlyStopping(patience=10, restore_best_weights=True)
]

# 모델 학습
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks)

# 모델 평가
test_loss, test_mae = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# 예측 수행
predictions = model.predict(x_test)

# 결과 복원 (정규화 해제)
predicted_prices = scaler.inverse_transform(np.hstack([np.zeros((len(predictions), scaled_data.shape[1]-1)), predictions.reshape(-1, 1)]))[:, -1]
real_prices = scaler.inverse_transform(np.hstack([np.zeros((len(y_test), scaled_data.shape[1]-1)), y_test.reshape(-1, 1)]))[:, -1]

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(real_prices, label='Real Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.legend()
plt.title('Bitcoin Price Prediction with LSTM')
plt.show()