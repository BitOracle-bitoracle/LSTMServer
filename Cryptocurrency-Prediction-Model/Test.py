import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 1. 간단한 데이터 다운로드 및 준비
btc_data = yf.download('BTC-USD', start='2021-01-01', end='2023-01-01')  # 최근 2년치 데이터
btc_data = btc_data[['Open', 'High', 'Low', 'Close', 'Volume']]  # 필요한 열만 선택
btc_data = btc_data.dropna()  # 결측치 제거

# 2. 데이터 정규화
scaler = StandardScaler()
scaled_data = scaler.fit_transform(btc_data)

# 3. 테스트용 입력 데이터 생성
window_size = 10  # 간단히 10일 데이터만 사용
x = []
y = []

for i in range(len(scaled_data) - window_size):
    x.append(scaled_data[i:i+window_size, :-1])  # 입력 데이터
    y.append(scaled_data[i+window_size, -1])     # 타겟 값

x = np.array(x[:100])  # 데이터 샘플 크기를 100으로 제한
y = np.array(y[:100])

# 4. 간단한 LSTM 모델 정의
model = Sequential()
model.add(LSTM(16, activation='relu', input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# 5. 빠른 학습 (에포크 3, 배치 크기 8)
model.fit(x, y, epochs=3, batch_size=8, verbose=1)

# 6. 간단한 예측 테스트
predictions = model.predict(x)

# 7. 결과 복원 (정규화 해제)
predicted_prices = scaler.inverse_transform(
    np.hstack([np.zeros((len(predictions), scaled_data.shape[1] - 1)), predictions.reshape(-1, 1)])
)[:, -1]
real_prices = scaler.inverse_transform(
    np.hstack([np.zeros((len(y), scaled_data.shape[1] - 1)), y.reshape(-1, 1)])
)[:, -1]

# 8. 간단한 차트 출력
plt.figure(figsize=(8, 4))
plt.plot(real_prices, label='Real Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.legend()
plt.title('Test Bitcoin Prediction')
plt.show()