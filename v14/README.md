# GRU v14 - BTC/USDT 방향 예측 모델

비트코인 1시간봉 데이터를 기반으로 **1시간 뒤 가격 방향(상승/하락)을 이진 분류**하는 GRU 딥러닝 모델과 FastAPI 예측 서버입니다.

---

## 파일 구조

```
v14/
├── GRU_v14_Tunned.py           # 모델 학습 스크립트
├── v14_Backtest(StopLoss).py   # 손절/익절 전략 포함 백테스팅
├── GRUServer.py                # FastAPI 예측 서버
├── best_model_GRU_tuned_v14.keras  # 저장된 학습 모델
├── advanced_backtest_result.png    # 백테스트 결과 시각화
└── GRUServer/                  # 서버 배포용 디렉토리
    ├── GRUServer.py
    └── best_model_GRU_tuned_v14.keras
```

---

## 모델 구조 (GRU v14)

| 레이어 | 설정 |
|---|---|
| GRU | 64 units, tanh, return_sequences=False |
| BatchNormalization | - |
| Dropout | 0.4 |
| Dense | 32 units, ReLU |
| Dropout | 0.3 |
| Dense (출력) | 1 unit, Sigmoid |

- **입력**: `(batch, 48, 14)` — 48시간 윈도우 × 14개 feature
- **출력**: 0~1 사이 상승 확률 (0.5 이상이면 상승 예측)
- **손실 함수**: Binary Crossentropy
- **옵티마이저**: Adam (lr=0.0001, clipnorm=1.0)

---

## 입력 Feature (총 14개)

### 가격 관련 (pct_change 변환)
| Feature | 설명 |
|---|---|
| Open, High, Low, Close | OHLC 가격 |
| Volume | 거래량 |
| MA5, MA20 | 5/20시간 이동평균 |

### 보조지표 (diff 변환)
| Feature | 설명 |
|---|---|
| RSI | 14봉 RSI |
| MACD | EMA(12) - EMA(26) |
| Signal_Line | MACD 9봉 EMA |
| Log_Return | 로그 수익률 |
| ATR | 14봉 평균 실제 범위 |
| %K, %D | 14봉 스토캐스틱 |

> 외부 데이터(S&P500, 10Y Yield, DXY, Gold) 수집이 실패한 경우 위 14개만 사용됩니다.

---

## 데이터 전처리 파이프라인

1. **데이터 수집**: Binance CCXT API (BTC/USDT 1시간봉, 2018-01-01~)
2. **외부 데이터 병합**: yfinance로 S&P500, 미국채 10년물, DXY, Gold (실패 시 스킵)
3. **보조지표 계산**: RSI, MACD, MA, ATR, Stochastic
4. **정상화 변환**: 가격 계열 → `pct_change`, 지표 계열 → `diff`
5. **스케일링**: StandardScaler
6. **시퀀스 생성**: Sliding Window (window_size=48, target=1시간 후)

---

## 학습 설정

| 항목 | 값 |
|---|---|
| Window Size | 48시간 (2일) |
| 예측 목표 | 1시간 후 방향 |
| Train/Test Split | 80 / 20 (시간순, shuffle=False) |
| Batch Size | 256 |
| Max Epochs | 150 |
| Early Stopping | val_accuracy 기준, patience=10 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |

---

## 백테스팅 전략 (`v14_Backtest(StopLoss).py`)

Buy & Hold 대비 성과를 검증하는 고급 전략입니다.

| 파라미터 | 값 | 설명 |
|---|---|---|
| 초기 자본 | $10,000 | - |
| 수수료 | 0.1% | 거래당 |
| 매수 임계값 | 60% | 상승 확률 ≥ 60% 시 매수 |
| 매도 임계값 | 40% | 상승 확률 ≤ 40% 시 매도 |
| 손절 (Stop Loss) | 3% | 매수가 대비 -3% 시 강제 청산 |
| 익절 (Take Profit) | 5% | 매수가 대비 +5% 시 강제 청산 |

---

## API 서버 (`GRUServer.py`)

FastAPI 기반 서버. 실시간 Binance 데이터를 수집해 모델 예측 결과를 제공합니다.

### 실행

```bash
python GRUServer.py
# 또는
uvicorn GRUServer:app --host 0.0.0.0 --port 8000 --reload
```

### 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/predict/chart` | AI 상승 확률 예측 결과 (시계열) |
| GET | `/api/price/chart` | 실제 BTC 가격 이력 |

#### 응답 예시 (`/api/predict/chart`)

```json
[
  {
    "date": "2025-03-25 14:00",
    "predicted": 0.732,
    "actual": 0
  }
]
```

### CORS
기본적으로 `allow_origins=["*"]`로 설정되어 있습니다. 로컬 테스트 시에는 소스 내 주석 처리된 `origins` 리스트로 교체하세요.

---

## 의존성

```bash
pip install tensorflow keras scikit-learn pandas numpy ccxt yfinance fastapi uvicorn matplotlib
```
