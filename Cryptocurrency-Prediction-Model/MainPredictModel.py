import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, GRU
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.regularizers import l1_l2

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Enhanced Technical Indicators Function
def calculate_advanced_technical_indicators(df):
    # Ensure we're working with a copy
    df = df.copy()

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
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # Log Returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(window=14).std() * np.sqrt(252)

    # Rate of Change
    df['ROC'] = df['Close'].pct_change(periods=14)

    return df.dropna()

# Data Preparation Function
def prepare_lstm_data(df, window_size=300, future_prediction_days=5):
    # Select features matching the current technical indicators
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'RSI', 'MACD', 'Signal_Line', 
        'BB_Middle', 'BB_Upper', 'BB_Lower', 
        'MA5', 'MA20', 
        'Log_Return', 'Volatility', 'ROC'
    ]
    
    # Normalize features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = StandardScaler()

    # Prepare features and target
    scaled_features = feature_scaler.fit_transform(df[features])
    scaled_target = target_scaler.fit_transform(df[['Close']])
    
    # Create sequences
    x, y = [], []
    for i in range(len(scaled_features) - window_size - future_prediction_days + 1):
        x.append(scaled_features[i:i+window_size])
        y.append(scaled_target[i+window_size+future_prediction_days-1])
    
    return np.array(x), np.array(y), feature_scaler, target_scaler

# Build Advanced Hybrid Model
def build_hybrid_model(input_shape, learning_rate=0.0005):
    model = Sequential([
        # Hybrid LSTM-GRU Architecture
        LSTM(256, activation='tanh', input_shape=input_shape, 
             return_sequences=True, 
             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        
        GRU(128, activation='relu', return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, activation='tanh'),
        BatchNormalization(),
        
        Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        Dense(1)
    ])
    
    # Use RMSprop with lower learning rate and decay
    optimizer = RMSprop(learning_rate=learning_rate, decay=1e-6)
    
    model.compile(optimizer=optimizer, 
                  loss='huber', 
                  metrics=['mae', 'mape'])
    
    return model

# Main Execution
def main():
    # Download Bitcoin Data
    end_date = datetime.today().strftime('%Y-%m-%d')
    btc_data = yf.download('BTC-USD', start='2020-01-01', end=end_date)

    # Calculate Advanced Technical Indicators
    btc_data = calculate_advanced_technical_indicators(btc_data)
    
    # Prepare Data
    X, y, feature_scaler, target_scaler = prepare_lstm_data(btc_data)
    
    # Split Data with Time Series Split
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize results tracking
    cv_results = []
    
    # Cross-Validation
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Build and Train Model
        model = build_hybrid_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5),
            ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
        ]
        
        # Train Model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Predict and Evaluate
        predictions = model.predict(X_test)
        
        # Inverse Transform Predictions
        real_prices = target_scaler.inverse_transform(y_test)
        predicted_prices = target_scaler.inverse_transform(predictions)
        
        # Calculate Metrics
        mae = mean_absolute_error(real_prices, predicted_prices)
        mse = mean_squared_error(real_prices, predicted_prices)
        mape = mean_absolute_percentage_error(real_prices, predicted_prices)
        
        cv_results.append({
            'MAE': mae,
            'MSE': mse,
            'MAPE': mape
        })
    
    # Print Cross-Validation Results
    print("Cross-Validation Results:")
    for i, result in enumerate(cv_results, 1):
        print(f"Fold {i}:")
        print(f"  MAE: {result['MAE']:.2f}")
        print(f"  MSE: {result['MSE']:.2f}")
        print(f"  MAPE: {result['MAPE']:.2%}")
    
    # Visualize Results
    plt.figure(figsize=(15, 7))
    plt.plot(real_prices, label='Real Prices', color='blue')
    plt.plot(predicted_prices, label='Predicted Prices', color='red')
    plt.title('Bitcoin Price Prediction Performance')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()