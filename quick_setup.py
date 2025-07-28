#!/usr/bin/env python3
"""
Quick Setup Script for Stock Market Prediction Dashboard
This script generates the required data files for the dashboard.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Create required directories."""
    directories = ['data/raw', 'data/processed', 'models', 'logs', 'outputs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directories created")

def collect_stock_data():
    """Collect stock data using yfinance."""
    print("📊 Collecting stock data...")
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA']
    start_date = '2022-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    all_data = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            data['Symbol'] = symbol
            data['price_change'] = data['Close'].pct_change()
            data['price_change_next_day'] = data['price_change'].shift(-1)
            all_data.append(data.reset_index())
            print(f"✅ {symbol}: {len(data)} records")
        except Exception as e:
            print(f"❌ Error with {symbol}: {e}")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save raw data
    combined_df.to_csv('data/raw/stock_data.csv', index=False)
    print(f"✅ Saved {len(combined_df)} total records to data/raw/stock_data.csv")
    
    return combined_df

def create_features(df):
    """Create technical indicators and features."""
    print("🔧 Creating features...")
    
    feature_df = df.copy()
    
    # Group by symbol to calculate features
    for symbol in feature_df['Symbol'].unique():
        mask = feature_df['Symbol'] == symbol
        symbol_data = feature_df[mask].copy()
        
        # Technical indicators
        symbol_data['SMA_5'] = symbol_data['Close'].rolling(5).mean()
        symbol_data['SMA_20'] = symbol_data['Close'].rolling(20).mean()
        symbol_data['SMA_50'] = symbol_data['Close'].rolling(50).mean()
        
        # RSI
        delta = symbol_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        symbol_data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = symbol_data['Close'].ewm(span=12).mean()
        exp2 = symbol_data['Close'].ewm(span=26).mean()
        symbol_data['MACD'] = exp1 - exp2
        symbol_data['MACD_signal'] = symbol_data['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        symbol_data['BB_middle'] = symbol_data['Close'].rolling(20).mean()
        bb_std = symbol_data['Close'].rolling(20).std()
        symbol_data['BB_upper'] = symbol_data['BB_middle'] + (bb_std * 2)
        symbol_data['BB_lower'] = symbol_data['BB_middle'] - (bb_std * 2)
        
        # Volume indicators
        symbol_data['volume_sma_20'] = symbol_data['Volume'].rolling(20).mean()
        symbol_data['volume_ratio'] = symbol_data['Volume'] / symbol_data['volume_sma_20']
        
        # Volatility
        symbol_data['volatility_20'] = symbol_data['Close'].rolling(20).std()
        
        # Update main dataframe
        feature_df.loc[mask] = symbol_data
    
    # Add sentiment features (simulated)
    np.random.seed(42)
    feature_df['sentiment_score'] = np.random.normal(0, 0.1, len(feature_df))
    feature_df['sentiment_positive'] = np.random.uniform(0, 1, len(feature_df))
    feature_df['sentiment_negative'] = np.random.uniform(0, 1, len(feature_df))
    feature_df['news_volume'] = np.random.poisson(5, len(feature_df))
    
    # Calendar features
    feature_df['Date'] = pd.to_datetime(feature_df['Date'])
    feature_df['day_of_week'] = feature_df['Date'].dt.dayofweek
    feature_df['month'] = feature_df['Date'].dt.month
    feature_df['quarter'] = feature_df['Date'].dt.quarter
    
    # Save processed data
    feature_df.to_csv('data/processed/featured_data.csv', index=False)
    print(f"✅ Created features - {feature_df.shape[1]} columns")
    
    return feature_df

def train_quick_models(df):
    """Train basic models for the dashboard."""
    print("🤖 Training models...")
    
    # Prepare data for ML
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'price_change',
        'SMA_5', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal',
        'BB_middle', 'BB_upper', 'BB_lower', 'volume_sma_20', 'volume_ratio',
        'volatility_20', 'sentiment_score', 'sentiment_positive', 'sentiment_negative',
        'news_volume', 'day_of_week', 'month', 'quarter'
    ]
    
    # Clean data
    ml_data = df.dropna().copy()
    
    if len(ml_data) < 100:
        print("❌ Not enough clean data for training")
        return
    
    X = ml_data[feature_columns]
    y = ml_data['price_change_next_day']
    
    # Remove any remaining NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    if len(X) < 50:
        print("❌ Insufficient data after cleaning")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Calculate directional accuracy
    actual_direction = (y_test > 0).astype(int)
    pred_direction = (rf_pred > 0).astype(int)
    directional_accuracy = (actual_direction == pred_direction).mean()
    
    print(f"✅ Model trained - Directional Accuracy: {directional_accuracy:.3f}")
    
    # Save models and scaler
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    # Save sample predictions for dashboard
    predictions_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': rf_pred,
        'symbol': X_test.index.map(lambda x: ml_data.loc[x, 'Symbol'] if x in ml_data.index else 'UNKNOWN'),
        'date': X_test.index.map(lambda x: ml_data.loc[x, 'Date'] if x in ml_data.index else pd.Timestamp.now())
    })
    predictions_df.to_csv('data/processed/predictions.csv', index=False)
    
    print("✅ Models and predictions saved")

def main():
    """Main execution function."""
    print("🚀 Quick Setup for Stock Market Dashboard")
    print("=" * 50)
    
    try:
        # Create directories
        create_directories()
        
        # Collect data
        stock_data = collect_stock_data()
        
        # Create features
        featured_data = create_features(stock_data)
        
        # Train models
        train_quick_models(featured_data)
        
        print("\n" + "=" * 50)
        print("✅ Setup complete! Dashboard data ready.")
        print("🌐 Now run: streamlit run dashboard/app.py")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
