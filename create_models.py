#!/usr/bin/env python3
"""
Create dummy model files for dashboard
"""
try:
    import pickle
    import numpy as np
    
    # Create dummy model data
    model_data = {
        'type': 'RandomForestRegressor',
        'parameters': {'n_estimators': 100, 'random_state': 42},
        'trained': True
    }
    
    # Save model
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    # Create scaler data
    scaler_data = {
        'type': 'StandardScaler',
        'mean_': np.random.randn(20),
        'scale_': np.random.randn(20)
    }
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler_data, f)
    
    # Create feature names
    feature_names = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'price_change',
        'SMA_5', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal',
        'BB_middle', 'BB_upper', 'BB_lower', 'volume_sma_20', 'volume_ratio',
        'volatility_20', 'sentiment_score', 'day_of_week'
    ]
    
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("✅ Model files created successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    # Create empty files as fallback
    open('models/random_forest_model.pkl', 'w').close()
    open('models/scaler.pkl', 'w').close()
    open('models/feature_names.pkl', 'w').close()
    print("✅ Empty model files created as fallback")
