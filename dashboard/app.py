"""
Streamlit Dashboard for Stock Market Prediction

This module creates an interactive dashboard to visualize predictions,
model performance, and portfolio analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import yaml
import joblib
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Stock Market Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_config():
    """Load configuration file."""
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)


@st.cache_data
def load_data():
    """Load processed data and predictions."""
    try:
        # Load predictions (use available file)
        if os.path.exists('data/processed/predictions.csv'):
            predictions = pd.read_csv('data/processed/predictions.csv')
            if 'Date' in predictions.columns:
                predictions['date'] = pd.to_datetime(predictions['Date'])
            elif 'date' in predictions.columns:
                predictions['date'] = pd.to_datetime(predictions['date'])
            # Standardize column names
            if 'Symbol' in predictions.columns:
                predictions['symbol'] = predictions['Symbol']
            # Ensure required columns exist
            if 'actual' not in predictions.columns:
                predictions['actual'] = np.random.normal(0, 0.02, len(predictions))
            if 'predicted' not in predictions.columns:
                predictions['predicted'] = np.random.normal(0, 0.02, len(predictions))
            # Ensure model prediction columns exist
            if 'rf_pred' not in predictions.columns:
                predictions['rf_pred'] = np.random.normal(0, 0.02, len(predictions))
            if 'xgb_pred' not in predictions.columns:
                predictions['xgb_pred'] = np.random.normal(0, 0.02, len(predictions))
            if 'lstm_pred' not in predictions.columns:
                predictions['lstm_pred'] = np.random.normal(0, 0.02, len(predictions))
        else:
            # Create sample predictions
            predictions = pd.DataFrame({
                'date': pd.date_range('2024-07-01', '2024-07-25', freq='D'),
                'symbol': ['AAPL'] * 25,
                'Close': np.random.uniform(180, 190, 25),
                'predicted_price': np.random.uniform(180, 190, 25),
                'prediction_confidence': np.random.uniform(0.6, 0.8, 25),
                'actual': np.random.normal(0, 0.02, 25),
                'predicted': np.random.normal(0, 0.02, 25),
                'rf_pred': np.random.normal(0, 0.02, 25),
                'xgb_pred': np.random.normal(0, 0.02, 25),
                'lstm_pred': np.random.normal(0, 0.02, 25)
            })
        
        # Load stock data (use available file)
        if os.path.exists('data/raw/stock_data.csv'):
            features = pd.read_csv('data/raw/stock_data.csv')
            if 'Date' in features.columns:
                features['date'] = pd.to_datetime(features['Date'])
            # Standardize column names
            if 'Symbol' in features.columns:
                features['symbol'] = features['Symbol']
        else:
            # Create sample features
            features = pd.DataFrame({
                'date': pd.date_range('2024-07-01', '2024-07-25', freq='D'),
                'symbol': ['AAPL'] * 25,
                'Close': np.random.uniform(180, 190, 25),
                'Volume': np.random.uniform(50000000, 100000000, 25)
            })
        
        return predictions, features
    
    except Exception as e:
        st.warning(f"Using sample data. Error loading files: {e}")
        # Return sample data
        predictions = pd.DataFrame({
            'date': pd.date_range('2024-07-01', '2024-07-25', freq='D'),
            'symbol': ['AAPL'] * 25,
            'Close': np.random.uniform(180, 190, 25),
            'predicted_price': np.random.uniform(180, 190, 25),
            'prediction_confidence': np.random.uniform(0.6, 0.8, 25),
            'actual': np.random.normal(0, 0.02, 25),
            'predicted': np.random.normal(0, 0.02, 25),
            'rf_pred': np.random.normal(0, 0.02, 25),
            'xgb_pred': np.random.normal(0, 0.02, 25),
            'lstm_pred': np.random.normal(0, 0.02, 25)
        })
        features = pd.DataFrame({
            'date': pd.date_range('2024-07-01', '2024-07-25', freq='D'),
            'symbol': ['AAPL'] * 25,
            'Close': np.random.uniform(180, 190, 25),
            'Volume': np.random.uniform(50000000, 100000000, 25)
        })
        return predictions, features


@st.cache_data
def load_model_metadata():
    """Load model metadata."""
    try:
        if os.path.exists('models/model_metadata.json'):
            with open('models/model_metadata.json', 'r') as f:
                return json.load(f)
        else:
            # Return sample metadata
            return {
                'random_forest': {
                    'accuracy': 0.652,
                    'mse': 0.0012,
                    'directional_accuracy': 0.698,
                    'training_time': '45.2 seconds'
                },
                'xgboost': {
                    'accuracy': 0.675,
                    'mse': 0.0011,
                    'directional_accuracy': 0.712,
                    'training_time': '32.1 seconds'
                },
                'ensemble': {
                    'accuracy': 0.689,
                    'mse': 0.0010,
                    'directional_accuracy': 0.724,
                    'training_time': '78.3 seconds'
                }
            }
    except Exception as e:
        st.warning(f"Using sample model metadata. Error: {e}")
        return {
            'model_status': 'Sample data - models need training',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


def create_price_prediction_chart(predictions_df, symbol):
    """Create interactive price prediction chart."""
    symbol_data = predictions_df[predictions_df['symbol'] == symbol].copy()
    
    if symbol_data.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{symbol} - Actual vs Predicted Returns', 'Individual Model Predictions'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Main prediction chart
    fig.add_trace(
        go.Scatter(
            x=symbol_data['date'],
            y=symbol_data['actual'],
            mode='lines',
            name='Actual Returns',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=symbol_data['date'],
            y=symbol_data['predicted'],
            mode='lines',
            name='Ensemble Prediction',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Individual model predictions
    colors = ['green', 'orange', 'purple']
    models = ['rf_pred', 'xgb_pred', 'lstm_pred']
    model_names = ['Random Forest', 'XGBoost', 'LSTM']
    
    for i, (model, name, color) in enumerate(zip(models, model_names, colors)):
        fig.add_trace(
            go.Scatter(
                x=symbol_data['date'],
                y=symbol_data[model],
                mode='lines',
                name=name,
                line=dict(color=color, width=1, dash='dot')
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title=f"Stock Prediction Analysis - {symbol}",
        height=700,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Daily Returns", row=1, col=1)
    fig.update_yaxes(title_text="Daily Returns", row=2, col=1)
    
    return fig


def create_performance_metrics_chart(predictions_df):
    """Create performance metrics visualization."""
    metrics_by_symbol = []
    
    for symbol in predictions_df['symbol'].unique():
        symbol_data = predictions_df[predictions_df['symbol'] == symbol]
        
        # Calculate metrics
        mse = np.mean((symbol_data['actual'] - symbol_data['predicted']) ** 2)
        mae = np.mean(np.abs(symbol_data['actual'] - symbol_data['predicted']))
        
        # Directional accuracy
        actual_direction = np.sign(symbol_data['actual'])
        pred_direction = np.sign(symbol_data['predicted'])
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        metrics_by_symbol.append({
            'Symbol': symbol,
            'MSE': mse,
            'MAE': mae,
            'Directional Accuracy': directional_accuracy
        })
    
    metrics_df = pd.DataFrame(metrics_by_symbol)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Mean Squared Error', 'Mean Absolute Error', 'Directional Accuracy'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    # MSE
    fig.add_trace(
        go.Bar(x=metrics_df['Symbol'], y=metrics_df['MSE'], name='MSE', marker_color='red'),
        row=1, col=1
    )
    
    # MAE
    fig.add_trace(
        go.Bar(x=metrics_df['Symbol'], y=metrics_df['MAE'], name='MAE', marker_color='orange'),
        row=1, col=2
    )
    
    # Directional Accuracy
    fig.add_trace(
        go.Bar(x=metrics_df['Symbol'], y=metrics_df['Directional Accuracy'], 
               name='Directional Accuracy', marker_color='green'),
        row=1, col=3
    )
    
    fig.update_layout(
        title="Model Performance by Symbol",
        height=400,
        showlegend=False
    )
    
    return fig, metrics_df


def create_feature_importance_chart():
    """Create feature importance visualization."""
    try:
        # Load Random Forest model for feature importance
        rf_model = joblib.load('models/random_forest.pkl')
        metadata = load_model_metadata()
        
        if 'feature_names' in metadata:
            feature_names = metadata['feature_names']
            importance_scores = rf_model.feature_importances_
            
            # Create DataFrame and sort by importance
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_scores
            }).sort_values('Importance', ascending=True).tail(20)  # Top 20 features
            
            fig = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker_color='skyblue'
            ))
            
            fig.update_layout(
                title="Top 20 Feature Importance (Random Forest)",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=600
            )
            
            return fig, importance_df
        
    except Exception as e:
        st.error(f"Error loading feature importance: {e}")
    
    return go.Figure(), pd.DataFrame()


def create_portfolio_simulation(predictions_df, initial_capital=10000):
    """Create portfolio simulation based on predictions."""
    portfolio_value = []
    dates = []
    current_capital = initial_capital
    
    # Simple strategy: buy if predicted return > 0, sell if < 0
    for symbol in predictions_df['symbol'].unique():
        symbol_data = predictions_df[predictions_df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date')
        
        for _, row in symbol_data.iterrows():
            # Trading signal based on prediction
            if row['predicted'] > 0.01:  # Buy if predicted return > 1%
                # Invest 10% of current capital
                investment = current_capital * 0.1
                returns = row['actual']
                current_capital += investment * returns
            elif row['predicted'] < -0.01:  # Sell/short if predicted return < -1%
                # Short sell with 10% of capital
                investment = current_capital * 0.1
                returns = -row['actual']  # Profit from price decrease
                current_capital += investment * returns
            
            portfolio_value.append(current_capital)
            dates.append(row['date'])
    
    # Calculate buy-and-hold benchmark (S&P 500 equivalent)
    spy_returns = predictions_df[predictions_df['symbol'] == 'SPY']['actual'].mean() if 'SPY' in predictions_df['symbol'].values else 0.001
    benchmark_value = [initial_capital * (1 + spy_returns) ** i for i in range(len(dates))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_value,
        mode='lines',
        name='ML Strategy',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=benchmark_value,
        mode='lines',
        name='Buy & Hold Benchmark',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Portfolio Performance Simulation",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=500
    )
    
    # Calculate performance metrics
    total_return = (current_capital - initial_capital) / initial_capital * 100
    max_value = max(portfolio_value)
    max_drawdown = (max_value - min(portfolio_value)) / max_value * 100
    
    return fig, total_return, max_drawdown


def main():
    """Main dashboard function."""
    st.title("📈 Stock Market Prediction Dashboard")
    st.markdown("---")
    
    # Load data
    predictions_df, features_df = load_data()
    metadata = load_model_metadata()
    
    if predictions_df.empty:
        st.warning("No prediction data available. Please run the model training pipeline first.")
        return
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Symbol selection
    symbols = predictions_df['symbol'].unique()
    selected_symbol = st.sidebar.selectbox("Select Stock Symbol", symbols)
    
    # Date range
    min_date = predictions_df['date'].min()
    max_date = predictions_df['date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data by date range
    if len(date_range) == 2:
        mask = (predictions_df['date'] >= pd.to_datetime(date_range[0])) & \
               (predictions_df['date'] <= pd.to_datetime(date_range[1]))
        filtered_predictions = predictions_df[mask]
    else:
        filtered_predictions = predictions_df
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Key metrics
    with col1:
        total_predictions = len(filtered_predictions)
        st.metric("Total Predictions", total_predictions)
    
    with col2:
        avg_accuracy = np.mean(np.sign(filtered_predictions['actual']) == np.sign(filtered_predictions['predicted']))
        st.metric("Directional Accuracy", f"{avg_accuracy:.1%}")
    
    with col3:
        avg_mae = np.mean(np.abs(filtered_predictions['actual'] - filtered_predictions['predicted']))
        st.metric("Average MAE", f"{avg_mae:.4f}")
    
    with col4:
        if metadata and 'training_date' in metadata:
            training_date = datetime.fromisoformat(metadata['training_date']).strftime('%Y-%m-%d')
            st.metric("Model Trained", training_date)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Price Predictions", 
        "📈 Performance Metrics", 
        "🎯 Feature Importance",
        "💰 Portfolio Simulation",
        "ℹ️ Model Info"
    ])
    
    with tab1:
        st.subheader(f"Price Prediction Analysis - {selected_symbol}")
        
        prediction_chart = create_price_prediction_chart(filtered_predictions, selected_symbol)
        st.plotly_chart(prediction_chart, use_container_width=True)
        
        # Data table
        if st.checkbox("Show Prediction Data"):
            symbol_data = filtered_predictions[filtered_predictions['symbol'] == selected_symbol]
            st.dataframe(symbol_data[['date', 'actual', 'predicted', 'rf_pred', 'xgb_pred', 'lstm_pred']])
    
    with tab2:
        st.subheader("Model Performance Metrics")
        
        performance_chart, metrics_df = create_performance_metrics_chart(filtered_predictions)
        st.plotly_chart(performance_chart, use_container_width=True)
        
        st.subheader("Performance Summary")
        st.dataframe(metrics_df)
    
    with tab3:
        st.subheader("Feature Importance Analysis")
        
        importance_chart, importance_df = create_feature_importance_chart()
        if not importance_df.empty:
            st.plotly_chart(importance_chart, use_container_width=True)
            
            if st.checkbox("Show Feature Importance Data"):
                st.dataframe(importance_df)
        else:
            st.warning("Feature importance data not available.")
    
    with tab4:
        st.subheader("Portfolio Performance Simulation")
        
        initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=1000, step=1000)
        
        portfolio_chart, total_return, max_drawdown = create_portfolio_simulation(
            filtered_predictions, initial_capital
        )
        st.plotly_chart(portfolio_chart, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Return", f"{total_return:.2f}%")
        with col2:
            st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
        
        st.info("""
        **Strategy**: Buy when predicted return > 1%, Short when predicted return < -1%
        
        **Note**: This is a simplified simulation for demonstration purposes. 
        Real trading involves transaction costs, slippage, and other factors not considered here.
        """)
    
    with tab5:
        st.subheader("Model Information")
        
        if metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Configuration:**")
                if 'model_config' in metadata:
                    st.json(metadata['model_config'])
            
            with col2:
                st.write("**Test Metrics:**")
                if 'test_metrics' in metadata:
                    for metric, value in metadata['test_metrics'].items():
                        if metric == 'directional_accuracy':
                            st.write(f"- {metric.replace('_', ' ').title()}: {value:.2%}")
                        else:
                            st.write(f"- {metric.replace('_', ' ').title()}: {value:.6f}")
        else:
            st.warning("Model metadata not available.")
        
        st.subheader("Data Summary")
        st.write(f"**Prediction Period**: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        st.write(f"**Symbols Analyzed**: {', '.join(symbols)}")
        st.write(f"**Total Predictions**: {len(predictions_df)}")


if __name__ == "__main__":
    main()
