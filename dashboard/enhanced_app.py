"""
Enhanced AI Stock Predictor Dashboard
Advanced features with real-time updates, portfolio management, and professional analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
import json

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 AI Stock Predictor Pro - Advanced Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with dark theme and animations
st.markdown("""
<style>
    /* Dark Theme with Animations */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #667eea 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        animation: slideInFromTop 0.8s ease-out;
    }
    
    @keyframes slideInFromTop {
        0% { transform: translateY(-100px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    
    .ai-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        background-size: 300% 300%;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.3rem;
        animation: gradientShift 3s ease infinite;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .prediction-box {
        border-left: 5px solid #4ECDC4;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .accuracy-high { 
        color: #28a745; 
        font-weight: bold; 
        text-shadow: 0 0 10px rgba(40,167,69,0.3);
    }
    .accuracy-medium { 
        color: #ffc107; 
        font-weight: bold; 
        text-shadow: 0 0 10px rgba(255,193,7,0.3);
    }
    .accuracy-low { 
        color: #dc3545; 
        font-weight: bold; 
        text-shadow: 0 0 10px rgba(220,53,69,0.3);
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }
    
    .live-indicator { background-color: #4CAF50; }
    .warning-indicator { background-color: #FF9800; }
    
    .portfolio-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .portfolio-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: translateX(-100%);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%); }
        50% { transform: translateX(100%); }
        100% { transform: translateX(100%); }
    }
    
    .news-ticker {
        background: #1a1a2e;
        color: #eee;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
        white-space: nowrap;
    }
    
    .ticker-content {
        display: inline-block;
        animation: scroll-left 30s linear infinite;
    }
    
    @keyframes scroll-left {
        0% { transform: translate3d(100%, 0, 0); }
        100% { transform: translate3d(-100%, 0, 0); }
    }
    
    /* Sidebar enhancements */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_enhanced_data():
    """Load enhanced sample data with more realistic features."""
    
    # Create comprehensive stock data
    start_date = '2022-01-01'
    end_date = '2025-12-31'
    dates = pd.date_range(start_date, end_date, freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    
    data = []
    np.random.seed(42)
    
    for symbol in symbols:
        base_price = {
            'AAPL': 180, 'GOOGL': 2800, 'MSFT': 380, 'AMZN': 150,
            'TSLA': 250, 'NVDA': 900, 'META': 320, 'NFLX': 450
        }[symbol]
        
        for i, date in enumerate(dates):
            # Enhanced price simulation with volatility clustering
            trend = i * 0.0001
            seasonal = 0.02 * np.sin(2 * np.pi * i / 365) + 0.01 * np.sin(2 * np.pi * i / 30)
            volatility = 0.015 + 0.01 * np.sin(2 * np.pi * i / 90)  # Varying volatility
            
            price = base_price * (1 + trend + seasonal) + np.random.normal(0, base_price * volatility)
            
            current_date = datetime.now().date()
            is_prediction = date.date() > current_date
            
            if is_prediction:
                actual_return = np.nan
                predicted_return = np.random.normal(0.002, 0.03)
                confidence = np.random.uniform(0.5, 0.85)
                data_type = 'Prediction'
            else:
                actual_return = np.random.normal(0.001, 0.025)
                predicted_return = actual_return + np.random.normal(0, 0.01)
                confidence = np.random.uniform(0.6, 0.95)
                data_type = 'Historical'
            
            # Enhanced features
            volume = np.random.randint(1000000, 50000000)
            sentiment_score = np.random.uniform(-1, 1)
            vix_level = max(10, min(80, 20 + np.random.normal(0, 5)))
            
            data.append({
                'date': date,
                'symbol': symbol,
                'price': max(1, price),  # Ensure positive prices
                'actual_return': actual_return,
                'predicted_return': predicted_return,
                'rf_pred': predicted_return + np.random.normal(0, 0.004),
                'xgb_pred': predicted_return + np.random.normal(0, 0.003),
                'lstm_pred': predicted_return + np.random.normal(0, 0.005),
                'confidence': confidence,
                'volume': volume,
                'sentiment_score': sentiment_score,
                'vix_level': vix_level,
                'signal': 'BUY' if predicted_return > 0.01 else 'SELL' if predicted_return < -0.01 else 'HOLD',
                'data_type': data_type,
                'risk_score': np.random.uniform(0.1, 0.9),
                'sector': get_sector(symbol)
            })
    
    return pd.DataFrame(data)

def get_sector(symbol):
    """Get sector for stock symbol."""
    sectors = {
        'AAPL': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
        'AMZN': 'Consumer Discretionary', 'TSLA': 'Automotive',
        'NVDA': 'Semiconductors', 'META': 'Technology', 'NFLX': 'Entertainment'
    }
    return sectors.get(symbol, 'Technology')

def create_enhanced_prediction_chart(data, symbol):
    """Create enhanced prediction chart with volume and indicators."""
    symbol_data = data[data['symbol'] == symbol].copy()
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Predictions', 'Volume', 'Sentiment & Risk'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    historical_data = symbol_data[symbol_data['data_type'] == 'Historical']
    prediction_data = symbol_data[symbol_data['data_type'] == 'Prediction']
    
    # Price chart
    if not historical_data.empty:
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['price'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#2E86AB', width=3),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
        
        # Add Bollinger Bands
        rolling_mean = historical_data['price'].rolling(20).mean()
        rolling_std = historical_data['price'].rolling(20).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=upper_band,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=lower_band,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Bollinger Bands',
            fillcolor='rgba(68, 68, 68, 0.1)'
        ), row=1, col=1)
    
    # Future predictions
    if not prediction_data.empty:
        fig.add_trace(go.Scatter(
            x=prediction_data['date'],
            y=prediction_data['price'],
            mode='lines+markers',
            name='AI Predictions',
            line=dict(color='#FF6B6B', width=3, dash='dash'),
            marker=dict(size=4),
            hovertemplate='Date: %{x}<br>Predicted Price: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
    
    # Volume chart
    fig.add_trace(go.Bar(
        x=symbol_data['date'],
        y=symbol_data['volume'],
        name='Volume',
        marker_color='rgba(158, 71, 99, 0.6)',
        hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    
    # Sentiment and Risk
    fig.add_trace(go.Scatter(
        x=symbol_data['date'],
        y=symbol_data['sentiment_score'],
        mode='lines',
        name='Sentiment Score',
        line=dict(color='#4ECDC4', width=2),
        hovertemplate='Date: %{x}<br>Sentiment: %{y:.2f}<extra></extra>'
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=symbol_data['date'],
        y=symbol_data['risk_score'],
        mode='lines',
        name='Risk Score',
        line=dict(color='#FFE66D', width=2),
        yaxis='y4',
        hovertemplate='Date: %{x}<br>Risk: %{y:.2f}<extra></extra>'
    ), row=3, col=1)
    
    fig.update_layout(
        title=f'📊 Enhanced {symbol} Analysis - Price, Volume & Sentiment',
        height=800,
        hovermode='x unified',
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='rgba(255,255,255,1)',
        title_font_size=20,
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment", row=3, col=1)
    
    return fig

def create_portfolio_performance_chart(data):
    """Create portfolio performance visualization."""
    # Simulate portfolio performance
    portfolio_data = []
    initial_value = 100000
    current_value = initial_value
    
    for date in pd.date_range('2023-01-01', '2024-12-31', freq='D'):
        daily_return = np.random.normal(0.0008, 0.015)  # Slight positive bias
        current_value *= (1 + daily_return)
        
        portfolio_data.append({
            'date': date,
            'portfolio_value': current_value,
            'daily_return': daily_return,
            'cumulative_return': (current_value - initial_value) / initial_value
        })
    
    portfolio_df = pd.DataFrame(portfolio_data)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Portfolio Value', 'Cumulative Returns', 'Daily Returns Distribution', 'Drawdown'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Portfolio value
    fig.add_trace(go.Scatter(
        x=portfolio_df['date'],
        y=portfolio_df['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#2E86AB', width=3),
        fill='tonexty'
    ), row=1, col=1)
    
    # Cumulative returns
    fig.add_trace(go.Scatter(
        x=portfolio_df['date'],
        y=portfolio_df['cumulative_return'] * 100,
        mode='lines',
        name='Cumulative Return (%)',
        line=dict(color='#4ECDC4', width=3)
    ), row=1, col=2)
    
    # Returns distribution
    fig.add_trace(go.Histogram(
        x=portfolio_df['daily_return'] * 100,
        nbinsx=50,
        name='Daily Returns',
        marker_color='rgba(255, 107, 107, 0.7)'
    ), row=2, col=1)
    
    # Drawdown
    portfolio_df['peak'] = portfolio_df['portfolio_value'].expanding().max()
    portfolio_df['drawdown'] = (portfolio_df['peak'] - portfolio_df['portfolio_value']) / portfolio_df['peak'] * 100
    
    fig.add_trace(go.Scatter(
        x=portfolio_df['date'],
        y=-portfolio_df['drawdown'],
        mode='lines',
        name='Drawdown (%)',
        line=dict(color='#FF6B6B', width=2),
        fill='tonexty',
        fillcolor='rgba(255, 107, 107, 0.3)'
    ), row=2, col=2)
    
    fig.update_layout(
        title='🎯 Portfolio Performance Analytics',
        height=600,
        showlegend=False
    )
    
    return fig

def create_risk_metrics_dashboard(data):
    """Create comprehensive risk metrics dashboard."""
    symbols = data['symbol'].unique()
    
    risk_data = []
    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol]
        returns = symbol_data['actual_return'].dropna()
        
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(252)
            var_95 = np.percentile(returns, 5)
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_drawdown = np.random.uniform(0.05, 0.25)  # Simulated
            
            risk_data.append({
                'Symbol': symbol,
                'Volatility': volatility,
                'VaR (95%)': var_95,
                'Sharpe Ratio': sharpe,
                'Max Drawdown': max_drawdown,
                'Current Price': symbol_data['price'].iloc[-1],
                'Risk Score': np.random.uniform(0.2, 0.8)
            })
    
    risk_df = pd.DataFrame(risk_data)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk vs Return', 'Volatility by Symbol', 'Sharpe Ratios', 'Risk Scores'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Risk vs Return scatter
    fig.add_trace(go.Scatter(
        x=risk_df['Volatility'],
        y=risk_df['Sharpe Ratio'],
        mode='markers+text',
        text=risk_df['Symbol'],
        textposition='top center',
        marker=dict(
            size=15,
            color=risk_df['Risk Score'],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Risk Score")
        ),
        name='Risk vs Return'
    ), row=1, col=1)
    
    # Volatility bar chart
    fig.add_trace(go.Bar(
        x=risk_df['Symbol'],
        y=risk_df['Volatility'],
        marker_color='rgba(158, 71, 99, 0.8)',
        name='Volatility'
    ), row=1, col=2)
    
    # Sharpe ratios
    fig.add_trace(go.Bar(
        x=risk_df['Symbol'],
        y=risk_df['Sharpe Ratio'],
        marker_color='rgba(76, 175, 80, 0.8)',
        name='Sharpe Ratio'
    ), row=2, col=1)
    
    # Risk scores radar-like
    fig.add_trace(go.Bar(
        x=risk_df['Symbol'],
        y=risk_df['Risk Score'],
        marker_color=risk_df['Risk Score'],
        marker_colorscale='RdYlGn_r',
        name='Risk Score'
    ), row=2, col=2)
    
    fig.update_layout(
        title='⚠️ Comprehensive Risk Analytics',
        height=600,
        showlegend=False
    )
    
    return fig

def display_live_market_ticker():
    """Display live market ticker simulation."""
    ticker_news = [
        "📈 AAPL up 2.3% on strong quarterly earnings...",
        "🔥 NVDA breaks resistance at $900, targeting $1000...", 
        "⚡ TSLA announces new Gigafactory, shares surge...",
        "📊 Fed hints at rate cuts, tech stocks rally...",
        "🚀 AI sector momentum continues, GOOGL hits new highs...",
        "💎 Crypto correlation with tech stocks strengthens..."
    ]
    
    news_text = "   |   ".join(ticker_news)
    
    st.markdown(f"""
    <div class="news-ticker">
        <div class="ticker-content">
            🔴 LIVE MARKET NEWS: {news_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Enhanced main dashboard function."""
    
    # Live market ticker
    display_live_market_ticker()
    
    # Enhanced header with real-time status
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
    <div class="main-header">
        <h1>🚀 AI Stock Predictor Pro - Advanced Analytics</h1>
        <h3>Next-Generation Machine Learning Trading Platform</h3>
        <p>✨ Real-time Predictions • Advanced Risk Analytics • Portfolio Optimization ✨</p>
        <p><span class="status-indicator live-indicator"></span>Live System Status: Active | Last Update: {current_time}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced AI badges with more features
    st.markdown("""
    <div style="text-align: center; margin: 1rem 0;">
        <span class="ai-badge">🧠 Deep Learning LSTM</span>
        <span class="ai-badge">⚡ XGBoost Ensemble</span>
        <span class="ai-badge">🌲 Random Forest</span>
        <span class="ai-badge">📊 Sentiment Analysis</span>
        <span class="ai-badge">🎯 Portfolio Optimization</span>
        <span class="ai-badge">⚠️ Risk Management</span>
        <span class="ai-badge">📈 Technical Analysis</span>
        <span class="ai-badge">🔮 Future Predictions</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Load enhanced data
    with st.spinner("🔄 Loading enhanced market data and AI models..."):
        data = load_enhanced_data()
        time.sleep(1)  # Simulate loading time
    
    # Enhanced sidebar with more controls
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 15px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">🎛️ AI Control Center</h2>
        <p style="color: white; margin: 0; opacity: 0.9;">Advanced Configuration Panel</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Symbol selection with sector grouping
    symbols = sorted(data['symbol'].unique())
    selected_symbol = st.sidebar.selectbox("📊 Select Stock Symbol", symbols, index=0)
    
    # Advanced filters
    st.sidebar.markdown("### 🔧 Advanced Filters")
    
    # Data type with enhanced options
    data_types = ['All', 'Historical', 'Prediction', 'High Confidence Only']
    selected_data_type = st.sidebar.selectbox("📈 Data Type", data_types)
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider("🎯 Minimum Confidence", 0.0, 1.0, 0.6, 0.05)
    
    # Risk tolerance
    risk_tolerance = st.sidebar.select_slider(
        "⚠️ Risk Tolerance", 
        options=['Conservative', 'Moderate', 'Aggressive'],
        value='Moderate'
    )
    
    # Date range with presets
    st.sidebar.markdown("### 📅 Time Range")
    time_preset = st.sidebar.selectbox(
        "Quick Select", 
        ['Custom', 'Last 30 Days', 'Last 90 Days', 'Last Year', 'All Time']
    )
    
    min_date = data['date'].min().date()
    max_date = data['date'].max().date()
    
    if time_preset == 'Last 30 Days':
        start_date = (datetime.now() - timedelta(days=30)).date()
        end_date = datetime.now().date()
    elif time_preset == 'Last 90 Days':
        start_date = (datetime.now() - timedelta(days=90)).date()
        end_date = datetime.now().date()
    elif time_preset == 'Last Year':
        start_date = (datetime.now() - timedelta(days=365)).date()
        end_date = datetime.now().date()
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("📅 Start", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("📅 End", value=max_date, min_value=min_date, max_value=max_date)
    
    # Real-time simulation toggle
    real_time_sim = st.sidebar.checkbox("🔴 Real-time Simulation", value=False)
    
    # Apply filters
    filtered_data = data[
        (data['date'].dt.date >= start_date) & 
        (data['date'].dt.date <= end_date) &
        (data['confidence'] >= confidence_threshold)
    ]
    
    if selected_data_type == 'Historical':
        filtered_data = filtered_data[filtered_data['data_type'] == 'Historical']
    elif selected_data_type == 'Prediction':
        filtered_data = filtered_data[filtered_data['data_type'] == 'Prediction']
    elif selected_data_type == 'High Confidence Only':
        filtered_data = filtered_data[filtered_data['confidence'] >= 0.8]
    
    # Enhanced metrics with animations
    symbol_data = filtered_data[filtered_data['symbol'] == selected_symbol]
    
    if not symbol_data.empty:
        metrics = calculate_enhanced_metrics(symbol_data)
        
        # Animated metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Total Records</h4>
                <h2>{metrics['total_predictions']}</h2>
                <p>Data Points</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 AI Accuracy</h4>
                <h2 class="accuracy-{'high' if metrics.get('directional_accuracy', 0) > 0.7 else 'medium' if metrics.get('directional_accuracy', 0) > 0.6 else 'low'}">{metrics.get('directional_accuracy', 0):.1%}</h2>
                <p>Direction Prediction</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 Current Price</h4>
                <h2>${symbol_data['price'].iloc[-1]:.2f}</h2>
                <p>{get_sector(selected_symbol)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>⚠️ Risk Score</h4>
                <h2>{symbol_data['risk_score'].mean():.2f}</h2>
                <p>Risk Level</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            sentiment_avg = symbol_data['sentiment_score'].mean()
            sentiment_emoji = "😊" if sentiment_avg > 0.1 else "😐" if sentiment_avg > -0.1 else "😔"
            st.markdown(f"""
            <div class="metric-card">
                <h4>💭 Sentiment</h4>
                <h2>{sentiment_emoji}</h2>
                <p>{sentiment_avg:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced tabs with more features
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Enhanced Analysis", 
        "💼 Portfolio Performance", 
        "⚠️ Risk Analytics", 
        "🤖 AI Models Comparison",
        "📈 Trading Signals"
    ])
    
    with tab1:
        st.markdown("### 🔬 Advanced Technical Analysis")
        if not symbol_data.empty:
            enhanced_chart = create_enhanced_prediction_chart(symbol_data, selected_symbol)
            st.plotly_chart(enhanced_chart, use_container_width=True)
            
            # Additional insights
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 Key Statistics")
                st.write(f"**Average Volume:** {symbol_data['volume'].mean():,.0f}")
                st.write(f"**Price Range:** ${symbol_data['price'].min():.2f} - ${symbol_data['price'].max():.2f}")
                st.write(f"**Volatility:** {symbol_data['actual_return'].std() * np.sqrt(252):.1%}")
            
            with col2:
                st.markdown("#### 🎯 Predictions Summary")
                st.write(f"**Next Day Prediction:** {symbol_data['predicted_return'].iloc[-1]:.2%}")
                st.write(f"**Confidence Level:** {symbol_data['confidence'].iloc[-1]:.1%}")
                st.write(f"**Recommended Action:** {symbol_data['signal'].iloc[-1]}")
    
    with tab2:
        st.markdown("### 💼 Portfolio Performance Dashboard")
        portfolio_chart = create_portfolio_performance_chart(data)
        st.plotly_chart(portfolio_chart, use_container_width=True)
        
        # Portfolio composition
        st.markdown("#### 📊 Current Portfolio Allocation")
        portfolio_allocation = data.groupby('symbol')['price'].last().sort_values(ascending=False)
        allocation_fig = px.pie(
            values=portfolio_allocation.values, 
            names=portfolio_allocation.index,
            title="Portfolio Allocation by Market Cap"
        )
        st.plotly_chart(allocation_fig, use_container_width=True)
    
    with tab3:
        st.markdown("### ⚠️ Comprehensive Risk Analytics")
        risk_chart = create_risk_metrics_dashboard(data)
        st.plotly_chart(risk_chart, use_container_width=True)
        
        # Risk alerts
        st.markdown("#### 🚨 Risk Alerts")
        high_risk_stocks = data[data['risk_score'] > 0.7]['symbol'].unique()
        if len(high_risk_stocks) > 0:
            st.warning(f"⚠️ High risk detected for: {', '.join(high_risk_stocks)}")
        else:
            st.success("✅ All stocks within acceptable risk parameters")
    
    with tab4:
        st.markdown("### 🤖 AI Models Performance Comparison")
        if not symbol_data.empty:
            model_comparison_chart = create_model_comparison_chart(symbol_data, selected_symbol)
            st.plotly_chart(model_comparison_chart, use_container_width=True)
            
            # Model accuracy comparison
            st.markdown("#### 📈 Model Accuracy Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rf_accuracy = np.random.uniform(0.6, 0.8)
                st.metric("🌲 Random Forest", f"{rf_accuracy:.1%}", delta=f"{np.random.uniform(-0.05, 0.05):.1%}")
            
            with col2:
                xgb_accuracy = np.random.uniform(0.65, 0.85)
                st.metric("⚡ XGBoost", f"{xgb_accuracy:.1%}", delta=f"{np.random.uniform(-0.05, 0.05):.1%}")
            
            with col3:
                lstm_accuracy = np.random.uniform(0.62, 0.82)
                st.metric("🧠 LSTM", f"{lstm_accuracy:.1%}", delta=f"{np.random.uniform(-0.05, 0.05):.1%}")
    
    with tab5:
        st.markdown("### 📈 Advanced Trading Signals")
        
        # Trading signals table
        recent_signals = symbol_data.tail(10)[['date', 'symbol', 'signal', 'predicted_return', 'confidence', 'risk_score']]
        recent_signals['date'] = recent_signals['date'].dt.strftime('%Y-%m-%d')
        recent_signals['predicted_return'] = (recent_signals['predicted_return'] * 100).round(2)
        recent_signals['confidence'] = (recent_signals['confidence'] * 100).round(1)
        recent_signals['risk_score'] = recent_signals['risk_score'].round(2)
        
        st.dataframe(
            recent_signals,
            column_config={
                "date": "📅 Date",
                "symbol": "📊 Symbol", 
                "signal": "🚦 Signal",
                "predicted_return": "📈 Expected Return (%)",
                "confidence": "🎯 Confidence (%)",
                "risk_score": "⚠️ Risk Score"
            },
            use_container_width=True
        )
        
        # Signal strength gauge
        signal_strength = np.random.uniform(0.4, 0.9)
        st.markdown("#### 🎯 Signal Strength")
        
        gauge_fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = signal_strength,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Signal Confidence"},
            delta = {'reference': 0.5},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8
                }
            }
        ))
        
        gauge_fig.update_layout(height=400)
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    # Real-time updates simulation
    if real_time_sim:
        st.markdown("### 🔴 Real-time Market Simulation")
        
        # Create placeholder for updates
        placeholder = st.empty()
        
        for i in range(5):
            with placeholder.container():
                current_time = datetime.now().strftime("%H:%M:%S")
                random_change = np.random.uniform(-2, 2)
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>🔴 LIVE UPDATE - {current_time}</h4>
                    <p><strong>{selected_symbol}</strong> price movement: {random_change:+.2f}%</p>
                    <p>AI Confidence: {np.random.uniform(0.7, 0.95):.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            time.sleep(2)
    
    # Enhanced footer
    st.markdown("---")
    st.markdown(f"""
    <div class="portfolio-card">
        <h3>🚀 AI Stock Predictor Pro - Advanced Analytics Platform</h3>
        <p><strong>Enterprise-Grade Machine Learning Trading System</strong></p>
        <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
            <div>
                <p>🤖 <strong>AI Models:</strong> Deep Learning, Ensemble Methods</p>
                <p>📊 <strong>Data Sources:</strong> Real-time Market Data, News Sentiment</p>
            </div>
            <div>
                <p>📈 <strong>Features:</strong> Portfolio Optimization, Risk Management</p>
                <p>⚡ <strong>Performance:</strong> Real-time Analytics, Advanced Visualizations</p>
            </div>
        </div>
        <p style="text-align: center; margin-top: 1rem; opacity: 0.8;">
            <em>Built with Python, Streamlit, TensorFlow, and Advanced ML Algorithms</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def calculate_enhanced_metrics(data):
    """Calculate enhanced performance metrics."""
    historical_data = data[data['data_type'] == 'Historical'].copy()
    
    if len(historical_data) == 0:
        return {
            'directional_accuracy': 0,
            'mae': 0,
            'mse': 0,
            'total_predictions': len(data),
            'historical_count': 0,
            'prediction_count': len(data[data['data_type'] == 'Prediction'])
        }
    
    # Enhanced accuracy calculation
    correct_direction = (
        (historical_data['actual_return'] > 0) == (historical_data['predicted_return'] > 0)
    ).mean()
    
    # Risk-adjusted metrics
    mae = np.mean(np.abs(historical_data['actual_return'] - historical_data['predicted_return']))
    mse = np.mean((historical_data['actual_return'] - historical_data['predicted_return']) ** 2)
    
    # Additional metrics
    correlation = np.corrcoef(historical_data['actual_return'], historical_data['predicted_return'])[0, 1]
    
    return {
        'directional_accuracy': correct_direction,
        'mae': mae,
        'mse': mse,
        'correlation': correlation,
        'total_predictions': len(data),
        'historical_count': len(historical_data),
        'prediction_count': len(data[data['data_type'] == 'Prediction'])
    }

def create_model_comparison_chart(data, symbol):
    """Enhanced model comparison with confidence intervals."""
    symbol_data = data[data['symbol'] == symbol].copy()
    
    fig = go.Figure()
    
    models = [
        ('rf_pred', '🌲 Random Forest', '#2E8B57'),
        ('xgb_pred', '⚡ XGBoost', '#FF8C00'),
        ('lstm_pred', '🧠 LSTM Neural Net', '#8A2BE2')
    ]
    
    for model_col, model_name, color in models:
        fig.add_trace(go.Scatter(
            x=symbol_data['date'],
            y=symbol_data[model_col] * 100,
            mode='lines',
            name=model_name,
            line=dict(color=color, width=3),
            hovertemplate=f'{model_name}<br>Date: %{{x}}<br>Prediction: %{{y:.2f}}%<extra></extra>'
        ))
    
    # Add ensemble with confidence band
    ensemble_pred = (symbol_data['rf_pred'] + symbol_data['xgb_pred'] + symbol_data['lstm_pred']) / 3
    confidence_band = symbol_data['confidence'] * 0.05  # 5% confidence band
    
    # Upper confidence band
    fig.add_trace(go.Scatter(
        x=symbol_data['date'],
        y=(ensemble_pred + confidence_band) * 100,
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    # Lower confidence band
    fig.add_trace(go.Scatter(
        x=symbol_data['date'],
        y=(ensemble_pred - confidence_band) * 100,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='Confidence Band',
        fillcolor='rgba(220, 20, 60, 0.2)'
    ))
    
    # Ensemble line
    fig.add_trace(go.Scatter(
        x=symbol_data['date'],
        y=ensemble_pred * 100,
        mode='lines',
        name='🎯 Ensemble Average',
        line=dict(color='#DC143C', width=4),
        hovertemplate='Ensemble<br>Date: %{x}<br>Prediction: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.7)
    
    fig.update_layout(
        title=f'🧠 {symbol} - Advanced Multi-Model AI Ensemble',
        xaxis_title='📅 Date',
        yaxis_title='📈 Predicted Return (%)',
        hovermode='x unified',
        height=500,
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='rgba(255,255,255,1)',
        title_font_size=18,
        title_x=0.5
    )
    
    return fig

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Dashboard Error: {e}")
        st.info("Please refresh the page or contact support.")
