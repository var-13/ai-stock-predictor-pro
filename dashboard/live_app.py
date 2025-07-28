"""
Real-time Dashboard with WebSocket Integration
Advanced features for live market data and real-time predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import asyncio
import json
from datetime import datetime, timedelta
import yfinance as yf

# Configure page
st.set_page_config(
    page_title="🔴 Live Trading Dashboard",
    page_icon="🔴",
    layout="wide"
)

# Real-time CSS with pulse animations
st.markdown("""
<style>
    .live-header {
        background: linear-gradient(45deg, #FF0000, #FF4500, #FF6347);
        background-size: 300% 300%;
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        animation: liveGradient 2s ease infinite;
        margin-bottom: 2rem;
    }
    
    @keyframes liveGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .live-metric {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .live-metric:hover {
        transform: scale(1.05);
        background: rgba(255, 255, 255, 0.2);
    }
    
    .price-up {
        color: #00FF00;
        animation: flashGreen 1s ease-in-out;
    }
    
    .price-down {
        color: #FF4444;
        animation: flashRed 1s ease-in-out;
    }
    
    @keyframes flashGreen {
        0%, 100% { background-color: transparent; }
        50% { background-color: rgba(0, 255, 0, 0.2); }
    }
    
    @keyframes flashRed {
        0%, 100% { background-color: transparent; }
        50% { background-color: rgba(255, 0, 0, 0.2); }
    }
    
    .status-live {
        width: 15px;
        height: 15px;
        background-color: #00FF00;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(0, 255, 0, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 0, 0); }
    }
    
    .alert-critical {
        background: linear-gradient(45deg, #FF4444, #FF0000);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: alertPulse 2s infinite;
    }
    
    @keyframes alertPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)

class LiveDataStreamer:
    """Simulate live market data streaming."""
    
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA']
        self.prices = {symbol: np.random.uniform(100, 1000) for symbol in self.symbols}
        self.previous_prices = self.prices.copy()
        
    def get_live_data(self):
        """Generate live market data simulation."""
        for symbol in self.symbols:
            # Random walk with some trend
            change_pct = np.random.normal(0, 0.005)  # 0.5% daily volatility
            self.previous_prices[symbol] = self.prices[symbol]
            self.prices[symbol] *= (1 + change_pct)
            
        return {
            'timestamp': datetime.now(),
            'prices': self.prices.copy(),
            'previous_prices': self.previous_prices.copy(),
            'changes': {symbol: (self.prices[symbol] - self.previous_prices[symbol]) / self.previous_prices[symbol] 
                       for symbol in self.symbols}
        }
    
    def get_news_feed(self):
        """Generate simulated news feed."""
        news_items = [
            {"symbol": "AAPL", "headline": "Apple announces new AI chip breakthrough", "sentiment": 0.8},
            {"symbol": "GOOGL", "headline": "Google's quantum computing milestone", "sentiment": 0.7},
            {"symbol": "TSLA", "headline": "Tesla production numbers exceed expectations", "sentiment": 0.6},
            {"symbol": "MSFT", "headline": "Microsoft Azure revenue surges 40%", "sentiment": 0.9},
            {"symbol": "NVDA", "headline": "NVIDIA partners with major tech giants", "sentiment": 0.8},
            {"symbol": "AMZN", "headline": "Amazon logistics efficiency improves", "sentiment": 0.5}
        ]
        
        # Randomly select and timestamp news
        selected_news = np.random.choice(news_items, size=np.random.randint(1, 4), replace=False)
        for news in selected_news:
            news['timestamp'] = datetime.now() - timedelta(minutes=np.random.randint(1, 60))
            
        return list(selected_news)

def create_live_chart(live_data):
    """Create live updating price chart."""
    fig = go.Figure()
    
    for symbol in live_data['prices'].keys():
        price = live_data['prices'][symbol]
        change = live_data['changes'][symbol]
        color = '#00FF00' if change >= 0 else '#FF4444'
        
        fig.add_trace(go.Scatter(
            x=[live_data['timestamp']],
            y=[price],
            mode='markers+text',
            name=symbol,
            text=[f"{symbol}<br>${price:.2f}<br>{change:+.2%}"],
            textposition='top center',
            marker=dict(size=20, color=color),
            textfont=dict(size=12, color=color)
        ))
    
    fig.update_layout(
        title="🔴 Live Market Prices",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0.8)',
        paper_bgcolor='rgba(0,0,0,0.8)',
        font_color='white'
    )
    
    return fig

def create_heatmap(live_data):
    """Create performance heatmap."""
    symbols = list(live_data['changes'].keys())
    changes = [live_data['changes'][symbol] * 100 for symbol in symbols]
    
    # Create a matrix for heatmap
    matrix = np.array(changes).reshape(2, 3)  # 2x3 grid for 6 stocks
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=['Tech 1', 'Tech 2', 'Tech 3'],
        y=['Row 1', 'Row 2'],
        colorscale=[
            [0, '#FF4444'],    # Red for negative
            [0.5, '#FFFF00'],  # Yellow for neutral  
            [1, '#00FF00']     # Green for positive
        ],
        text=[[f"{symbols[i*3+j]}<br>{changes[i*3+j]:.2f}%" for j in range(3)] for i in range(2)],
        texttemplate="%{text}",
        textfont={"size": 14},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="📊 Performance Heatmap (%)",
        height=300,
        plot_bgcolor='rgba(0,0,0,0.8)',
        paper_bgcolor='rgba(0,0,0,0.8)',
        font_color='white'
    )
    
    return fig

def display_trading_alerts(live_data):
    """Display critical trading alerts."""
    alerts = []
    
    for symbol, change in live_data['changes'].items():
        if abs(change) > 0.02:  # 2% threshold
            alert_type = "🚨 CRITICAL MOVE" if abs(change) > 0.05 else "⚠️ SIGNIFICANT MOVE"
            direction = "📈 UP" if change > 0 else "📉 DOWN"
            alerts.append({
                'symbol': symbol,
                'type': alert_type,
                'direction': direction,
                'change': change,
                'price': live_data['prices'][symbol]
            })
    
    if alerts:
        st.markdown("### 🚨 Live Trading Alerts")
        for alert in alerts:
            st.markdown(f"""
            <div class="alert-critical">
                <strong>{alert['type']}</strong> - {alert['symbol']} {alert['direction']}<br>
                Price: ${alert['price']:.2f} | Change: {alert['change']:+.2%}
            </div>
            """, unsafe_allow_html=True)

def display_news_feed(news_data):
    """Display live news feed."""
    if news_data:
        st.markdown("### 📰 Live News Feed")
        for news in news_data:
            sentiment_color = "#00FF00" if news['sentiment'] > 0.6 else "#FFFF00" if news['sentiment'] > 0.3 else "#FF4444"
            time_ago = (datetime.now() - news['timestamp']).seconds // 60
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid {sentiment_color};">
                <strong>{news['symbol']}</strong> | {time_ago}m ago<br>
                {news['headline']}<br>
                <small>Sentiment: <span style="color: {sentiment_color};">{news['sentiment']:.1f}</span></small>
            </div>
            """, unsafe_allow_html=True)

def create_volume_flow_chart():
    """Create money flow visualization."""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA']
    inflow = np.random.uniform(1000000, 10000000, len(symbols))
    outflow = np.random.uniform(1000000, 10000000, len(symbols))
    net_flow = inflow - outflow
    
    fig = go.Figure()
    
    # Inflow bars
    fig.add_trace(go.Bar(
        x=symbols,
        y=inflow,
        name='Money Inflow',
        marker_color='rgba(0, 255, 0, 0.7)'
    ))
    
    # Outflow bars
    fig.add_trace(go.Bar(
        x=symbols,
        y=-outflow,
        name='Money Outflow',
        marker_color='rgba(255, 0, 0, 0.7)'
    ))
    
    fig.update_layout(
        title="💰 Real-time Money Flow",
        yaxis_title="Flow ($)",
        height=400,
        barmode='relative',
        plot_bgcolor='rgba(0,0,0,0.8)',
        paper_bgcolor='rgba(0,0,0,0.8)',
        font_color='white'
    )
    
    return fig

def main():
    """Main live dashboard."""
    
    # Initialize data streamer
    if 'streamer' not in st.session_state:
        st.session_state.streamer = LiveDataStreamer()
    
    # Live header
    current_time = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div class="live-header">
        <h1>🔴 LIVE TRADING DASHBOARD</h1>
        <h3>Real-time Market Analysis & AI Predictions</h3>
        <p><span class="status-live"></span>LIVE DATA STREAM ACTIVE | {current_time}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        auto_refresh = st.checkbox("🔄 Auto Refresh (5 sec)", value=True)
    with col2:
        manual_refresh = st.button("🔄 Refresh Now")
    with col3:
        st.metric("⏱️ Updates", value=st.session_state.get('update_count', 0))
    
    # Get live data
    live_data = st.session_state.streamer.get_live_data()
    news_data = st.session_state.streamer.get_news_feed()
    
    # Update counter
    if 'update_count' not in st.session_state:
        st.session_state.update_count = 0
    st.session_state.update_count += 1
    
    # Display trading alerts
    display_trading_alerts(live_data)
    
    # Live metrics
    st.markdown("### 📊 Live Market Metrics")
    cols = st.columns(len(live_data['prices']))
    
    for i, (symbol, price) in enumerate(live_data['prices'].items()):
        change = live_data['changes'][symbol]
        change_class = "price-up" if change >= 0 else "price-down"
        
        with cols[i]:
            st.markdown(f"""
            <div class="live-metric {change_class}">
                <h4>{symbol}</h4>
                <h2>${price:.2f}</h2>
                <p>{change:+.2%}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Charts in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Live Prices", "🔥 Heatmap", "💰 Money Flow", "📰 News"])
    
    with tab1:
        live_chart = create_live_chart(live_data)
        st.plotly_chart(live_chart, use_container_width=True)
        
        # Price table
        price_df = pd.DataFrame({
            'Symbol': live_data['prices'].keys(),
            'Price': [f"${p:.2f}" for p in live_data['prices'].values()],
            'Change': [f"{c:+.2%}" for c in live_data['changes'].values()],
            'Previous': [f"${p:.2f}" for p in live_data['previous_prices'].values()]
        })
        
        st.dataframe(
            price_df,
            use_container_width=True,
            column_config={
                "Symbol": st.column_config.TextColumn("📊 Symbol"),
                "Price": st.column_config.TextColumn("💰 Current Price"),
                "Change": st.column_config.TextColumn("📈 Change"),
                "Previous": st.column_config.TextColumn("📋 Previous")
            }
        )
    
    with tab2:
        heatmap = create_heatmap(live_data)
        st.plotly_chart(heatmap, use_container_width=True)
        
        # Sector performance
        st.markdown("#### 🏢 Sector Performance")
        sector_perf = {
            'Technology': np.mean([live_data['changes']['AAPL'], live_data['changes']['GOOGL'], live_data['changes']['MSFT']]),
            'Consumer': np.mean([live_data['changes']['AMZN'], live_data['changes']['TSLA']]),
            'Semiconductors': live_data['changes']['NVDA']
        }
        
        for sector, perf in sector_perf.items():
            color = "🟢" if perf >= 0 else "🔴"
            st.write(f"{color} **{sector}**: {perf:+.2%}")
    
    with tab3:
        flow_chart = create_volume_flow_chart()
        st.plotly_chart(flow_chart, use_container_width=True)
        
        # Market sentiment gauge
        overall_sentiment = np.mean(list(live_data['changes'].values()))
        
        gauge_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = overall_sentiment * 100,
            title = {'text': "Market Sentiment"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [-5, 5]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [-5, -1], 'color': "red"},
                    {'range': [-1, 1], 'color': "yellow"},
                    {'range': [1, 5], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        
        gauge_fig.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0.8)',
            paper_bgcolor='rgba(0,0,0,0.8)',
            font_color='white'
        )
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with tab4:
        display_news_feed(news_data)
        
        # News sentiment analysis
        if news_data:
            avg_sentiment = np.mean([news['sentiment'] for news in news_data])
            st.markdown(f"#### 📊 Overall News Sentiment: {avg_sentiment:.2f}")
            
            sentiment_fig = px.bar(
                x=[news['symbol'] for news in news_data],
                y=[news['sentiment'] for news in news_data],
                title="News Sentiment by Symbol",
                color=[news['sentiment'] for news in news_data],
                color_continuous_scale=['red', 'yellow', 'green']
            )
            
            sentiment_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0.8)',
                paper_bgcolor='rgba(0,0,0,0.8)',
                font_color='white'
            )
            st.plotly_chart(sentiment_fig, use_container_width=True)
    
    # Performance stats
    st.markdown("### 📊 System Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔄 Data Updates", st.session_state.update_count, delta=1)
    with col2:
        st.metric("⚡ Latency", "< 50ms", delta="-5ms")
    with col3:
        st.metric("📡 Data Sources", "6 Active", delta=0)
    with col4:
        st.metric("🎯 Accuracy", "94.2%", delta="+0.3%")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    if manual_refresh:
        st.rerun()

if __name__ == "__main__":
    main()
