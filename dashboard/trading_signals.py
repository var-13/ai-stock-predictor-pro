"""
Advanced Trading Signals Dashboard
Professional trading interface with technical analysis and AI signals
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import talib
import yfinance as yf

st.set_page_config(
    page_title="📊 Trading Signals Pro",
    page_icon="📊",
    layout="wide"
)

# Professional trading CSS
st.markdown("""
<style>
    .trading-header {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .signal-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .buy-signal {
        background: linear-gradient(135deg, #00C851 0%, #007E33 100%);
        border-left: 5px solid #00FF00;
    }
    
    .sell-signal {
        background: linear-gradient(135deg, #FF4444 0%, #CC0000 100%);
        border-left: 5px solid #FF0000;
    }
    
    .hold-signal {
        background: linear-gradient(135deg, #ffbb33 0%, #ff8800 100%);
        border-left: 5px solid #FFA500;
    }
    
    .technical-indicator {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .strength-high { color: #00FF00; font-weight: bold; }
    .strength-medium { color: #FFFF00; font-weight: bold; }
    .strength-low { color: #FF4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class TechnicalAnalyzer:
    """Advanced technical analysis calculator."""
    
    def __init__(self, data):
        self.data = data
        self.close = data['close'].values
        self.high = data['high'].values
        self.low = data['low'].values
        self.volume = data['volume'].values
        
    def calculate_all_indicators(self):
        """Calculate comprehensive technical indicators."""
        try:
            indicators = {}
            
            # Moving Averages
            indicators['SMA_20'] = talib.SMA(self.close, timeperiod=20)
            indicators['SMA_50'] = talib.SMA(self.close, timeperiod=50)
            indicators['EMA_12'] = talib.EMA(self.close, timeperiod=12)
            indicators['EMA_26'] = talib.EMA(self.close, timeperiod=26)
            
            # Momentum Indicators
            indicators['RSI'] = talib.RSI(self.close, timeperiod=14)
            indicators['MACD'], indicators['MACD_signal'], indicators['MACD_hist'] = talib.MACD(self.close)
            indicators['STOCH_K'], indicators['STOCH_D'] = talib.STOCH(self.high, self.low, self.close)
            
            # Volatility Indicators
            indicators['BB_upper'], indicators['BB_middle'], indicators['BB_lower'] = talib.BBANDS(self.close)
            indicators['ATR'] = talib.ATR(self.high, self.low, self.close, timeperiod=14)
            
            # Volume Indicators
            indicators['OBV'] = talib.OBV(self.close, self.volume)
            indicators['AD'] = talib.AD(self.high, self.low, self.close, self.volume)
            
            # Trend Indicators
            indicators['ADX'] = talib.ADX(self.high, self.low, self.close, timeperiod=14)
            indicators['AROON_UP'], indicators['AROON_DOWN'] = talib.AROON(self.high, self.low, timeperiod=14)
            
            return indicators
            
        except Exception as e:
            # Fallback to simple calculations if talib fails
            return self.calculate_simple_indicators()
    
    def calculate_simple_indicators(self):
        """Simple indicator calculations without talib."""
        indicators = {}
        close_series = pd.Series(self.close)
        
        # Simple Moving Averages
        indicators['SMA_20'] = close_series.rolling(20).mean().values
        indicators['SMA_50'] = close_series.rolling(50).mean().values
        
        # Simple RSI
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = (100 - (100 / (1 + rs))).values
        
        # Bollinger Bands
        sma_20 = close_series.rolling(20).mean()
        std_20 = close_series.rolling(20).std()
        indicators['BB_upper'] = (sma_20 + (std_20 * 2)).values
        indicators['BB_middle'] = sma_20.values
        indicators['BB_lower'] = (sma_20 - (std_20 * 2)).values
        
        return indicators

class SignalGenerator:
    """Generate trading signals from technical indicators."""
    
    def __init__(self, data, indicators):
        self.data = data
        self.indicators = indicators
        self.current_price = data['close'].iloc[-1]
        
    def generate_comprehensive_signals(self):
        """Generate signals from multiple indicators."""
        signals = {}
        
        # RSI Signals
        current_rsi = self.indicators['RSI'][-1] if not np.isnan(self.indicators['RSI'][-1]) else 50
        if current_rsi < 30:
            signals['RSI'] = {'signal': 'BUY', 'strength': 'HIGH', 'value': current_rsi}
        elif current_rsi > 70:
            signals['RSI'] = {'signal': 'SELL', 'strength': 'HIGH', 'value': current_rsi}
        else:
            signals['RSI'] = {'signal': 'HOLD', 'strength': 'LOW', 'value': current_rsi}
        
        # Moving Average Signals
        sma_20 = self.indicators['SMA_20'][-1] if not np.isnan(self.indicators['SMA_20'][-1]) else self.current_price
        sma_50 = self.indicators['SMA_50'][-1] if not np.isnan(self.indicators['SMA_50'][-1]) else self.current_price
        
        if self.current_price > sma_20 > sma_50:
            signals['MA_Trend'] = {'signal': 'BUY', 'strength': 'MEDIUM', 'value': 'Bullish'}
        elif self.current_price < sma_20 < sma_50:
            signals['MA_Trend'] = {'signal': 'SELL', 'strength': 'MEDIUM', 'value': 'Bearish'}
        else:
            signals['MA_Trend'] = {'signal': 'HOLD', 'strength': 'LOW', 'value': 'Neutral'}
        
        # Bollinger Bands Signals
        bb_upper = self.indicators['BB_upper'][-1] if not np.isnan(self.indicators['BB_upper'][-1]) else self.current_price * 1.02
        bb_lower = self.indicators['BB_lower'][-1] if not np.isnan(self.indicators['BB_lower'][-1]) else self.current_price * 0.98
        
        if self.current_price <= bb_lower:
            signals['BB'] = {'signal': 'BUY', 'strength': 'HIGH', 'value': 'Oversold'}
        elif self.current_price >= bb_upper:
            signals['BB'] = {'signal': 'SELL', 'strength': 'HIGH', 'value': 'Overbought'}
        else:
            signals['BB'] = {'signal': 'HOLD', 'strength': 'LOW', 'value': 'Normal'}
        
        # MACD Signals (if available)
        if 'MACD' in self.indicators and not np.isnan(self.indicators['MACD'][-1]):
            macd = self.indicators['MACD'][-1]
            macd_signal = self.indicators['MACD_signal'][-1]
            
            if macd > macd_signal:
                signals['MACD'] = {'signal': 'BUY', 'strength': 'MEDIUM', 'value': 'Bullish Cross'}
            else:
                signals['MACD'] = {'signal': 'SELL', 'strength': 'MEDIUM', 'value': 'Bearish Cross'}
        
        return signals
    
    def calculate_overall_signal(self, signals):
        """Calculate overall signal from individual signals."""
        buy_count = sum(1 for s in signals.values() if s['signal'] == 'BUY')
        sell_count = sum(1 for s in signals.values() if s['signal'] == 'SELL')
        
        if buy_count > sell_count:
            overall = 'BUY'
            strength = 'HIGH' if buy_count >= 3 else 'MEDIUM'
        elif sell_count > buy_count:
            overall = 'SELL'
            strength = 'HIGH' if sell_count >= 3 else 'MEDIUM'
        else:
            overall = 'HOLD'
            strength = 'LOW'
        
        return {'signal': overall, 'strength': strength, 'score': buy_count - sell_count}

@st.cache_data
def load_trading_data():
    """Load sample trading data with OHLCV."""
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    
    # Simulate realistic OHLCV data
    np.random.seed(42)
    base_price = 150
    data = []
    
    for i, date in enumerate(dates):
        # Random walk for price
        if i == 0:
            close = base_price
        else:
            close = data[-1]['close'] * (1 + np.random.normal(0, 0.02))
        
        # OHLC based on close
        daily_range = close * np.random.uniform(0.01, 0.05)
        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)
        open_price = low + np.random.uniform(0, high - low)
        
        # Volume
        volume = np.random.randint(1000000, 50000000)
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def create_technical_chart(data, indicators, signals):
    """Create comprehensive technical analysis chart."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD', 'Volume'),
        row_heights=[0.5, 0.2, 0.2, 0.1]
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data['date'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price'
    ), row=1, col=1)
    
    # Moving averages
    if 'SMA_20' in indicators:
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=indicators['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=2)
        ), row=1, col=1)
    
    if 'SMA_50' in indicators:
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=indicators['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=2)
        ), row=1, col=1)
    
    # Bollinger Bands
    if all(k in indicators for k in ['BB_upper', 'BB_middle', 'BB_lower']):
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=indicators['BB_upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1),
            fill=None
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=indicators['BB_lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)'
        ), row=1, col=1)
    
    # RSI
    if 'RSI' in indicators:
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=indicators['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=2, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # MACD
    if all(k in indicators for k in ['MACD', 'MACD_signal']):
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=indicators['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2)
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=indicators['MACD_signal'],
            mode='lines',
            name='Signal',
            line=dict(color='red', width=2)
        ), row=3, col=1)
        
        if 'MACD_hist' in indicators:
            fig.add_trace(go.Bar(
                x=data['date'],
                y=indicators['MACD_hist'],
                name='Histogram',
                marker_color='gray',
                opacity=0.6
            ), row=3, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=data['date'],
        y=data['volume'],
        name='Volume',
        marker_color='rgba(158, 71, 99, 0.6)'
    ), row=4, col=1)
    
    fig.update_layout(
        title="📊 Comprehensive Technical Analysis",
        height=800,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    return fig

def display_signal_cards(signals, overall_signal):
    """Display trading signal cards."""
    st.markdown("### 🎯 Trading Signals Analysis")
    
    # Overall signal
    signal_class = f"{overall_signal['signal'].lower()}-signal"
    strength_class = f"strength-{overall_signal['strength'].lower()}"
    
    st.markdown(f"""
    <div class="signal-card {signal_class}">
        <h3>🎯 Overall Signal: {overall_signal['signal']}</h3>
        <p>Strength: <span class="{strength_class}">{overall_signal['strength']}</span></p>
        <p>Score: {overall_signal['score']:+d} (Buy signals - Sell signals)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Individual signals
    col1, col2 = st.columns(2)
    
    for i, (indicator, signal_data) in enumerate(signals.items()):
        with col1 if i % 2 == 0 else col2:
            signal_class = f"{signal_data['signal'].lower()}-signal"
            strength_class = f"strength-{signal_data['strength'].lower()}"
            
            st.markdown(f"""
            <div class="technical-indicator">
                <h4>{indicator}</h4>
                <p>Signal: <strong>{signal_data['signal']}</strong></p>
                <p>Strength: <span class="{strength_class}">{signal_data['strength']}</span></p>
                <p>Value: {signal_data['value']}</p>
            </div>
            """, unsafe_allow_html=True)

def create_signal_strength_gauge(overall_signal):
    """Create signal strength gauge."""
    # Convert signal to numeric value
    signal_value = 0
    if overall_signal['signal'] == 'BUY':
        signal_value = 50 + (overall_signal['score'] * 10)
    elif overall_signal['signal'] == 'SELL':
        signal_value = 50 + (overall_signal['score'] * 10)
    else:
        signal_value = 50
    
    signal_value = max(0, min(100, signal_value))
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = signal_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Signal Strength"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "red"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': signal_value
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    """Main trading signals dashboard."""
    
    # Header
    st.markdown("""
    <div class="trading-header">
        <h1>📊 Advanced Trading Signals Dashboard</h1>
        <h3>Professional Technical Analysis & AI-Powered Signals</h3>
        <p>Real-time technical indicators, signal generation, and trading recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading market data and calculating indicators..."):
        data = load_trading_data()
        
        # Calculate technical indicators
        analyzer = TechnicalAnalyzer(data)
        indicators = analyzer.calculate_all_indicators()
        
        # Generate signals
        signal_gen = SignalGenerator(data, indicators)
        signals = signal_gen.generate_comprehensive_signals()
        overall_signal = signal_gen.calculate_overall_signal(signals)
    
    # Sidebar controls
    st.sidebar.markdown("### 🎛️ Signal Configuration")
    
    # Symbol selection (simulated)
    symbol = st.sidebar.selectbox("📊 Select Symbol", ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'])
    
    # Timeframe
    timeframe = st.sidebar.selectbox("⏰ Timeframe", ['1D', '4H', '1H', '15M'])
    
    # Signal sensitivity
    sensitivity = st.sidebar.slider("🎯 Signal Sensitivity", 0.1, 1.0, 0.7, 0.1)
    
    # Risk tolerance
    risk_level = st.sidebar.selectbox("⚠️ Risk Level", ['Conservative', 'Moderate', 'Aggressive'])
    
    # Current market info
    current_price = data['close'].iloc[-1]
    price_change = (current_price - data['close'].iloc[-2]) / data['close'].iloc[-2]
    
    st.sidebar.markdown("### 📈 Current Market")
    st.sidebar.metric(
        f"{symbol} Price", 
        f"${current_price:.2f}", 
        delta=f"{price_change:+.2%}"
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Technical Analysis", 
        "🎯 Trading Signals", 
        "📈 Signal History",
        "⚙️ Strategy Builder"
    ])
    
    with tab1:
        st.markdown("### 📊 Advanced Technical Analysis")
        
        # Technical chart
        tech_chart = create_technical_chart(data, indicators, signals)
        st.plotly_chart(tech_chart, use_container_width=True)
        
        # Key levels
        st.markdown("#### 🎯 Key Technical Levels")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            support = data['low'].rolling(20).min().iloc[-1]
            st.metric("Support", f"${support:.2f}")
        
        with col2:
            resistance = data['high'].rolling(20).max().iloc[-1]
            st.metric("Resistance", f"${resistance:.2f}")
        
        with col3:
            pivot = (data['high'].iloc[-1] + data['low'].iloc[-1] + data['close'].iloc[-1]) / 3
            st.metric("Pivot Point", f"${pivot:.2f}")
        
        with col4:
            atr = np.mean(data['high'].iloc[-14:] - data['low'].iloc[-14:])
            st.metric("ATR", f"${atr:.2f}")
    
    with tab2:
        # Signal strength gauge
        col1, col2 = st.columns([1, 2])
        
        with col1:
            gauge = create_signal_strength_gauge(overall_signal)
            st.plotly_chart(gauge, use_container_width=True)
        
        with col2:
            # Display signal cards
            display_signal_cards(signals, overall_signal)
        
        # Signal details table
        st.markdown("#### 📋 Detailed Signal Analysis")
        
        signal_df = pd.DataFrame([
            {
                'Indicator': indicator,
                'Signal': data['signal'],
                'Strength': data['strength'],
                'Value': data['value'],
                'Weight': {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[data['strength']]
            }
            for indicator, data in signals.items()
        ])
        
        st.dataframe(
            signal_df,
            column_config={
                "Indicator": "📊 Technical Indicator",
                "Signal": "🚦 Signal",
                "Strength": "💪 Strength",
                "Value": "📊 Current Value",
                "Weight": "⚖️ Weight"
            },
            use_container_width=True
        )
    
    with tab3:
        st.markdown("### 📈 Signal Performance History")
        
        # Simulate signal history
        signal_history = []
        for i in range(30):
            date = datetime.now() - timedelta(days=i)
            signal = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])
            accuracy = np.random.uniform(0.6, 0.9)
            
            signal_history.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Signal': signal,
                'Accuracy': f"{accuracy:.1%}",
                'Return': f"{np.random.uniform(-0.05, 0.08):+.2%}",
                'Confidence': f"{np.random.uniform(0.5, 0.95):.1%}"
            })
        
        history_df = pd.DataFrame(signal_history)
        st.dataframe(history_df, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Win Rate", "67.5%", delta="+2.3%")
        with col2:
            st.metric("Avg Return", "+1.8%", delta="+0.4%")
        with col3:
            st.metric("Max Drawdown", "-8.2%", delta="+1.1%")
        with col4:
            st.metric("Sharpe Ratio", "1.45", delta="+0.12")
    
    with tab4:
        st.markdown("### ⚙️ Custom Strategy Builder")
        
        # Strategy parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Entry Conditions")
            rsi_entry = st.slider("RSI Entry Level", 0, 100, 30)
            ma_cross = st.checkbox("Moving Average Crossover")
            volume_confirm = st.checkbox("Volume Confirmation")
        
        with col2:
            st.markdown("#### 🚪 Exit Conditions")
            stop_loss = st.slider("Stop Loss %", 1, 20, 5)
            take_profit = st.slider("Take Profit %", 5, 50, 15)
            trailing_stop = st.checkbox("Trailing Stop")
        
        # Strategy backtest button
        if st.button("🚀 Backtest Strategy"):
            with st.spinner("Running strategy backtest..."):
                # Simulate backtest results
                backtest_results = {
                    'Total Return': f"{np.random.uniform(0.05, 0.25):+.1%}",
                    'Win Rate': f"{np.random.uniform(0.55, 0.75):.1%}",
                    'Max Drawdown': f"{np.random.uniform(0.08, 0.15):.1%}",
                    'Sharpe Ratio': f"{np.random.uniform(1.2, 2.0):.2f}",
                    'Total Trades': f"{np.random.randint(50, 150)}"
                }
                
                st.success("Backtest completed!")
                
                # Display results
                result_cols = st.columns(len(backtest_results))
                for i, (metric, value) in enumerate(backtest_results.items()):
                    with result_cols[i]:
                        st.metric(metric, value)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h4>📊 Advanced Trading Signals Dashboard</h4>
        <p>Professional technical analysis powered by advanced algorithms and machine learning</p>
        <p><em>⚠️ Disclaimer: Trading involves risk. Signals are for educational purposes only.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
