"""
Multi-Dashboard Launcher
Centralized access to all dashboard versions with feature comparison
"""

import streamlit as st
import subprocess
import sys
import os
from datetime import datetime

st.set_page_config(
    page_title="🚀 Dashboard Launcher",
    page_icon="🚀",
    layout="wide"
)

# Launcher CSS
st.markdown("""
<style>
    .launcher-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    .dashboard-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #4facfe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    
    .feature-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .pro-badge {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #000;
        font-weight: bold;
    }
    
    .new-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .comparison-table {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_dashboard_info():
    """Get information about available dashboards."""
    dashboards = {
        "simple_app.py": {
            "name": "🤖 AI Stock Predictor Pro",
            "description": "Professional ML dashboard with ensemble predictions",
            "features": ["Ensemble Models", "Technical Analysis", "Clean UI", "Portfolio Metrics"],
            "complexity": "Standard",
            "best_for": "Interviews & Demos",
            "status": "Stable"
        },
        "enhanced_app.py": {
            "name": "🚀 Enhanced Analytics Platform", 
            "description": "Advanced dashboard with animations and real-time features",
            "features": ["Advanced UI", "Risk Analytics", "Portfolio Optimization", "Enhanced Charts", "Dark Theme"],
            "complexity": "Advanced",
            "best_for": "Professional Presentation",
            "status": "New"
        },
        "live_app.py": {
            "name": "🔴 Live Trading Dashboard",
            "description": "Real-time market simulation with live updates",
            "features": ["Real-time Updates", "Live Data Stream", "Market Alerts", "News Feed", "Auto-refresh"],
            "complexity": "Professional",
            "best_for": "Live Demonstrations",
            "status": "Beta"
        },
        "trading_signals.py": {
            "name": "📊 Trading Signals Pro",
            "description": "Advanced technical analysis and signal generation",
            "features": ["Technical Indicators", "Signal Generation", "Strategy Builder", "Backtesting", "Professional UI"],
            "complexity": "Expert",
            "best_for": "Trading Focus",
            "status": "New"
        }
    }
    
    return dashboards

def launch_dashboard(dashboard_file):
    """Launch a specific dashboard."""
    try:
        # Change to dashboard directory
        dashboard_path = os.path.join("dashboard", dashboard_file)
        
        if os.path.exists(dashboard_path):
            st.success(f"🚀 Launching {dashboard_file}...")
            st.info("Opening in new browser tab...")
            
            # Create a command to run the dashboard
            cmd = f"streamlit run {dashboard_path} --server.port 8502"
            st.code(cmd)
            st.info("Copy and run this command in your terminal to launch the dashboard")
            
        else:
            st.error(f"Dashboard file not found: {dashboard_path}")
            
    except Exception as e:
        st.error(f"Error launching dashboard: {e}")

def create_feature_comparison():
    """Create feature comparison table."""
    dashboards = get_dashboard_info()
    
    # Feature comparison data
    features = [
        "Basic ML Predictions",
        "Ensemble Models", 
        "Technical Analysis",
        "Real-time Updates",
        "Advanced UI/UX",
        "Risk Analytics",
        "Portfolio Management",
        "Trading Signals",
        "News Integration",
        "Strategy Builder",
        "Dark Theme",
        "Animations"
    ]
    
    comparison_data = {
        "Feature": features,
        "Simple": ["✅", "✅", "✅", "❌", "✅", "✅", "✅", "❌", "❌", "❌", "❌", "❌"],
        "Enhanced": ["✅", "✅", "✅", "❌", "✅", "✅", "✅", "✅", "✅", "❌", "✅", "✅"],
        "Live": ["✅", "✅", "✅", "✅", "✅", "✅", "✅", "❌", "✅", "❌", "✅", "✅"],
        "Trading": ["✅", "❌", "✅", "❌", "✅", "✅", "❌", "✅", "❌", "✅", "❌", "❌"]
    }
    
    import pandas as pd
    comparison_df = pd.DataFrame(comparison_data)
    
    st.markdown("### 📊 Feature Comparison Matrix")
    st.dataframe(
        comparison_df,
        column_config={
            "Feature": "🔧 Features",
            "Simple": "🤖 Simple",
            "Enhanced": "🚀 Enhanced", 
            "Live": "🔴 Live",
            "Trading": "📊 Trading"
        },
        use_container_width=True,
        hide_index=True
    )

def main():
    """Main launcher interface."""
    
    # Header
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
    <div class="launcher-header">
        <h1>🚀 Multi-Dashboard Launcher</h1>
        <h3>Professional ML Trading Platform Suite</h3>
        <p>Choose the perfect dashboard for your needs - from simple demos to advanced trading systems</p>
        <p><small>Last updated: {current_time}</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    dashboards = get_dashboard_info()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Dashboards", len(dashboards))
    with col2:
        st.metric("🆕 New Features", "12+")
    with col3:
        st.metric("🎯 Use Cases", "4")
    with col4:
        st.metric("⭐ Complexity Levels", "4")
    
    # Dashboard selection
    st.markdown("## 🎛️ Choose Your Dashboard")
    
    # Create dashboard cards
    for i, (filename, info) in enumerate(dashboards.items()):
        
        # Determine card styling based on status
        if info["status"] == "New":
            status_badge = '<span class="feature-badge new-badge">🆕 NEW</span>'
        elif info["status"] == "Beta":
            status_badge = '<span class="feature-badge">🧪 BETA</span>'
        else:
            status_badge = '<span class="feature-badge">✅ STABLE</span>'
        
        if info["complexity"] == "Expert":
            complexity_badge = '<span class="feature-badge pro-badge">💎 PRO</span>'
        else:
            complexity_badge = f'<span class="feature-badge">{info["complexity"]}</span>'
        
        # Create columns for layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div class="dashboard-card">
                <h3>{info['name']} {status_badge} {complexity_badge}</h3>
                <p>{info['description']}</p>
                <p><strong>Best for:</strong> {info['best_for']}</p>
                <div>
                    {''.join([f'<span class="feature-badge">{feature}</span>' for feature in info['features']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)  # Spacing
            if st.button(f"🚀 Launch", key=f"launch_{i}"):
                launch_dashboard(filename)
            
            if st.button(f"📋 View Code", key=f"code_{i}"):
                with st.expander(f"Code Preview - {filename}"):
                    try:
                        with open(f"dashboard/{filename}", 'r') as f:
                            code_preview = f.read()[:1000] + "..." if len(f.read()) > 1000 else f.read()
                        st.code(code_preview, language='python')
                    except:
                        st.error("Code preview not available")
    
    # Feature comparison
    st.markdown("---")
    create_feature_comparison()
    
    # Usage recommendations
    st.markdown("---")
    st.markdown("## 🎯 Usage Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 👨‍💼 For Job Interviews
        **Recommended:** 🤖 **Simple App** or 🚀 **Enhanced App**
        
        - Clean, professional interface
        - Demonstrates core ML concepts
        - Easy to explain and navigate
        - Stable and reliable
        - Good balance of features
        """)
        
        st.markdown("""
        ### 🎓 For Learning & Development
        **Recommended:** 📊 **Trading Signals** or 🚀 **Enhanced App**
        
        - Advanced technical concepts
        - Multiple analysis techniques
        - Strategy building features
        - Educational value
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 For Live Demonstrations
        **Recommended:** 🔴 **Live App** or 🚀 **Enhanced App**
        
        - Real-time updates
        - Interactive features
        - Eye-catching animations
        - Engaging for audience
        """)
        
        st.markdown("""
        ### 💼 For Client Presentations
        **Recommended:** 🚀 **Enhanced App** or 📊 **Trading Signals**
        
        - Professional appearance
        - Comprehensive analytics
        - Risk management features
        - Business-focused metrics
        """)
    
    # Technical requirements
    st.markdown("---")
    st.markdown("## ⚙️ Technical Requirements")
    
    requirements_info = {
        "🤖 Simple App": {
            "Python": "3.8+",
            "RAM": "2GB", 
            "Dependencies": "Basic (Streamlit, Plotly)",
            "Load Time": "< 5 seconds"
        },
        "🚀 Enhanced App": {
            "Python": "3.8+",
            "RAM": "4GB",
            "Dependencies": "Standard (+ Advanced UI)",
            "Load Time": "< 10 seconds"
        },
        "🔴 Live App": {
            "Python": "3.8+", 
            "RAM": "4GB",
            "Dependencies": "Standard (+ Real-time)",
            "Load Time": "< 8 seconds"
        },
        "📊 Trading Signals": {
            "Python": "3.8+",
            "RAM": "6GB",
            "Dependencies": "Full (+ TA-Lib)",
            "Load Time": "< 15 seconds"
        }
    }
    
    req_df = pd.DataFrame(requirements_info).T
    st.dataframe(req_df, use_container_width=True)
    
    # Footer with quick actions
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Launcher"):
            st.rerun()
    
    with col2:
        if st.button("📚 View Documentation"):
            st.info("Documentation available in README.md and QUICKSTART.md")
    
    with col3:
        if st.button("💡 Get Recommendations"):
            st.balloons()
            st.success("For beginners: Start with Simple App, then try Enhanced App!")

if __name__ == "__main__":
    main()
