#!/usr/bin/env python3
"""
Quick Start Script for Stock Market Prediction Project
Run this script to set up and execute the complete ML pipeline.
"""

import os
import sys
import subprocess
from datetime import datetime

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def run_command(command, description):
    """Run a system command with error handling."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'yfinance', 'scikit-learn', 
        'xgboost', 'streamlit', 'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("📦 Installing essential packages...")
        # Install essential packages without problematic ones
        essential_cmd = f"pip install {' '.join(missing_packages)}"
        return run_command(essential_cmd, "Installing essential dependencies")
    else:
        print("✅ All required packages are installed!")
        return True

def main():
    """Main execution function."""
    print_header("🚀 Stock Market Prediction ML Project - Quick Start")
    
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Working directory: {os.getcwd()}")
    
    # Step 1: Check requirements
    print_header("📦 Checking Requirements")
    if not check_requirements():
        print("❌ Failed to install requirements. Please install manually.")
        return
    
    # Step 2: Data Collection
    print_header("📊 Data Collection")
    if not run_command("python src/data_collection/collect_data.py", "Collecting stock market data"):
        print("⚠️ Data collection failed. You can still continue with demo data.")
    
    # Step 3: Feature Engineering
    print_header("🔧 Feature Engineering")
    if not run_command("python src/feature_engineering/create_features.py", "Engineering features"):
        print("⚠️ Feature engineering failed. Check the data collection step.")
    
    # Step 4: Model Training
    print_header("🤖 Model Training")
    if not run_command("python src/models/train_ensemble.py", "Training ML models"):
        print("⚠️ Model training failed. Check previous steps.")
    
    # Step 5: Launch Dashboard
    print_header("📈 Launching Dashboard")
    print("🎯 Starting interactive dashboard...")
    print("🌐 Dashboard will open in your browser at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run("streamlit run dashboard/app.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except subprocess.CalledProcessError:
        print("❌ Failed to start dashboard. You can run it manually:")
        print("   streamlit run dashboard/app.py")
    
    # Final Summary
    print_header("🎉 Project Setup Complete!")
    print("✅ Stock market prediction ML project is ready!")
    print("\n📚 What you can do now:")
    print("   1. Explore the Jupyter notebook: notebooks/stock_prediction_demo.ipynb")
    print("   2. Run the dashboard: streamlit run dashboard/app.py")
    print("   3. Customize settings in: config.yaml")
    print("   4. Add more stocks or features to improve predictions")
    print("\n🏆 Perfect for internship portfolios and technical interviews!")
    
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("📧 Check the README.md for troubleshooting tips")
