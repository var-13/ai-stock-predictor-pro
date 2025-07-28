# Contributing to AI Stock Predictor Pro

Thank you for your interest in contributing to AI Stock Predictor Pro! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs
- Include detailed description and steps to reproduce
- Specify your Python version and operating system

### Suggesting Features
- Open an issue with the "enhancement" label
- Describe the feature and its benefits
- Include use cases and examples

### Code Contributions

#### 1. Fork and Clone
```bash
git clone https://github.com/var-13/ai-stock-predictor-pro.git
cd ai-stock-predictor-pro
```

#### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

#### 3. Set Up Development Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 4. Make Changes
- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Include type hints where appropriate
- Add tests for new functionality

#### 5. Test Your Changes
```bash
python -m pytest tests/
python run_project.py  # Ensure the main pipeline works
```

#### 6. Commit and Push
```bash
git add .
git commit -m "feat: add amazing new feature"
git push origin feature/your-feature-name
```

#### 7. Create Pull Request
- Provide clear description of changes
- Reference any related issues
- Include screenshots for UI changes

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use meaningful variable and function names
- Keep functions focused and small
- Add comments for complex logic

### Commit Messages
Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `refactor:` for code refactoring
- `test:` for adding tests

### Project Structure
```
src/
├── data_collection/    # Data gathering modules
├── feature_engineering/  # Feature creation
├── models/            # ML model implementations
├── evaluation/        # Model evaluation
└── utils/            # Helper utilities
```

### Adding New Models
1. Create model class in `src/models/`
2. Implement `train()` and `predict()` methods
3. Add configuration to `config.yaml`
4. Update ensemble in `train_ensemble.py`
5. Add tests and documentation

### Adding New Features
1. Implement in `src/feature_engineering/`
2. Add to feature list in config
3. Update documentation
4. Test with existing models

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Test Structure
- Unit tests for individual functions
- Integration tests for complete workflows
- Backtesting for trading strategies

## 📚 Documentation

### Docstring Format
```python
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Series of stock prices
        period: Number of periods for RSI calculation
        
    Returns:
        Series of RSI values (0-100)
        
    Example:
        >>> rsi = calculate_rsi(stock_data['close'])
        >>> print(rsi.head())
    """
```

### README Updates
- Update README.md for significant changes
- Include examples for new features
- Update performance metrics if improved

## 🚀 Release Process

### Version Numbering
Follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Creating Releases
1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create GitHub release with notes
4. Tag the release: `git tag v1.2.3`

## 💡 Ideas for Contributions

### High Priority
- [ ] Add cryptocurrency prediction support
- [ ] Implement real-time data streaming
- [ ] Add more technical indicators
- [ ] Improve LSTM architecture
- [ ] Add options pricing models

### Medium Priority
- [ ] Create mobile-responsive dashboard
- [ ] Add more news sources
- [ ] Implement portfolio rebalancing
- [ ] Add sector analysis
- [ ] Create API endpoints

### Low Priority
- [ ] Add internationalization
- [ ] Create Docker containers
- [ ] Add social media sentiment
- [ ] Implement reinforcement learning
- [ ] Add fundamental analysis

## 📞 Contact

- **Project Maintainer**: var-13
- **GitHub Issues**: Use for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions

## 📜 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the project
- Show empathy towards other contributors

Thank you for contributing to AI Stock Predictor Pro! 🚀
