# Stock Price Forecasting

## Project Overview

This project tackles the challenge of stock price prediction through two-stage approach that addresses the fundamental problem of feature availability in forecasting scenarios. Rather than relying solely on historical data, the system forecasts future feature values and uses these predictions to generate price predictions.

The system is designed to work with any publicly traded stock by automatically collecting relevant data and generating predictions using the same methodology across different securities.

## How It Works

### Stage 1: Feature Forecasting System

The system first predicts future values of features using LightGBM models:

- **Technical Indicators**: Moving averages (SMA, EMA), momentum indicators (RSI, MACD), volatility measures (ATR, Bollinger Bands)
- **Market Context**: Major indices (S&P 500, NASDAQ, Dow Jones), sector performance, volatility index (VIX)
- **Economic Variables**: Treasury rates, interest rate differentials, economic sentiment indicators

Each feature gets its own optimized LightGBM model trained on lagged versions of all features, creating a forecasting network.

### Stage 2: Price Prediction Ensemble

Multiple machine learning algorithms use the forecasted features to predict stock prices:

**Tree-Based Models (Primary)**:

- **Random Forest**: Ensemble of decision trees with bootstrap aggregating
- **XGBoost**: Gradient boosting with advanced regularization
- **CatBoost**: Categorical boosting with automatic feature selection
- **LightGBM**: Efficient gradient boosting with leaf-wise tree growth
- **Extra Trees**: Extremely randomized trees with random feature splitting for enhanced diversity

**Linear Models (Benchmarks)**:

- **Ridge Regression**: L2 regularization for multicollinearity handling
- **Lasso Regression**: L1 regularization with automatic feature selection
- **Huber Regression**: Robust regression for outlier resistance
- **Linear Regression**: Standard regression as a benchmark for linear models

## Technical Methodology

### Data Engineering Pipeline

1. **Data Collection**: Automated retrieval from Yahoo Finance (stock data) and FRED API (economic indicators)
2. **Feature Engineering**: Creation of engineered features including technical indicators, economic derivatives, and market context variables
3. **Data Preprocessing**: Forward-filling of economic indicators, handling missing values, and temporal alignment

### Feature Forecasting Architecture

- **Lag Structure**: Uses 1, 2, and 3-period lags to capture short and medium-term dependencies
- **Model Training**: Individual LightGBM models for each predictable feature with hyperparameter optimization
- **Recursive Prediction**: Multi-step ahead forecasting using previously predicted values
- **Validation Sets**: Historical validation datasets created with forecasted features to evaluate prediction accuracy against actual prices
- **Feature Categories**:

  - Predictable features (daily-changing): Technical indicators, market indices, treasury rates
  - Forward-filled features (monthly/structural): CPI, PPI, unemployment rate, economic sentiment

### Model Training & Selection

- **Hyperparameter Optimization**: RandomizedSearchCV with time-series aware splits
- **Performance Metrics**: MSE, MAE, R², with focus on out-of-sample validation performance
- **Model Selection**: Best model chosen based on validation set performance

### Handling Data Challenges

The project addresses common financial data issues through algorithmic solutions:

- **Non-Stationary Prices**: Tree-based models naturally handle trends and regime changes without requiring stationarity transformations
- **Multicollinearity**: High VIF values (>100,000) managed through tree-based feature selection rather than preprocessing
- **Feature Redundancy**: Algorithms automatically select most informative features at each decision point

## Key Features

### Feature Engineering

- **Technical Analysis**: 15+ technical indicators with optimized parameters
- **Economic Integration**: Real-time economic data integration with proper lag handling
- **Market Context**: Cross-asset correlations and market regime indicators
- **Temporal Features**: Multi-timeframe analysis with various lookback periods

### Robust Forecasting Framework

- **Feature Forecasting**: Predicts future feature values rather than assuming static conditions
- **Validation Strategy**: Time-series aware validation preventing data leakage
- **Performance Monitoring**: Comprehensive metrics tracking and model diagnostics

## Results & Performance

The system generates:

- **Price Predictions**: 5-day ahead stock price forecasts
- **Feature Forecasts**: Predicted values for all engineered features
- **Model Comparisons**: Performance analysis across all algorithms
- **Feature Importance**: Analysis of key prediction drivers
- **Validation Metrics**: Comprehensive performance evaluation on unseen data

## Technical Requirements

- **Python 3.8+** with scientific computing stack
- **Machine Learning**: scikit-learn, XGBoost, CatBoost, LightGBM
- **Data Analysis**: pandas, numpy, scipy, statsmodels
- **Visualization**: matplotlib, seaborn
- **Data Sources**: yfinance, fredapi
- **Development**: Jupyter notebooks for analysis and experimentation

## Usage

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python src/main_pipeline.py --ticker AAPL --stages all

# Or run individual stages
python src/feature_engineering.py --ticker AAPL
python src/model_training.py --ticker AAPL
```

### Notebook Workflow

Execute notebooks sequentially for detailed analysis and experimentation:

1. Data collection and initial processing
2. Feature engineering and validation
3. Exploratory data analysis with stationarity testing
4. Feature forecasting model development
5. Price prediction model training and selection
6. Final predictions and performance analysis

## Model Performance Insights

The project reveal some key limitations:

- Feature forecasting can improve prediction accuracy but may struggle with less stable stocks due to error accumulation from recursive prediction
- Recursive forecasting introduces compounding errors as predicted values are used for subsequent predictions
