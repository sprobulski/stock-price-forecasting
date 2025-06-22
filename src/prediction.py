# FEATURE-BASED STOCK PRICE FORECASTING
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

def load_best_model(ticker):
    """Load the best trained model
    
    Args:
        ticker: Ticker symbol to load the best model for.
        
    Returns:
        tuple: (model, feature_scaler, target_scaler, metadata)
    """

    # Load metadata
    with open(f'../models/{ticker}/best_model_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"Loading best model: {metadata['model_name']}")
    print(f"Model type: {metadata.get('model_type', 'Unknown')}")
    print(f"Val MSE: {metadata['validation_mse']:.4f}")

    # Load sklearn model
    with open(f'../models/{ticker}/best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    print(f"{metadata['model_name']} loaded successfully")

    # Load scalers
    feature_scaler = None

    try:
        if metadata.get('uses_feature_scaling', False):
            with open(f'../models/{ticker}/best_model_scaler.pkl', 'rb') as f:
                feature_scaler = pickle.load(f)
            print("Feature scaler loaded")

    except FileNotFoundError as e:
        print(f"Scaler file not found: {e}")

    return model, feature_scaler, metadata

def load_data(ticker: str, metadata: dict) -> tuple:
    """Load data for prediction
    
    Args:
        ticker: Ticker symbol to load the data for.
        metadata: Metadata for the model.
        
    Returns:
        tuple: (current_price, current_date, future_features_df, feature_columns)
    """

    try:
        # Load current data
        df = pd.read_csv(f'../data/final/{ticker}/preprocessed_features.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        current_price = df['Close'].iloc[-1]
        current_date = df['Date'].iloc[-1]

        # Load future features
        future_features_df = pd.read_csv(f'../data/predicted_features/{ticker}/future_prediction_features.csv')
        feature_columns = metadata['feature_columns']

        print("Data loaded successfully")
        print(f"Prediction date: {current_date.strftime('%Y-%m-%d')}")
        print(f"   Current {ticker} price: ${current_price:.2f}")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Future features shape: {future_features_df.shape}")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")

    return current_price, current_date, future_features_df, feature_columns

def make_future_predictions(model: object, scaler: object, future_features_df: pd.DataFrame, feature_columns: list) -> dict:
    """Make predictions using forecasted features from future_prediction_features.csv"""

    print("\n" + "="*60)
    print("FUTURE PRICE PREDICTIONS")
    print("="*60)

    predictions = {}

    print("Making predictions with forecasted features...")

    for i, row in future_features_df.iterrows():
        day = i + 1

        # Create feature vector
        feature_vector = np.array([[row[col] for col in feature_columns]])

        # Scale features if needed
        if scaler:
            feature_vector_scaled = scaler.transform(feature_vector)
        else:
            feature_vector_scaled = feature_vector

        # Make prediction
        pred_price = model.predict(feature_vector_scaled)[0]
        predictions[day] = pred_price

    return predictions

def display_predictions(future_predictions: dict, current_date: datetime, current_price: float) -> None:
    """Display predictions with dates.
    
    Args:
        future_predictions: Dictionary containing future predictions.
        current_date: Current date.
        current_price: Current price.
    """

    print("\nFINAL PREDICTIONS:")
    for day in [1, 2, 3, 4, 5]:
        if day in future_predictions:
            future_date = current_date + timedelta(days=day)
            while future_date.weekday() >= 5:  # Skip weekends
                future_date += timedelta(days=1)

            change = future_predictions[day] - current_price
            change_pct = (change / current_price) * 100

            print(f"  {future_date.strftime('%Y-%m-%d')}: ${future_predictions[day]:.2f} ({change_pct:+.1f}%)")


        # 5-day outlook
    if 5 in future_predictions:
        trend = "📈 Upward" if future_predictions[5] > current_price else "📉 Downward"
        total_change = ((future_predictions[5] - current_price) / current_price) * 100
        print(f"\n5-DAY OUTLOOK: {trend} trend ({total_change:+.1f}%)")
