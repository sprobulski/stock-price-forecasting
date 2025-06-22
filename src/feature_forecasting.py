"""Feature Forecasting module for stock price forecasting."""
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from permetrics.regression import RegressionMetric
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(ticker: str) -> tuple[pd.DataFrame, list, list, list]:
    """Load data and define feature categories.
    
    Args:
        ticker: Ticker symbol of the company.
        
    Returns:
        tuple: (df, forward_filled_features, predictable_features, all_features)
    """
    # Load the preprocessed data
    df = pd.read_csv(f'../data/final/{ticker}/preprocessed_features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nColumns: {list(df.columns)}")

    # Define feature categories
    # Forward-filled features (don't change daily - economic indicators)
    forward_filled_features = [
        'Fed_Funds_Rate', 'CPI', 'PPI', 'Unemployment_Rate', 
        'Industrial_Production', 'Consumer_Sentiment', 'Housing_Starts',
        'CPI_YoY_Change', 'PPI_YoY_Change', 'Yield_Curve_Spread', 'Fed_10Y_Spread', 'Treasury_10Y_DoD', 'Treasury_3M_DoD'
    ]

    # Predictable features (change daily)
    all_features = [col for col in df.columns if col not in ['Date', 'Close']]
    predictable_features = [col for col in all_features if col not in forward_filled_features]

    print(f"Forward-filled features ({len(forward_filled_features)}): {forward_filled_features}")
    print(f"\nPredictable features ({len(predictable_features)}): {predictable_features}")

    # Verify all features are accounted for
    total_features = len(forward_filled_features) + len(predictable_features)
    print(f"\nTotal features: {len(all_features)}, Categorized: {total_features}")
    assert len(all_features) == total_features, "Some features not categorized!"

    return df, forward_filled_features, predictable_features, all_features


def create_lagged_features(df: pd.DataFrame, features: list, lag_periods: list) -> pd.DataFrame:
    """Create lagged features for LightGBM training.
    
    Args:
        df: Input DataFrame.
        features: List of features to create lags for.
        lag_periods: List of lag periods.
        
    Returns:
        DataFrame: DataFrame with lagged features.
    """
    df_lagged = df[['Date'] + features].copy()

    # Create lagged features
    for feature in features:
        for lag in lag_periods:
            df_lagged[f'{feature}_lag{lag}'] = df_lagged[feature].shift(lag)

    # Drop rows with NaN values (due to lagging)
    max_lag = max(lag_periods)
    df_lagged = df_lagged.iloc[max_lag:].reset_index(drop=True)

    return df_lagged


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Comprehensive evaluation using Permetrics library.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    evaluator = RegressionMetric(y_true, y_pred)

    return {
        'rmse': evaluator.RMSE(),
        'mae': evaluator.MAE(),
        'mape': evaluator.MAPE(),           # Regular MAPE
        'smape': evaluator.SMAPE(),        # Symmetric MAPE
        'r2': evaluator.R2()
    }


def classify_performance(smape_score: float) -> tuple[str, str]:
    """Classify model performance based on sMAPE score.
    
    Args:
        smape_score: sMAPE score.
        
    Returns:
        tuple: (Performance classification, emoji)
    """
    if smape_score < 0.01:
        return "EXCELLENT", "🟢"
    elif smape_score < 0.05:
        return "VERY GOOD", "🟢"
    elif smape_score < 0.1:
        return "GOOD", "🟡"
    elif smape_score < 0.2:
        return "ACCEPTABLE", "🟡"
    elif smape_score < 0.3:
        return "POOR", "🟠"
    else:
        return "VERY POOR", "🔴"


def train_feature_models(df_lagged: pd.DataFrame, predictable_features: list,
                        lag_periods: list, n_iter: int = 50, cv_folds: int = 3,
                        param_grid: dict = None) -> tuple[dict, dict, dict, list]:
    """Train LightGBM models for each predictable feature with enhanced sMAPE evaluation.
    
    Args:
        df_lagged: DataFrame with lagged features.
        predictable_features: List of predictable features.
        lag_periods: List of lag periods used.
        n_iter: Number of iterations for RandomizedSearchCV.
        cv_folds: Number of cross-validation folds.
        param_grid: Dictionary of hyperparameters to search.
        
    Returns:
        tuple: (models, scalers, performance, lagged_feature_names)
    """
    models = {}
    scalers = {}
    performance = {}

    # Get lagged feature names
    lagged_feature_names = []
    for feature in predictable_features:
        for lag in lag_periods:
            lagged_feature_names.append(f'{feature}_lag{lag}')

    # Normal train/test split (80/20)
    split_idx = int(0.8 * len(df_lagged))
    train_data = df_lagged.iloc[:split_idx].copy()
    test_data = df_lagged.iloc[split_idx:].copy()

    print(f"Training on {len(train_data)} samples")
    print(f"Testing on {len(test_data)} samples")
    print(f"Using {len(lagged_feature_names)} lagged features")

    # Train model for each predictable feature
    for i, target_feature in enumerate(predictable_features):
        print(f"\nTraining model for {target_feature} ({i+1}/{len(predictable_features)})...")

        # Prepare training data
        X_train = train_data[lagged_feature_names].values
        y_train = train_data[target_feature].values
        X_test = test_data[lagged_feature_names].values
        y_test = test_data[target_feature].values

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train LightGBM model with hyperparameter tuning
        model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        model = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=TimeSeriesSplit(n_splits=cv_folds), 
            random_state=42, n_jobs=5, scoring='neg_mean_squared_error'
        )

        model.fit(X_train_scaled, y_train)

        # Calculate performance using Permetrics
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        train_metrics = evaluate_predictions(y_train, train_pred)
        test_metrics = evaluate_predictions(y_test, test_pred)

        # Store model and results
        models[target_feature] = model
        scalers[target_feature] = scaler
        performance[target_feature] = {
            'train_rmse': train_metrics['rmse'],
            'test_rmse': test_metrics['rmse'],
            'train_mae': train_metrics['mae'],
            'test_mae': test_metrics['mae'],
            'train_mape': train_metrics['mape'],
            'test_mape': test_metrics['mape'],
            'train_smape': train_metrics['smape'],  # NEW: sMAPE metrics
            'test_smape': test_metrics['smape'],
            'train_r2': train_metrics['r2'],
            'test_r2': test_metrics['r2']
        }

        # Performance classification based on sMAPE
        perf_class, emoji = classify_performance(test_metrics['smape'])

        print(f"  Train RMSE: {train_metrics['rmse']:.6f}, Test RMSE: {test_metrics['rmse']:.6f}")
        print(f"  Train sMAPE: {train_metrics['smape']:.2f}%, Test sMAPE: {test_metrics['smape']:.2f}% {emoji}")
        print(f"  Performance: {perf_class}")

    return models, scalers, performance, lagged_feature_names


def predict_features_recursively(df_lagged: pd.DataFrame, models: dict, scalers: dict,
                                predictable_features: list, lag_periods: list,
                                forward_filled_features: list, start_idx: int,
                                horizon: int = 5) -> pd.DataFrame:
    """Predict features recursively for the specified horizon.
    
    Args:
        df_lagged: DataFrame with lagged features.
        models: Dictionary of trained models.
        scalers: Dictionary of scalers.
        predictable_features: List of predictable features.
        forward_filled_features: List of forward-filled features.
        start_idx: Starting index for prediction.
        horizon: Number of days to predict.
        
    Returns:
        DataFrame: Predictions for the horizon.
    """
    # Initialize prediction dataframe
    predictions = []

    # Get the starting point data
    current_data = df_lagged.iloc[start_idx:start_idx+max(lag_periods)].copy()

    for day in range(horizon):
        print(f"Predicting day {day+1}/{horizon}...")

        # Create feature vector from current lagged data
        feature_vector = []

        for feature in predictable_features:
            for lag in lag_periods:
                # Get the value from lag periods ago
                lag_idx = len(current_data) - lag
                if lag_idx >= 0:
                    feature_vector.append(current_data.iloc[lag_idx][feature])
                else:
                    # If we don't have enough history, use the earliest available
                    feature_vector.append(current_data.iloc[0][feature])

        feature_vector = np.array(feature_vector).reshape(1, -1)

        # Predict each feature
        predicted_features = {}

        for feature in predictable_features:
            # Scale features
            X_scaled = scalers[feature].transform(feature_vector)

            # Predict
            pred = models[feature].predict(X_scaled)[0]
            predicted_features[feature] = pred

        # Add forward-filled features (use last known values)
        for feature in forward_filled_features:
            if feature in current_data.columns:
                predicted_features[feature] = current_data.iloc[-1][feature]

        # Add date
        if 'Date' in current_data.columns:
            last_date = pd.to_datetime(current_data.iloc[-1]['Date'])
            predicted_features['Date'] = last_date + pd.Timedelta(days=1)

        predictions.append(predicted_features)

        # Update current_data with the new prediction for next iteration
        new_row = predicted_features.copy()
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)

        # Keep only the last max(lag_periods) + 1 rows
        if len(current_data) > max(lag_periods) + 1:
            current_data = current_data.tail(max(lag_periods) + 1).reset_index(drop=True)

    return pd.DataFrame(predictions)


def create_historical_validation_data(df: pd.DataFrame, df_lagged: pd.DataFrame, 
                                    models: dict, scalers: dict,
                                    predictable_features: list, forward_filled_features: list,
                                    lag_periods: list) -> pd.DataFrame:
    """Create historical validation data with predicted features.
    
    Args:
        df: Original DataFrame.
        df_lagged: DataFrame with lagged features.
        models: Dictionary of trained models.
        scalers: Dictionary of scalers.
        lagged_feature_names: List of lagged feature names.
        predictable_features: List of predictable features.
        forward_filled_features: List of forward-filled features.
        lag_periods: List of lag periods.
        
    Returns:
        DataFrame: Combined historical validation data.
    """
    # Calculate test set boundaries for multiple validation periods
    test_start_idx = int(0.8 * len(df_lagged))  # Where test set starts
    test_end_idx = len(df_lagged)  # Where test set ends
    test_size = test_end_idx - test_start_idx

    print(f"Test set: indices {test_start_idx} to {test_end_idx} (size: {test_size})")

    # Generate predictions for multiple historical periods
    print("\n" + "="*60)
    print("GENERATING MULTIPLE VALIDATION DATASETS")
    print("="*60)

    validation_periods = []

    # 1. LAST 5 DAYS (original validation period)
    print("\n1. LAST 5 DAYS VALIDATION:")
    historical_start_idx_last5 = len(df_lagged) - 5 - max(lag_periods)  # Start before the last 5 days
    print(f"   Historical start index: {historical_start_idx_last5} (out of {len(df_lagged)} total)")
    print("   Predicting features for last 5 historical days")

    historical_predictions_last5 = predict_features_recursively(
        df_lagged, models, scalers,
        predictable_features, lag_periods, forward_filled_features,
        historical_start_idx_last5, horizon=5
    )

    # Add actual Close prices and dates from the last 5 days
    actual_last_5 = df.iloc[-5:][['Date', 'Close']].copy().reset_index(drop=True)
    historical_predictions_last5['Date'] = actual_last_5['Date']
    historical_predictions_last5['Close'] = actual_last_5['Close']
    historical_predictions_last5['Period'] = 'Last_5_Days'

    print(f"   Shape: {historical_predictions_last5.shape}")
    print(f"   Date range: {historical_predictions_last5['Date'].min()} to {historical_predictions_last5['Date'].max()}")

    validation_periods.append(("Last_5_Days", historical_predictions_last5))

    # 2. FIRST 5 DAYS OF TEST SET
    print("\n2. FIRST 5 DAYS OF TEST SET:")
    first_test_start_idx = test_start_idx - max(lag_periods)  # Start before first 5 test days
    first_test_data_start = test_start_idx + max(lag_periods)  # Actual data start in original df
    print(f"   Start index: {first_test_start_idx} (predicting from test index {test_start_idx})")

    historical_predictions_first5 = predict_features_recursively(
        df_lagged, models, scalers,
        predictable_features, lag_periods, forward_filled_features,
        first_test_start_idx, horizon=5
    )

    # Add actual Close prices and dates from the first 5 test days
    actual_first_5_test = df.iloc[first_test_data_start:first_test_data_start+5][['Date', 'Close']].copy().reset_index(drop=True)
    historical_predictions_first5['Date'] = actual_first_5_test['Date']
    historical_predictions_first5['Close'] = actual_first_5_test['Close']
    historical_predictions_first5['Period'] = 'First_5_Test_Days'

    print(f"   Shape: {historical_predictions_first5.shape}")
    print(f"   Date range: {historical_predictions_first5['Date'].min()} to {historical_predictions_first5['Date'].max()}")

    validation_periods.append(("First_5_Test_Days", historical_predictions_first5))

    # 3. Add more periods from test set
    # Generate validation every 15 days through test set
    for offset in range(15, test_size-5, 15):
        period_name = f'Test_Days_{offset}-{offset+4}'

        pred_start = test_start_idx + offset - max(lag_periods)
        predictions = predict_features_recursively(
            df_lagged, models, scalers,
            predictable_features, lag_periods, forward_filled_features,
            pred_start
        )

        # Add actual dates and close prices
        actual_data_start = test_start_idx + offset + max(lag_periods)
        if actual_data_start + 5 <= len(df):
            actual_data = df.iloc[actual_data_start:actual_data_start+5][['Date', 'Close']].copy().reset_index(drop=True)
            predictions['Date'] = actual_data['Date']
            predictions['Close'] = actual_data['Close']
            predictions['Period'] = period_name

            print(f"Generated {period_name}: {predictions.shape}")
            validation_periods.append((period_name, predictions))

    # Combine all periods
    all_validation_data = pd.concat([data for _, data in validation_periods], ignore_index=True)
    print(f"\nTOTAL VALIDATION SAMPLES: {len(all_validation_data)}")

    print("="*60)
    print("ADDING FORWARD-FILLED FEATURES TO VALIDATION DATASETS")

    # Extract forward-filled features from original df
    df_forward_filled = df[['Date'] + forward_filled_features].copy()
    print(f"Forward-filled features to add: {forward_filled_features}")

    print("\nAdding to Validation dataset...")
    all_validation_data = pd.merge(
        all_validation_data, 
        df_forward_filled, 
        on='Date', 
        how='left'
    )
    print(f"   Updated shape: {all_validation_data.shape}")

    print("\nFINAL FEATURE COUNT:")
    print(f"   • Total features per dataset: {len(all_validation_data.columns) - 3}")  # Minus Date, Close, Period
    print(f"   • Predictable features: {len(predictable_features)}")
    print(f"   • Forward-filled features: {len(forward_filled_features)}")
    print(f"   • Expected total: {len(predictable_features) + len(forward_filled_features) + 3}")  # +3 for Date, Close, Period

    return all_validation_data


def create_future_prediction_data(df: pd.DataFrame, df_lagged: pd.DataFrame,
                                models: dict, scalers: dict,
                                predictable_features: list, forward_filled_features: list,
                                lag_periods: list, prediction_horizon: int = 5) -> pd.DataFrame:
    """Create future prediction data with predicted features.
    
    Args:
        df: Original DataFrame.
        df_lagged: DataFrame with lagged features.
        models: Dictionary of trained models.
        scalers: Dictionary of scalers.
        lagged_feature_names: List of lagged feature names.
        predictable_features: List of predictable features.
        forward_filled_features: List of forward-filled features.
        lag_periods: List of lag periods.
        prediction_horizon: Number of days to predict.
        
    Returns:
        DataFrame: Future prediction data.
    """
    print("Generating future prediction database...")
    future_start_idx = len(df_lagged) - max(lag_periods)  # Start before the end
    print(f"Future start index: {future_start_idx} (out of {len(df_lagged)} total)")
    print(f"Predicting features for next {prediction_horizon} days beyond the dataset")

    future_predictions = predict_features_recursively(
        df_lagged, models, scalers,
        predictable_features, lag_periods, forward_filled_features,
        future_start_idx
    )

    print("\nAdding forward-filled features from last available day...")
    # Get forward-filled features from the last day
    last_day_forward_filled = df[forward_filled_features].iloc[-1]

    for feature in forward_filled_features:
        future_predictions[feature] = last_day_forward_filled[feature]

    print(f"Forward-filled features added: {forward_filled_features}")
    print(f"Updated shape: {future_predictions.shape}")

    # Add placeholder Close price column (to be predicted by price models)
    future_predictions['Close'] = np.nan

    print(f"\nFuture prediction database shape: {future_predictions.shape}")
    print("\nSample of future predictions:")
    print(future_predictions.head())

    return future_predictions


def diagnostic_analysis(validation_data: pd.DataFrame, df: pd.DataFrame, predictable_features: list) -> None:
    """Perform diagnostic analysis on model predictions.
    
    Args:
        validation_data: Validation data with actual and predicted values.
        df: Original data with actual values.
        predictable_features: List of predictable features.
    """

    comparison_data = pd.merge(
        validation_data, 
        df[['Date'] + predictable_features], 
        on='Date', 
        how='inner',
        suffixes=('_predicted', '_actual')
    )

    print("COMPARISON DATASET:")
    print(f"   • Total comparison samples: {len(comparison_data)}")
    print(f"   • Validation periods: {comparison_data['Period'].nunique()}")
    print(f"   • Date range: {comparison_data['Date'].min()} to {comparison_data['Date'].max()}")
    print(f"   • Features to compare: {len(predictable_features)}")



    print("DIAGNOSTIC ANALYSIS: MODEL PREDICTION ISSUES")
    print("=" * 80)

    # Focus on the main stock price features
    price_features = ['High', 'Low', 'Open', 'MACD', 'RSI', 'MACD_Signal', 'OBV', 'Volume', 'SMA_50', 'SMA_200', 'EMA_50',
                    'Dow_Jones','NASDAQ','S&P_500','VIX','CPI','Unemployment_Rate','Inflation_Rate','GDP','Interest_Rate'
                    'Treasury_10Y','Treasury_3M']
    available_price_features = [f for f in price_features if f in predictable_features]

    print("ANALYZING TEMPORAL PREDICTION PATTERNS:")
    print("-" * 60)

    # Analyze prediction patterns by time period
    for feature in available_price_features:
        pred_col = f"{feature}_predicted"
        actual_col = f"{feature}_actual"
        
        if pred_col in comparison_data.columns and actual_col in comparison_data.columns:
            print(f"\n{feature.upper()} ANALYSIS:")
            
            # Calculate the difference between predicted and actual means
            pred_means = comparison_data.groupby('Period')[pred_col].mean()
            actual_means = comparison_data.groupby('Period')[actual_col].mean()
            differences = actual_means - pred_means
            error_pcts = (differences / actual_means * 100).round(1)
            
            print("Period-wise Analysis:")
            for period in pred_means.index:
                pred_mean = pred_means[period]
                actual_mean = actual_means[period]
                diff = differences[period]
                error_pct = error_pcts[period]
                
                status = "🔴 SEVERE" if abs(error_pct) > 15 else "🟠 HIGH" if abs(error_pct) > 10 else "🟡 MODERATE" if abs(error_pct) > 5 else "🟢 GOOD"
                
                print(f"  {period:25s}: Pred={pred_mean:6.1f} | Actual={actual_mean:6.1f} | "
                    f"Diff={diff:+6.1f} | Error={error_pct:+5.1f}% {status}")