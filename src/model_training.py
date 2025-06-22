"""Model Training module for stock price forecasting."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, HuberRegressor, Lasso, LinearRegression
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import pickle
import json




def load_data(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and validation data.
    
    Args:
        ticker: Ticker symbol of the company.
        
    Returns:
        tuple: (training_data, validation_data)
    """
    print("Loading data...")
    df = pd.read_csv(f'../data/final/{ticker}/preprocessed_features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    print("Loading validation data...")
    df_val = pd.read_csv(f'../data/predicted_features/{ticker}/historical_validation_features_combined.csv')
    print(f"Validation data shape: {df_val.shape}")

    return df, df_val


def prepare_data_for_modeling(df: pd.DataFrame, df_val: pd.DataFrame) -> tuple:
    """Prepare data for modeling by separating features and target.
    
    Args:
        df: Training DataFrame.
        df_val: Validation DataFrame.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, X_val, y_val, feature_columns, dates)
    """
    print("Preparing data for modeling...")

    # Separate features and target
    feature_columns = [col for col in df.columns if col not in ['Date', 'Close']]
    X = df[feature_columns].values
    y = df['Close'].values
    dates = df['Date'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    X_val = df_val[feature_columns].values
    y_val = df_val['Close'].values

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")

    print("Data preparation complete.")

    return X_train, X_test, y_train, y_test, X_val, y_val, feature_columns, dates


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, 
                       y_test: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple:
    """Train Random Forest model.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        X_val: Validation features.
        y_val: Validation targets.
        
    Returns:
        tuple: (rf_model, rf_train_pred, rf_test_pred, rf_val_pred, metrics)
    """
    print("Training Random Forest...")

    rf_model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    tscv = TimeSeriesSplit(n_splits=5)
    rf_model = RandomizedSearchCV(rf_model, param_grid, n_jobs=5, n_iter=200, 
                                 cv=tscv, scoring='neg_mean_squared_error', verbose=1).fit(X_train, y_train)

    # Random Forest predictions
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    rf_val_pred = rf_model.predict(X_val)

    # Random Forest metrics
    rf_train_mse = mean_squared_error(y_train, rf_train_pred)
    rf_test_mse = mean_squared_error(y_test, rf_test_pred)
    rf_val_mse = mean_squared_error(y_val, rf_val_pred)

    print(f"Random Forest Train MSE: {rf_train_mse:.4f}")
    print(f"Random Forest Test MSE: {rf_test_mse:.4f}")
    print(f"Random Forest Validation MSE: {rf_val_mse:.4f}")

    metrics = {
        'train_mse': rf_train_mse,
        'test_mse': rf_test_mse,
        'val_mse': rf_val_mse
    }

    return rf_model, rf_train_pred, rf_test_pred, rf_val_pred, metrics


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, 
                 y_test: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple:
    """Train XGBoost model.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        X_val: Validation features.
        y_val: Validation targets.
        
    Returns:
        tuple: (xgb_model, xgb_train_pred, xgb_test_pred, xgb_val_pred, metrics)
    """
    print("Training XGBoost...")

    model = xgb.XGBRegressor(random_state=42)
    param_distributions = {
        'learning_rate': [0.005, 0.01, 0.05, 0.1],
        'max_depth': [3, 4],              # Keep trees shallow
        'min_child_weight': [3, 5, 7],       # Force more data per split
        'subsample': [0.6, 0.8],             # Train on row subsets
        'colsample_bytree': [0.6, 0.8],      # Use feature subsets
        'gamma': [0.3, 0.5],                 # Prune low-gain splits
        'reg_alpha': [0.5, 1.0],             # L1 penalty (sparse trees)
        'reg_lambda': [1, 2, 3, 5],            # L2 penalty (smooth weights)
        'n_estimators': [100, 200, 300]      # Fewer trees + early stopping
    }
    tscv = TimeSeriesSplit(n_splits=10)

    # 4. Randomized search
    xgb_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=100,
        scoring='neg_mean_squared_error',
        cv=tscv,
        verbose=1,
        n_jobs=5
    ).fit(X_train, y_train)

    # Predictions
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)
    xgb_val_pred = xgb_model.predict(X_val)

    # Metrics
    xgb_train_mse = mean_squared_error(xgb_train_pred, y_train)
    xgb_test_mse = mean_squared_error(xgb_test_pred, y_test)
    xgb_val_mse = mean_squared_error(xgb_val_pred, y_val)

    print(f"XGBoost Train MSE: {xgb_train_mse:.4f}")
    print(f"XGBoost Test MSE: {xgb_test_mse:.4f}")
    print(f"XGBoost Validation MSE: {xgb_val_mse:.4f}")

    metrics = {
        'train_mse': xgb_train_mse,
        'test_mse': xgb_test_mse,
        'val_mse': xgb_val_mse
    }

    return xgb_model, xgb_train_pred, xgb_test_pred, xgb_val_pred, metrics


def train_catboost(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, 
                  y_test: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple:
    """Train CatBoost model.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        X_val: Validation features.
        y_val: Validation targets.
        
    Returns:
        tuple: (cat_model, cat_train_pred, cat_test_pred, cat_val_pred, metrics)
    """
    print("Training CatBoost...")

    cat_model = cb.CatBoostRegressor(random_state=42)
    param_grid = {
        'iterations': [100, 200, 300],
        'depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'od_type': ['IncToDec', 'Iter'],
        'min_child_samples': [3, 5, 7, 9],
        'leaf_estimation_iterations': [3, 5, 7, 9],
        'leaf_estimation_method': ['Newton', 'Gradient'],
        'colsample_bylevel': [0.5, 0.7, 0.9],
        'bagging_temperature': [0.5, 0.7, 0.9],
        'subsample': [0.5, 0.7, 0.9],
        'reg_lambda': [0.0, 0.1, 0.2],
        'random_strength': [0.0, 0.1, 0.2],
        'silent': [True]
    }

    tscv = TimeSeriesSplit(n_splits=10)
    cat_model = RandomizedSearchCV(cat_model, param_grid, n_jobs=5, n_iter=50, 
                                  cv=tscv, scoring='neg_mean_squared_error', verbose=1).fit(X_train, y_train)

    # CatBoost predictions
    cat_train_pred = cat_model.predict(X_train)
    cat_test_pred = cat_model.predict(X_test)
    cat_val_pred = cat_model.predict(X_val)

    # CatBoost metrics
    cat_train_mse = mean_squared_error(y_train, cat_train_pred)
    cat_test_mse = mean_squared_error(y_test, cat_test_pred)
    cat_val_mse = mean_squared_error(y_val, cat_val_pred)

    print(f"CatBoost Train MSE: {cat_train_mse:.4f}")
    print(f"CatBoost Test MSE: {cat_test_mse:.4f}")
    print(f"CatBoost Validation MSE: {cat_val_mse:.4f}")

    metrics = {
        'train_mse': cat_train_mse,
        'test_mse': cat_test_mse,
        'val_mse': cat_val_mse
    }

    return cat_model, cat_train_pred, cat_test_pred, cat_val_pred, metrics


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, 
                  y_test: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple:
    """Train LightGBM model.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        X_val: Validation features.
        y_val: Validation targets.
        
    Returns:
        tuple: (lgb_model, lgb_train_pred, lgb_test_pred, lgb_val_pred, metrics)
    """
    print("Training LightGBM...")

    lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [2, 3, 4],
        'min_child_samples': [10, 20, 30, 40, 50],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'reg_alpha': [0.1, 0.3, 0.5],
        'reg_lambda': [0.3, 0.5, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=10)
    lgb_model = RandomizedSearchCV(lgb_model, param_grid, n_jobs=5, n_iter=200, 
                                  cv=tscv, scoring='neg_mean_squared_error').fit(X_train, y_train)

    lgb_train_pred = lgb_model.predict(X_train)
    lgb_test_pred = lgb_model.predict(X_test)
    lgb_val_pred = lgb_model.predict(X_val)

    lgb_train_mse = mean_squared_error(y_train, lgb_train_pred)
    lgb_test_mse = mean_squared_error(y_test, lgb_test_pred)
    lgb_val_mse = mean_squared_error(y_val, lgb_val_pred)

    print(f"LightGBM Train MSE: {lgb_train_mse:.4f}")
    print(f"LightGBM Test MSE: {lgb_test_mse:.4f}")
    print(f"LightGBM Validation MSE: {lgb_val_mse:.4f}")

    metrics = {
        'train_mse': lgb_train_mse,
        'test_mse': lgb_test_mse,
        'val_mse': lgb_val_mse
    }

    return lgb_model, lgb_train_pred, lgb_test_pred, lgb_val_pred, metrics


def train_extra_trees(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, 
                     y_test: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple:
    """Train Extra Trees Regressor model.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        X_val: Validation features.
        y_val: Validation targets.
        
    Returns:
        tuple: (et_model, et_train_pred, et_test_pred, et_val_pred, metrics)
    """
    print("Training Extra Trees Regressor...")

    et_model = ExtraTreesRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [2, 3, 4],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    tscv = TimeSeriesSplit(n_splits=10)
    et_model = RandomizedSearchCV(et_model, param_grid, n_jobs=5, n_iter=100, 
                                 cv=tscv, scoring='neg_mean_squared_error', verbose=1).fit(X_train, y_train)

    et_train_pred = et_model.predict(X_train)
    et_test_pred = et_model.predict(X_test)
    et_val_pred = et_model.predict(X_val)

    et_train_mse = mean_squared_error(y_train, et_train_pred)
    et_test_mse = mean_squared_error(y_test, et_test_pred)
    et_val_mse = mean_squared_error(y_val, et_val_pred)

    print(f"Extra Trees Regressor Train MSE: {et_train_mse:.4f}")
    print(f"Extra Trees Regressor Test MSE: {et_test_mse:.4f}")
    print(f"Extra Trees Regressor Validation MSE: {et_val_mse:.4f}")

    metrics = {
        'train_mse': et_train_mse,
        'test_mse': et_test_mse,
        'val_mse': et_val_mse
    }

    return et_model, et_train_pred, et_test_pred, et_val_pred, metrics


def train_linear_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, 
                       y_test: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Train linear models (Ridge, Lasso, Huber, Linear).
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        X_val: Validation features.
        y_val: Validation targets.
        
    Returns:
        dict: Dictionary containing models and their predictions and metrics.
    """
    print("Training Linear Models...")

    # Scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    linear_models = {}

    # Ridge Regression
    print("Training Ridge Regression...")
    ridge_model = Ridge(alpha=1, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)

    ridge_train_pred = ridge_model.predict(X_train_scaled)
    ridge_test_pred = ridge_model.predict(X_test_scaled)
    ridge_val_pred = ridge_model.predict(X_val_scaled)

    ridge_metrics = {
        'train_mse': mean_squared_error(y_train, ridge_train_pred),
        'test_mse': mean_squared_error(y_test, ridge_test_pred),
        'val_mse': mean_squared_error(y_val, ridge_val_pred)
    }

    linear_models['Ridge'] = {
        'model': ridge_model,
        'train_pred': ridge_train_pred,
        'test_pred': ridge_test_pred,
        'val_pred': ridge_val_pred,
        'metrics': ridge_metrics,
        'scaler': scaler
    }

    print(f"Ridge Regression Train MSE: {ridge_metrics['train_mse']:.4f}")
    print(f"Ridge Regression Test MSE: {ridge_metrics['test_mse']:.4f}")
    print(f"Ridge Regression Validation MSE: {ridge_metrics['val_mse']:.4f}")

    # Huber Regressor
    print("Training Huber Regressor...")
    huber_model = HuberRegressor(epsilon=1.1, alpha=0, max_iter=1000)
    huber_model.fit(X_train_scaled, y_train)

    huber_train_pred = huber_model.predict(X_train_scaled)
    huber_test_pred = huber_model.predict(X_test_scaled)
    huber_val_pred = huber_model.predict(X_val_scaled)

    huber_metrics = {
        'train_mse': mean_squared_error(y_train, huber_train_pred),
        'test_mse': mean_squared_error(y_test, huber_test_pred),
        'val_mse': mean_squared_error(y_val, huber_val_pred)
    }

    linear_models['Huber Regressor'] = {
        'model': huber_model,
        'train_pred': huber_train_pred,
        'test_pred': huber_test_pred,
        'val_pred': huber_val_pred,
        'metrics': huber_metrics,
        'scaler': scaler
    }

    print(f"Huber Regressor Train MSE: {huber_metrics['train_mse']:.4f}")
    print(f"Huber Regressor Test MSE: {huber_metrics['test_mse']:.4f}")
    print(f"Huber Regressor Validation MSE: {huber_metrics['val_mse']:.4f}")

    # Lasso Regression
    print("Training Lasso Regression...")
    lasso_model = Lasso(alpha=1, random_state=42)
    lasso_model.fit(X_train_scaled, y_train)

    lasso_train_pred = lasso_model.predict(X_train_scaled)
    lasso_test_pred = lasso_model.predict(X_test_scaled)
    lasso_val_pred = lasso_model.predict(X_val_scaled)

    lasso_metrics = {
        'train_mse': mean_squared_error(y_train, lasso_train_pred),
        'test_mse': mean_squared_error(y_test, lasso_test_pred),
        'val_mse': mean_squared_error(y_val, lasso_val_pred)
    }

    linear_models['Lasso'] = {
        'model': lasso_model,
        'train_pred': lasso_train_pred,
        'test_pred': lasso_test_pred,
        'val_pred': lasso_val_pred,
        'metrics': lasso_metrics,
        'scaler': scaler
    }

    print(f"Lasso Regression Train MSE: {lasso_metrics['train_mse']:.4f}")
    print(f"Lasso Regression Test MSE: {lasso_metrics['test_mse']:.4f}")
    print(f"Lasso Regression Validation MSE: {lasso_metrics['val_mse']:.4f}")

    # Linear Regression
    print("Training Linear Regression...")
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)

    linear_train_pred = linear_model.predict(X_train_scaled)
    linear_test_pred = linear_model.predict(X_test_scaled)
    linear_val_pred = linear_model.predict(X_val_scaled)

    linear_metrics = {
        'train_mse': mean_squared_error(y_train, linear_train_pred),
        'test_mse': mean_squared_error(y_test, linear_test_pred),
        'val_mse': mean_squared_error(y_val, linear_val_pred)
    }

    linear_models['Linear'] = {
        'model': linear_model,
        'train_pred': linear_train_pred,
        'test_pred': linear_test_pred,
        'val_pred': linear_val_pred,
        'metrics': linear_metrics,
        'scaler': scaler
    }

    print(f"Linear Regression Train MSE: {linear_metrics['train_mse']:.4f}")
    print(f"Linear Regression Test MSE: {linear_metrics['test_mse']:.4f}")
    print(f"Linear Regression Validation MSE: {linear_metrics['val_mse']:.4f}")

    return linear_models


def evaluate_all_models(model_results: dict, y_train: np.ndarray, y_test: np.ndarray, y_val: np.ndarray) -> pd.DataFrame:
    """Comprehensive evaluation of all models.
    
    Args:
        model_results: Dictionary containing all model results.
        y_train: Training targets.
        y_test: Test targets.
        y_val: Validation targets.
        
    Returns:
        pd.DataFrame: Results dataframe with all metrics.
    """
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)

    results = []

    for name, results_data in model_results.items():
        if 'metrics' in results_data:
            # Tree models
            train_pred = results_data['train_pred']
            test_pred = results_data['test_pred']
            val_pred = results_data['val_pred']
        else:
            # Linear models
            train_pred = results_data['train_pred']
            test_pred = results_data['test_pred']
            val_pred = results_data['val_pred']

        train_mse = mean_squared_error(y_train, train_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        val_r2 = r2_score(y_val, val_pred)

        results.append({
            'Model': name,
            'Train MSE': train_mse,
            'Test MSE': test_mse,
            'Train MAE': train_mae,
            'Test MAE': test_mae,
            'Val MAE': val_mae,
            'Val MSE': val_mse,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Val R²': val_r2,
            'Overfitting': train_mse / test_mse if test_mse > 0 else float('inf')
        })

        print(f"\n{name}:")
        print(f"  Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f} | Val MSE: {val_mse:.4f}")
        print(f"  Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f} | Val MAE: {val_mae:.4f}")
        print(f"  Train R²:  {train_r2:.4f} | Test R²:  {test_r2:.4f} | Val R²:  {val_r2:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    print("\n\nSUMMARY TABLE:")
    print(results_df.round(4))

    return results_df


def save_best_model(model_results: dict, results_df: pd.DataFrame, feature_columns: list, ticker: str) -> None:
    """Save the best model and its metadata.
    
    Args:
        model_results: Dictionary containing all model results.
        results_df: DataFrame containing comparison results.
        feature_columns: List of feature column names.
        ticker: Ticker symbol of the company.
    """
    print("\n" + "="*60)
    print("SAVING BEST MODEL")
    print("="*60)

    # Create a dictionary with actual model objects
    actual_models = {
        'Random Forest': model_results['Random Forest']['model'],
        'XGBoost': model_results['XGBoost']['model'],
        'CatBoost': model_results['CatBoost']['model'],
        'LightGBM': model_results['LightGBM']['model'],
        'Extra Trees': model_results['Extra Trees']['model'],
        'Ridge': model_results['Ridge']['model'],
        'Huber Regressor': model_results['Huber Regressor']['model'],
        'Lasso': model_results['Lasso']['model'],
        'Linear': model_results['Linear']['model']
    }

    # Create a dictionary with scalers for each model
    model_scalers = {
        'Random Forest': None,              # Tree models don't need scaling
        'XGBoost': None,                   # Tree models don't need scaling
        'CatBoost': None,                  # Tree models don't need scaling
        'LightGBM': None,                  # Tree models don't need scaling
        'Extra Trees': None,               # Tree models don't need scaling
        'Ridge': model_results['Ridge']['scaler'],                   # Linear models need feature scaling
        'Huber Regressor': model_results['Huber Regressor']['scaler'],         # Linear models need feature scaling
        'Lasso': model_results['Lasso']['scaler'],                   # Linear models need feature scaling
        'Linear': model_results['Linear']['scaler']                   # Linear models need feature scaling
    }

    print("Available models for selection:")
    for model_name in results_df['Model'].values:
        mse = results_df[results_df['Model'] == model_name]['Val MSE'].iloc[0]
        print(f"  {model_name}: MSE = {mse:.4f}")

    # Find the best model (single models only)
    best_model_idx = results_df['Val MSE'].idxmin()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    best_val_mse = results_df.loc[best_model_idx, 'Val MSE']

    print(f"\nBest model selected: {best_model_name}")
    print(f"Best Val MSE: {best_val_mse:.4f}")

    best_model_obj = actual_models[best_model_name]
    best_feature_scaler = model_scalers[best_model_name]

    # Save the best model
    os.makedirs(f'../models/{ticker}', exist_ok=True)
    # Save single model
    if hasattr(best_model_obj, 'best_estimator_'):
        model_to_save = best_model_obj.best_estimator_
        print("Extracting best estimator from RandomizedSearchCV")
    else:
        model_to_save = best_model_obj

    with open(f'../models/{ticker}/best_model.pkl', 'wb') as f:
        pickle.dump(model_to_save, f)

    print(f"✅ Model saved: {type(model_to_save).__name__}")

    # Save scalers
    if best_feature_scaler is not None:
        with open(f'../models/{ticker}/best_model_scaler.pkl', 'wb') as f:
            pickle.dump(best_feature_scaler, f)
        print("✅ Feature scaler saved")

    # Save metadata
    model_metadata = {
        'model_name': best_model_name,
        'model_type': type(model_to_save).__name__,
        'validation_mse': float(best_val_mse),
        'feature_columns': feature_columns,
        'training_date': pd.Timestamp.now().isoformat(),
        'uses_feature_scaling': best_feature_scaler is not None,
        'uses_target_scaling': False
    }

    with open(f'../models/{ticker}/best_model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(model_metadata, f, indent=2)

    print("✅ Model metadata saved")
    print("\nReady to run prediction notebook!")


def visualize_results(models: dict, y_val: np.ndarray) -> None:
    """Visualize the results of the models.
    
    Args:
        models: Dictionary containing all model results.
        y_val: Validation targets.
    """
    # Visualization of results - 3x3 grid
    _, axes = plt.subplots(3, 3, figsize=(18, 18))

    # All models to plot (individual + ensemble)
    all_models_to_plot = ['Random Forest', 'XGBoost', 'CatBoost', 'LightGBM', 'Extra Trees', 'Ridge', 'Huber Regressor', 'Lasso', 'Linear']

    for i, model_name in enumerate(all_models_to_plot):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        _, _, val_pred, val_mse = models[model_name]

        ax.scatter(y_val, val_pred, alpha=0.6, s=10)
        min_val = min(y_val.min(), val_pred.min())
        max_val = max(y_val.max(), val_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax.set_xlabel('Actual Stock Price')
        ax.set_ylabel('Predicted Stock Price')
        ax.set_title(f'{model_name}\nValidation MSE: {val_mse:.2f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def create_validation_visualizations(models: dict, y_test: np.ndarray, y_val: np.ndarray, results_df: pd.DataFrame) -> None:
    """Create validation visualizations.
    
    Args:
        models: Dictionary containing all model results.
        y_test: Test targets.
        y_val: Validation targets.
        results_df: DataFrame containing comparison results.
    """
    #  Validation Performance Visualizations
    print("Creating validation visualizations...")

    # Get best model based on validation performance
    best_model_name = results_df.loc[results_df['Val MSE'].idxmin(), 'Model']
    _, best_test_pred,best_val_pred,_ = models[best_model_name]

    plt.figure(figsize=(18, 12))

    # 1. Validation Predictions vs Actuals
    plt.subplot(3, 2, 1)
    plt.scatter(y_val, best_val_pred, alpha=0.7, s=60, c='red')
    min_val = min(y_val.min(), best_val_pred.min())
    max_val = max(y_val.max(), best_val_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
    plt.xlabel('Actual Stock Price ($)')
    plt.ylabel('Predicted Stock Price ($)')
    plt.title(f'Validation: Predicted vs Actual\n{best_model_name} (MSE: {results_df.loc[results_df["Val MSE"].idxmin(), "Val MSE"]:.2f})')
    plt.grid(True, alpha=0.3)

    # Add R² score
    val_r2 = r2_score(y_val, best_val_pred)
    plt.text(0.05, 0.95, f'R² = {val_r2:.3f}', transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. Test vs Validation Performance Comparison
    plt.subplot(3, 2, 2)
    plt.scatter(y_test, best_test_pred, alpha=0.6, s=40, c='blue', label='Test')
    plt.scatter(y_val, best_val_pred, alpha=0.7, s=60, c='red', label='Validation')
    min_val = min(y_test.min(), y_val.min(), best_test_pred.min(), best_val_pred.min())
    max_val = max(y_test.max(), y_val.max(), best_test_pred.max(), best_val_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
    plt.xlabel('Actual Stock Price ($)')
    plt.ylabel('Predicted Stock Price ($)')
    plt.title('Test vs Validation Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Validation Residuals Analysis
    plt.subplot(3, 2, 3)
    val_residuals = best_val_pred - y_val
    plt.scatter(y_val, val_residuals, alpha=0.7, s=60, c='red')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.8)
    plt.xlabel('Actual Stock Price ($)')
    plt.ylabel('Residual (Predicted - Actual)')
    plt.title('Validation Residuals vs Actual Prices')
    plt.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(y_val, val_residuals, 1)
    p = np.poly1d(z)
    plt.plot(sorted(y_val), p(sorted(y_val)), "r--", alpha=0.8, linewidth=2)

    # 4. Error Distribution by Price Range
    plt.subplot(3, 2, 4)
    # Bin prices into ranges
    price_bins = np.percentile(y_val, [0, 25, 50, 75, 100])
    bin_labels = ['Low', 'Med-Low', 'Med-High', 'High']
    val_errors = np.abs(val_residuals)

    binned_errors = []
    for i in range(len(price_bins)-1):
        mask = (y_val >= price_bins[i]) & (y_val < price_bins[i+1])
        if i == len(price_bins)-2:  # Last bin includes upper bound
            mask = (y_val >= price_bins[i]) & (y_val <= price_bins[i+1])
        binned_errors.append(val_errors[mask])

    plt.boxplot(binned_errors, labels=bin_labels)
    plt.xlabel('Price Range')
    plt.ylabel('Absolute Error ($)')
    plt.title('Validation Error Distribution by Price Range')
    plt.grid(True, alpha=0.3)

    # 5. Time Series of Validation Predictions
    plt.subplot(3, 2, 5)
    val_indices = range(len(y_val))
    plt.plot(val_indices, y_val, 'o-', label='Actual', linewidth=2, markersize=6)
    plt.plot(val_indices, best_val_pred, 's-', label='Predicted', linewidth=2, markersize=6)
    plt.xlabel('Validation Sample')
    plt.ylabel('Stock Price ($)')
    plt.title('Validation Time Series: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Highlight high price samples
    high_price_threshold = np.percentile(y_val, 75)
    high_price_mask = y_val > high_price_threshold
    plt.scatter(np.array(val_indices)[high_price_mask], y_val[high_price_mask], 
            c='red', s=100, alpha=0.5, label='High Prices')

    # 6. Model Comparison on Validation Set
    plt.subplot(3, 2, 6)
    model_names = []
    val_mses = []
    val_maes = []

    for name, (_, _, val_pred, _) in models.items():
        model_names.append(name)
        val_mses.append(mean_squared_error(y_val, val_pred))
        val_maes.append(mean_absolute_error(y_val, val_pred))

    x_pos = np.arange(len(model_names))
    plt.bar(x_pos, val_mses, alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('Validation MSE')
    plt.title('Model Comparison: Validation MSE')
    plt.xticks(x_pos, model_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    # Highlight best model
    best_idx = val_mses.index(min(val_mses))
    plt.bar(best_idx, val_mses[best_idx], color='red', alpha=0.8)

    plt.tight_layout()
    plt.show()
