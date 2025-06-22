"""Feature Engineering module for stock price forecasting."""
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import ta  # Technical analysis library


def setup_directories(ticker: str) -> None:
    """Create necessary directories for processed data.
    
    Args:
        ticker: Ticker symbol of the company.
    """
    os.makedirs('../data/processed', exist_ok=True)
    os.makedirs(f'../data/processed/{ticker}', exist_ok=True)
    os.makedirs(f'../data/final/{ticker}', exist_ok=True)


def load_stock_data(ticker: str) -> tuple[pd.DataFrame, datetime, datetime]:
    """Load and process stock data.
    
    Args:
        ticker: Ticker symbol of the company.
        
    Returns:
        tuple: (stock_data, start_date, end_date)
    """
    print("Loading stock data...")

    try:
        # Load stock data
        stock_data_path = f'../data/raw_data/{ticker}/stock_data.csv'
        col_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

        # Skip the first two rows and set column names
        stock_data = pd.read_csv(
            stock_data_path,
            skiprows=3,
            names=col_names
        )
        print(f"Loaded stock data with shape {stock_data.shape}")
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)

        start_date = stock_data.index.min()
        end_date = stock_data.index.max()

        return stock_data, start_date, end_date

    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
        print(f"Error loading stock data: {e}")
        raise


def process_economic_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load and process daily economic data.
    
    Args:
        ticker: Ticker symbol of the company.
        start_date: Start date for data processing.
        end_date: End date for data processing.
        
    Returns:
        DataFrame: Processed economic data.
    """
    print("\nProcessing daily economic data...")

    try:
        # Load daily economic data (already has daily Treasury rates)
        economic_data_path = f'../data/raw_data/{ticker}/economic_data.csv'

        if os.path.exists(economic_data_path):
            economic_data = pd.read_csv(economic_data_path, parse_dates=['DATE'])
            economic_data.set_index('DATE', inplace=True)
            print(f"Loaded daily economic data with {len(economic_data)} rows")

            # Reindex to match stock data business days
            daily_dates = pd.date_range(start=start_date, end=end_date, freq='B')
            economic_data_daily = economic_data.reindex(daily_dates)

            # Process each column appropriately
            # Treasury rates (10Y, 3M) - already daily, just forward fill gaps
            for col in ['Treasury_10Y', 'Treasury_3M']:
                if col in economic_data_daily.columns:
                    economic_data_daily[col] = economic_data_daily[col].ffill()
                    print(f"Forward filled {col}")

            # Fed Funds Rate - step function (changes on announcement dates)
            if 'Fed_Funds_Rate' in economic_data_daily.columns:
                economic_data_daily['Fed_Funds_Rate'] = economic_data_daily['Fed_Funds_Rate'].ffill()
                print("Forward filled Fed_Funds_Rate")

            # Monthly economic indicators - forward fill with reporting lag
            monthly_indicators = ['CPI', 'PPI', 'Unemployment_Rate', 'Industrial_Production', 
                                'Consumer_Sentiment', 'Housing_Starts']

            for col in monthly_indicators:
                if col in economic_data_daily.columns:
                    # Forward fill and apply reporting lag
                    economic_data_daily[col] = economic_data_daily[col].ffill().shift(21)
                    print(f"Forward filled and lagged {col}")

            # Visualize sample indicators
            sample_indicators = {
                'Treasury_10Y': 'Daily Data (Forward Fill)',
                'Fed_Funds_Rate': 'Step Function',
                'CPI': 'Monthly Data + Lag'
            }

            for indicator, method in sample_indicators.items():
                if indicator in economic_data_daily.columns and not economic_data_daily[indicator].isna().all():
                    plt.figure(figsize=(14, 8))

                    plt.plot(economic_data_daily.index, economic_data_daily[indicator], 
                            label=f'Daily {indicator} ({method})', color='blue', linewidth=1.5)

                    plt.title(f'Daily {indicator} Data')
                    plt.ylabel(indicator)
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

            # Save the processed daily economic data
            economic_data_daily.to_csv(f'../data/processed/{ticker}/economic_data_daily.csv')
            print(f"\nProcessed daily economic data saved to ../data/processed/{ticker}/economic_data_daily.csv")
            print(f"Shape: {economic_data_daily.shape}")

            return economic_data_daily

        else:
            print(f"Daily economic data file not found at {economic_data_path}")
            return pd.DataFrame()

    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
        print(f"Error processing daily economic data: {e}")
        return pd.DataFrame()


def process_market_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Process market data with forward filling.
    
    Args:
        ticker: Ticker symbol of the company.
        start_date: Start date for data processing.
        end_date: End date for data processing.
        
    Returns:
        DataFrame: Processed market data.
    """
    try:
        # Forward fill for market data
        print("\nForward filling market data...")

        # Load market data
        market_data = pd.read_csv(f'../data/raw_data/{ticker}/market_data.csv', index_col=0, parse_dates=True)

        # Reindex market_data to include all dates up to last_date
        all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        market_data = market_data.reindex(all_dates)

        # Forward fill missing values
        market_data = market_data.ffill()

        # Save the result if needed
        market_data.to_csv(f'../data/processed/{ticker}/market_data_filled.csv')
        print(f"Market data forward filled and saved to ../data/processed/{ticker}/market_data_filled.csv")

        return market_data
        
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
        print(f"Error processing market data: {e}")
        return pd.DataFrame()


def combine_all_data(ticker: str) -> pd.DataFrame:
    """Combine all data sources into a single DataFrame.
    
    Args:
        ticker: Ticker symbol of the company.
        
    Returns:
        DataFrame: Combined data from all sources.
    """
    # Define file paths
    data_paths = {
        'stock_data': f'../data/raw_data/{ticker}/stock_data.csv',
        'market_data': f'../data/raw_data/{ticker}/market_data.csv',
        'economic_data': f'../data/processed/{ticker}/economic_data_daily.csv'
    }

    dfs = {}
    missing_files = []

    # Load each DataFrame with error handling
    for name, path in data_paths.items():
        if os.path.exists(path):
            try:
                if name == 'stock_data':
                    col_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
                    df = pd.read_csv(path, skiprows=3, names=col_names)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    print(f"Loaded {name} with shape {df.shape}")
                else:
                    df = pd.read_csv(path, index_col=0, parse_dates=True)
                dfs[name] = df

            except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
                print(f"Error loading {name} from {path}: {e}")
        else:
            print(f"File not found: {path}")
            missing_files.append(name)

    if missing_files:
        print(f"Missing files: {missing_files}. Cannot combine all data.")
        return pd.DataFrame()
    else:
        # Use stock_data's index as the master date index
        master_index = dfs['stock_data'].index

        # Reindex all other DataFrames to match stock_data's dates
        for name in dfs:
            if name != 'stock_data':
                dfs[name] = dfs[name].reindex(master_index)

        # Combine all DataFrames on the date index (outer join for robustness)
        combined = pd.concat([dfs['stock_data'], dfs['market_data'], dfs['economic_data']], axis=1, join='outer')
        combined = combined.sort_index().sort_index(axis=1)
        combined.to_csv(f'../data/processed/{ticker}/combined_daily_data.csv')
        print("Combined DataFrame shape:", combined.shape)
        print(f"Combined DataFrame saved to ../data/processed/{ticker}/combined_daily_data.csv")

        return combined


def calculate_technical_indicators(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate technical indicators from combined data.
    
    Args:
        ticker: Ticker symbol of the company.
        
    Returns:
        tuple: (technical_indicators, combined_with_indicators)
    """
    print("\nCalculating technical indicators from combined data...")

    try:
        combined_path = f'../data/processed/{ticker}/combined_daily_data.csv'
        if os.path.exists(combined_path):
            combined = pd.read_csv(combined_path, index_col=0, parse_dates=True)
            close_series = combined['Close']
            high_series = combined['High']
            low_series = combined['Low']
            volume_series = combined['Volume']

            # Moving Averages
            combined['SMA_20'] = ta.trend.sma_indicator(close_series, window=20)
            combined['SMA_50'] = ta.trend.sma_indicator(close_series, window=50)
            combined['EMA_20'] = ta.trend.ema_indicator(close_series, window=20)

            # Volatility Indicators
            combined['ATR'] = ta.volatility.average_true_range(high_series, low_series, close_series, window=14)
            bollinger = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
            combined['Bollinger_Upper'] = bollinger.bollinger_hband()
            combined['Bollinger_Lower'] = bollinger.bollinger_lband()

            # Momentum Indicators
            combined['RSI'] = ta.momentum.rsi(close_series, window=14)


            # Volume Indicators
            combined['OBV'] = ta.volume.on_balance_volume(close_series, volume_series)

            # Delete created NaN values
            combined = combined.dropna()

            # Save only the technical indicators
            tech_cols = ['SMA_20','SMA_50','EMA_20','ATR','Bollinger_Upper','Bollinger_Lower','RSI','OBV']
            technical_indicators = combined[tech_cols]
            technical_indicators.to_csv(f'../data/processed/{ticker}/technical_indicators.csv')
            print(f"Technical indicators calculated and saved to ../data/processed/{ticker}/technical_indicators.csv")
            combined.to_csv(f'../data/processed/{ticker}/technical_indicators_combined.csv')
            print(f'Combined Technical indicators with combined_daily_data and saved to ../data/processed/{ticker}/technical_indicators_combined.csv')

            return technical_indicators, combined
        else:
            print(f"Combined data not found at {combined_path}")
            return pd.DataFrame(), pd.DataFrame()
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
        print(f"Error calculating technical indicators: {e}")
        return pd.DataFrame(), pd.DataFrame()


def process_economic_features(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process economic indicators features from technical indicators combined data.
    
    Args:
        ticker: Ticker symbol of the company.
        
    Returns:
        tuple: (economic_features, combined_features)
    """
    print("Processing economic indicators features from technical_indicators_combined...")

    try:
        combined_path = f'../data/processed/{ticker}/technical_indicators_combined.csv'
        if os.path.exists(combined_path):
            combined = pd.read_csv(combined_path, index_col=0, parse_dates=True)
            econ = combined.copy()

            # 1. Calculate inflation rates (year-over-year changes, daily aligned)
            if 'CPI' in econ.columns:
                econ['CPI_YoY_Change'] = econ['CPI'].pct_change(periods=252, fill_method=None) * 100  # 252 trading days ~ 1 year
                print("Calculated CPI year-over-year change (daily aligned)")

            if 'PPI' in econ.columns:
                econ['PPI_YoY_Change'] = econ['PPI'].pct_change(periods=252, fill_method=None) * 100
                print("Calculated PPI year-over-year change (daily aligned)")

            # 2. Calculate rate differentials (spreads)
            if 'Treasury_10Y' in econ.columns and 'Treasury_3M' in econ.columns:
                econ['Yield_Curve_Spread'] = econ['Treasury_10Y'] - econ['Treasury_3M']
                print("Calculated yield curve spread (10Y-3M)")

            if 'Fed_Funds_Rate' in econ.columns and 'Treasury_10Y' in econ.columns:
                econ['Fed_10Y_Spread'] = econ['Treasury_10Y'] - econ['Fed_Funds_Rate']
                print("Calculated Fed Funds to 10Y Treasury spread")

            # 3. Calculate rate momentum (rate of change)
            for rate_col in ['Treasury_10Y', 'Treasury_3M']:
                if rate_col in econ.columns:
                    econ[f'{rate_col}_DoD'] = econ[rate_col].diff()
                    print(f"Calculated day-over-day change for {rate_col}")

            # Save the enhanced economic features
            econ_features = econ[[col for col in econ.columns if
                                  'CPI_YoY_Change' in col or
                                  'PPI_YoY_Change' in col or
                                  'Spread' in col or
                                  '_DoD' in col]]
            econ_features.to_csv(f'../data/processed/{ticker}/economic_features.csv')

            # Delete created NaN values
            econ = econ.dropna()

            print("Saving economic features...")
            print(f"Economic features (daily) saved to ../data/processed/{ticker}/economic_features.csv")

            print("Saving combined features...")
            econ.to_csv(f'../data/final/{ticker}/preprocessed_features.csv')
            print(f"Combined features saved to ../data/final/{ticker}/preprocessed_features.csv")

            return econ_features, econ
        else:
            print(f"Combined data not found at {combined_path}")
            return pd.DataFrame(), pd.DataFrame()

    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
        print(f"Error processing economic features: {e}")
        return pd.DataFrame(), pd.DataFrame()
