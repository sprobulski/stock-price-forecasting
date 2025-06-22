"""Data collection module for stock price forecasting."""
import os
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import pandas_datareader as pdr


def get_ticker_info(years: int = 7) -> tuple[str, str, dict, datetime, datetime]:
    """Get ticker information and date range for data collection.
    
    Args:
        years: Number of years of data to collect.
        
    Returns:
        tuple:
            - ticker: Ticker symbol of the company.
            - company_name: Name of the company.
            - company_info: Dictionary containing company information.
            - start_date: Start date for data collection.
            - end_date: End date for data collection.
    """

    # Default values
    ticker = "AAPL"
    company_name = "Apple Inc."
    start_date = datetime.now() - timedelta(days=365*years)
    end_date = datetime.now()

    # User Input: Specify the company ticker you want to predict
    ticker = input(
        "Enter the ticker symbol of the company you want to predict (e.g., AAPL, MSFT, TSLA):\n"
        f"Default: {ticker}\n"
        ).upper()

    # Set up date ranges
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)

    # Get company name for display purposes
    try:
        company_info = yf.Ticker(ticker).info
        company_name = company_info.get('shortName', ticker)
        print(f"Selected company: {company_name} ({ticker})")
    except (KeyError, ConnectionError, ValueError) as e:
        print(f"Error: {e}")
        print("Using default values")
        company_info = None

    print(f"Data collection period: {start_date.strftime('%Y-%m-%d')} "
          f"to {end_date.strftime('%Y-%m-%d')}")
    return ticker, company_name, company_info, start_date, end_date


def get_financial_data(ticker: str, company_info: dict, start_date: datetime, end_date: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get stock data for a given ticker and date range.
    
    Args:
        ticker: Ticker symbol of the company.
        company_info: Dictionary containing company information.
        start_date: Start date for data collection.
        end_date: End date for data collection.
    """

    ## 1.1 Download Historical OHLCV Data
    print(f"\nDownloading historical OHLCV data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    # Check if we got data
    if stock_data.empty:
        print(f"No data available for {ticker}. Please check if the ticker symbol is correct.")
    else:
        print(f"Downloaded {len(stock_data)} days of historical data")

        # Display the first few rows
        print("\nFirst few rows of historical OHLCV data:")
        print(stock_data.head())

    ## 1.2 Market Context Data (Indices and Sector)
    print("\nDownloading market context data...")

    # Determine which sector ETF is most relevant for this company
    try:
        sector = company_info.get('sector', 'Unknown')
        industry = company_info.get('industry', 'Unknown')
        print(f"Company sector: {sector}")
        print(f"Company industry: {industry}")
    except (KeyError, ConnectionError, ValueError) as e:
        print(f"Error: {e}")
        sector = "Unknown"
        industry = "Unknown"
        print("Could not determine company sector/industry")

    # Download relevant market indices
    indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']  # S&P 500, Dow Jones, NASDAQ, VIX
    indices_names = ['S&P_500', 'Dow_Jones', 'NASDAQ', 'VIX']

    # Map sectors to sector ETFs
    sector_etf_map = {
        'Technology': 'XLK',
        'Financial Services': 'XLF',
        'Healthcare': 'XLV',
        'Consumer Defensive': 'XLP',
        'Consumer Cyclical': 'XLY',
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Basic Materials': 'XLB',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC'
    }

    # Add relevant sector ETF if we can identify it
    relevant_etf = sector_etf_map.get(sector, None)
    if relevant_etf:
        indices.append(relevant_etf)
        indices_names.append(f'Sector_{sector}')
        print(f"Added sector ETF: {relevant_etf} for {sector}")

    # Download indices data
    market_data = yf.download(indices, start=start_date, end=end_date,auto_adjust=True)['Close']
    if len(indices) > 1:  # If we have multiple indices
        market_data.columns = indices_names
    else:  # If we only have one index
        market_data = pd.DataFrame(market_data)
        market_data.columns = indices_names

    # Save the data
    os.makedirs(f'../data/raw_data/{ticker}', exist_ok=True)
    stock_data.to_csv(f'../data/raw_data/{ticker}/stock_data.csv')
    market_data.to_csv(f'../data/raw_data/{ticker}/market_data.csv')

    return stock_data, market_data


def get_economic_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Get economic data for a given ticker and date range.
    
    Args:
        ticker: Ticker symbol of the company.
        start_date: Start date for data collection.
        end_date: End date for data collection.
    """

    print("Downloading economic indicators relevant for stock price prediction...")

    # For demonstration, we'll use pandas_datareader which doesn't require an API key
    print("Getting economic data from FRED...")

    # List of key economic indicators to download
    economic_indicators = {
        'FEDFUNDS': 'Fed_Funds_Rate',       # Federal Funds Rate
        'DGS10': 'Treasury_10Y',            # 10-Year Treasury Rate
        'DTB3': 'Treasury_3M',              # 3-Month Treasury Rate
        'CPIAUCSL': 'CPI',                  # Consumer Price Index
        'PPIACO': 'PPI',                    # Producer Price Index
        'UNRATE': 'Unemployment_Rate',      # Unemployment Rate
        'INDPRO': 'Industrial_Production',  # Industrial Production
        'UMCSENT': 'Consumer_Sentiment',    # Consumer Sentiment
        'HOUST': 'Housing_Starts'           # Housing Starts
    }

    # Download all indicators
    economic_data = {}
    for fred_code, indicator_name in economic_indicators.items():
        try:
            data = pdr.get_data_fred(fred_code, start=start_date, end=end_date)
            if not data.empty:
                economic_data[indicator_name] = data
                print(f"Downloaded {indicator_name} data: {len(data)} observations")
            else:
                print(f"No data available for {indicator_name}")
        except (ValueError, KeyError, ConnectionError, pd.errors.EmptyDataError) as e:
            print(f"Error downloading {indicator_name}: {e}")

    # Combine all economic indicators into one DataFrame
    combined_economic = pd.DataFrame()
    for name, data in economic_data.items():
        if not data.empty:
            data.columns = [name]
            if combined_economic.empty:
                combined_economic = data
            else:
                combined_economic = combined_economic.join(data, how='outer')

    # We'll save the raw data without calculating derived features
    # Year-over-year calculations will be done in the feature engineering notebook

    if not combined_economic.empty:
        print(f"Downloaded {len(combined_economic)} days of economic data")

        # Save both the original and resampled data
        combined_economic.to_csv(f'../data/raw_data/{ticker}/economic_data.csv')
        print(f"Economic data saved to ../data/raw_data/{ticker}/economic_data.csv")

    else:
        print("No economic data was downloaded successfully.")

    return combined_economic


def create_data_summary(ticker: str) -> pd.DataFrame:
    """Summarize the data for a given ticker and date range.
    
    Args:
        ticker: Ticker symbol of the company.
    """

    print("\nData collection completed for ticker:", ticker)
    print("\nThe following data files have been saved:")

    # Check which files exist
    files_saved = {
        'Stock OHLCV Data': os.path.exists(f'../data/raw_data/{ticker}/stock_data.csv'),
        'Market Context Data': os.path.exists(f'../data/raw_data/{ticker}/market_data.csv'),
        'Economic Indicators': os.path.exists(f'../data/raw_data/{ticker}/economic_data.csv'),
    }

    # Create a summary DataFrame
    summary_df = pd.DataFrame({'Data Available': files_saved})
    print(summary_df)

    # Save the summary
    summary_df.to_csv(f'../data/raw_data/{ticker}/data_collection_summary.csv')

    return summary_df

def create_readme(ticker: str, stock_data: pd.DataFrame, company_name: str) -> None:
    """Create a conclusion for the data collection process.
    
    Args:
        ticker: Ticker symbol of the company.
        stock_data: Stock data for the company.
        company_name: Name of the company.
    """

    # 4. Conclusion
    print("\nData Collection Process Complete!")
    print(f"All data for {ticker} has been collected and saved to the ../data/raw_data/{ticker}/ directory.")

    # Show a summary of the data collection process
    print("\nSummary of Collected Data:")
    print(f"- Historical OHLCV data: {len(stock_data)} days")
    print("- Market context data: S&P 500, Dow Jones, NASDAQ, VIX, and relevant sector ETF")
    print("- Economic indicators: Interest rates, inflation, unemployment, etc.")

    print("\nNext Steps:")
    print("1. Open the 02_feature_engineering.ipynb notebook")
    print("2. Combine all data sources into a single dataset")
    print("3. Create additional derived features")
    print("4. Prepare the data for modeling")
    print("5. Select and train predictive models")

    # Create a README file in the data directory
    readme_content = f"""# Data Collection for {ticker} Stock Price Forecasting

    ## Data Collection Date: {datetime.now().strftime('%Y-%m-%d')}

    This directory contains the following data files for {ticker} ({company_name}):

    - stock_data.csv: Historical OHLCV (Open, High, Low, Close, Volume) data
    - market_data.csv: Market indices and relevant sector ETF data
    - economic_data.csv: Economic indicators from FRED

    This data is used for stock price forecasting as part of the project.
    """

    # Write the README file
    with open(f'../data/raw_data/{ticker}/README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"\nA README.md file has been created in the ../data/raw_data/{ticker}/ directory with details about the collected data.")

