"""
Prophet Forecasting Model with Regressors
This script uses Meta's Prophet to forecast time series data with additional regressors:
- Weather (rainy/sunny/overcast)
- Temperature
- Festival events
- Time of day

Supports multiple products - creates separate forecasts for each product.
Input CSV format: Each row represents a product sale record with timestamp, product name, 
units sold, and regressors. Sales are recorded in 4-hour intervals (8AM-12PM, 12PM-4PM, 4PM-8PM).
"""

import pandas as pd
import numpy as np
import re
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(csv_path):
    """
    Load CSV data in long format where each row represents a product sale.
    
    Expected CSV format:
    - timestamp: datetime column (4-hour intervals: 8AM, 12PM, 4PM)
    - product: product/SKU name (e.g., 'Danish', 'Muffins', 'Coffee', 'Tea')
    - units_sold: number of units sold (target variable 'y')
    - weather: categorical (rainy/sunny/overcast)
    - temperature: numeric temperature values
    - festival_events: binary (0/1) or numeric indicating festival presence
    - time_of_day: hour of day (0-23) or time feature
    
    Returns:
        DataFrame with all data, plus list of unique product names
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Check for required columns
    required_cols = ['timestamp', 'product', 'units_sold']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV must contain these columns: {missing_cols}")
    
    # Convert timestamp to datetime
    df['ds'] = pd.to_datetime(df['timestamp'])
    
    # Get unique products
    unique_products = sorted(df['product'].unique())
    
    if len(unique_products) == 0:
        raise ValueError("No products found in the 'product' column")
    
    print(f"Found {len(unique_products)} unique products: {unique_products}")
    
    # Encode weather as numeric (one-hot encoding would create multiple regressors)
    # For simplicity, we'll encode as: sunny=0, overcast=1, rainy=2
    weather_mapping = {'sunny': 0, 'overcast': 1, 'rainy': 2}
    if 'weather' in df.columns:
        df['weather_encoded'] = df['weather'].map(weather_mapping).fillna(0)
    
    # Ensure numeric columns are numeric
    numeric_cols = ['units_sold', 'temperature', 'festival_events', 'time_of_day']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df, unique_products


def prepare_product_data(df, product_name):
    """
    Prepare data for a specific product.
    
    Args:
        df: Full dataframe with all product sales
        product_name: Name of the product (e.g., 'Danish', 'Muffins')
    
    Returns:
        DataFrame prepared for Prophet (with 'ds', 'y', and regressors)
    """
    # Filter data for this product
    product_df = df[df['product'] == product_name].copy()
    
    if len(product_df) == 0:
        return None
    
    # Rename units_sold to y
    product_df = product_df.rename(columns={'units_sold': 'y'})
    
    # Remove rows where y is NaN
    product_df = product_df.dropna(subset=['y'])
    
    if len(product_df) == 0:
        return None
    
    # Sort by timestamp
    product_df = product_df.sort_values('ds').reset_index(drop=True)
    
    # Select only required columns for Prophet
    prophet_cols = ['ds', 'y']
    regressor_cols = ['weather_encoded', 'festival_events']
    for col in regressor_cols:
        if col in product_df.columns:
            prophet_cols.append(col)
    
    return product_df[prophet_cols].copy()


def load_future_regressors(csv_path):
    """
    Load future regressor values from CSV file.
    
    Expected CSV format:
    - timestamp: datetime column
    - weather: categorical (rainy/sunny/overcast)
    - festival_events: binary (0/1) or numeric
    
    Returns:
        DataFrame with 'ds', 'weather_encoded', and 'festival_events' columns
    """
    if csv_path is None or not os.path.exists(csv_path):
        return None
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Check for required columns
    required_cols = ['timestamp', 'weather', 'festival_events']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Future regressors file missing columns: {missing_cols}. Using last known values.")
        return None
    
    # Convert timestamp to datetime
    df['ds'] = pd.to_datetime(df['timestamp'])
    
    # Encode weather
    weather_mapping = {'sunny': 0, 'overcast': 1, 'rainy': 2}
    df['weather_encoded'] = df['weather'].map(weather_mapping).fillna(0)
    
    # Ensure festival_events is numeric
    df['festival_events'] = pd.to_numeric(df['festival_events'], errors='coerce').fillna(0)
    
    # Select only required columns
    return df[['ds', 'weather_encoded', 'festival_events']].copy()


def create_prophet_model(df):
    """
    Create and configure Prophet model with regressors.
    """
    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative'  # Can be 'additive' or 'multiplicative'
    )
    
    # Add regressors
    if 'weather_encoded' in df.columns:
        model.add_regressor('weather_encoded')
        print("Added regressor: weather_encoded")
    
    if 'festival_events' in df.columns:
        model.add_regressor('festival_events')
        print("Added regressor: festival_events")
    
    return model


def train_and_forecast(model, df, periods=30, future_regressors=None):
    """
    Train the model and make forecasts.
    
    Args:
        model: Prophet model instance
        df: DataFrame with 'ds' and 'y' columns plus regressors
        periods: Number of future periods to forecast
        future_regressors: DataFrame with future regressor values (optional)
    """
    # Fit the model
    print("\nTraining Prophet model...")
    model.fit(df)
    print("Model training completed!")
    
    # Create future dataframe with specific times: 8:00, 12:00, 16:00 for each day
    # Get the last timestamp from training data
    last_timestamp = df['ds'].max()
    
    # Calculate number of days to forecast (periods / 3 observations per day)
    num_days = periods // 3
    if periods % 3 != 0:
        num_days += 1
    
    # Generate future timestamps: start from next day at 8:00
    next_day = last_timestamp.normalize() + pd.Timedelta(days=1)
    future_timestamps = []
    
    # Generate timestamps for the specified number of days
    for day in range(num_days):
        current_date = next_day + pd.Timedelta(days=day)
        # Add 8:00, 12:00, and 16:00 for each day
        future_timestamps.extend([
            current_date + pd.Timedelta(hours=8),
            current_date + pd.Timedelta(hours=12),
            current_date + pd.Timedelta(hours=16)
        ])
    
    # Take only the requested number of periods
    future_timestamps = future_timestamps[:periods]
    
    # Create future-only dataframe for adding regressors
    future_only = pd.DataFrame({'ds': future_timestamps})
    
    # Combine with historical data to create full dataframe for Prophet
    # Prophet needs both historical and future data in one dataframe
    historical_ds = df[['ds']].copy()
    future = pd.concat([historical_ds, future_only], ignore_index=True)
    future = future.drop_duplicates(subset=['ds']).sort_values('ds').reset_index(drop=True)
    
    # Add regressors to future dataframe
    # First, add historical regressor values from training data
    regressor_cols = [col for col in df.columns if col not in ['ds', 'y']]
    for col in regressor_cols:
        # Create a mapping from ds to regressor value for historical data
        historical_map = dict(zip(df['ds'], df[col]))
        future[col] = future['ds'].map(historical_map)
    
    # Add future regressors if provided
    if future_regressors is not None:
        # Merge future regressors (this will add/update values for future timestamps)
        for col in future_regressors.columns:
            if col != 'ds':
                # Create mapping for future regressor values
                future_map = dict(zip(future_regressors['ds'], future_regressors[col]))
                # Update future periods with regressor values from future_regressors
                future.loc[future['ds'] > last_timestamp, col] = future.loc[
                    future['ds'] > last_timestamp, 'ds'
                ].map(future_map)
                # Forward fill any remaining missing values
                future[col] = future[col].ffill().fillna(0)
    else:
        # If no future regressors provided, use last known values for future periods
        last_values = df.iloc[-1]
        for col in regressor_cols:
            # Fill future values with last known value
            future.loc[future['ds'] > last_timestamp, col] = last_values[col]
    
    # Make predictions
    print(f"\nMaking forecasts for {periods} periods...")
    forecast = model.predict(future)
    
    return forecast


def plot_results(model, forecast, df, product_name, output_dir='outputs'):
    """
    Plot the forecast results for a specific product.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize product name for filename
    safe_product_name = product_name.replace(' ', '_').replace('/', '_')
    
    # Plot 1: Forecast
    fig1 = model.plot(forecast)
    plt.title(f'Prophet Forecast - Product: {product_name}')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'prophet_forecast_{safe_product_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved forecast plot to '{plot_path}'")
    plt.close()
    
    # Plot 2: Components
    fig2 = model.plot_components(forecast)
    components_path = os.path.join(output_dir, f'prophet_components_{safe_product_name}.png')
    plt.savefig(components_path, dpi=300, bbox_inches='tight')
    print(f"Saved components plot to '{components_path}'")
    plt.close()


def create_wide_format_forecast(all_forecasts):
    """
    Create a wide format consolidated forecast where each product's forecast values
    are in separate columns.
    
    Args:
        all_forecasts: List of forecast DataFrames, each with 'product', 'ds', 'yhat', 
                      'yhat_lower', 'yhat_upper' columns
    
    Returns:
        DataFrame in wide format with columns:
        - ds: timestamp
        - ProductName_yhat, ProductName_yhat_lower, ProductName_yhat_upper
        - etc.
    """
    if not all_forecasts:
        return pd.DataFrame()
    
    # Get all unique timestamps from all forecasts
    all_timestamps = set()
    for forecast in all_forecasts:
        all_timestamps.update(forecast['ds'].values)
    
    # Create base dataframe with all timestamps
    wide_df = pd.DataFrame({'ds': sorted(all_timestamps)})
    
    # For each forecast, add columns for that product
    for forecast in all_forecasts:
        product = forecast['product'].iloc[0]
        # Sanitize product name for column name
        safe_product_name = product.replace(' ', '_').replace('/', '_')
        
        # Create product-specific forecast columns
        product_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        product_forecast = product_forecast.rename(columns={
            'yhat': f'{safe_product_name}_yhat',
            'yhat_lower': f'{safe_product_name}_yhat_lower',
            'yhat_upper': f'{safe_product_name}_yhat_upper'
        })
        
        # Merge with wide_df on 'ds'
        wide_df = wide_df.merge(product_forecast, on='ds', how='left')
    
    # Sort by timestamp
    wide_df = wide_df.sort_values('ds').reset_index(drop=True)
    
    return wide_df


def process_product(df, product_name, forecast_periods, output_dir='outputs', future_regressors=None, skip_plots=False):
    """
    Process forecasting for a single product.
    
    Args:
        df: Full dataframe with all product sales
        product_name: Name of the product (e.g., 'Danish', 'Muffins')
        forecast_periods: Number of periods to forecast
        output_dir: Directory to save outputs
        future_regressors: DataFrame with future regressor values (optional)
        skip_plots: If True, skip generating plot files (useful for API contexts)
    
    Returns:
        Dictionary with forecast results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Processing Product: {product_name}")
    print(f"{'='*60}")
    
    # Prepare data for this product
    product_df = prepare_product_data(df, product_name)
    
    if product_df is None or len(product_df) == 0:
        print(f"Warning: No data found for {product_name}")
        return None
    
    print(f"Data points: {len(product_df)}")
    print(f"Date range: {product_df['ds'].min()} to {product_df['ds'].max()}")
    
    # Create model
    print("\nCreating Prophet model...")
    model = create_prophet_model(product_df)
    
    # Train and forecast
    forecast = train_and_forecast(model, product_df, periods=forecast_periods, future_regressors=future_regressors)
    
    # Add product column to forecast
    forecast['product'] = product_name
    
    # Display forecast summary
    print(f"\nLast {min(10, forecast_periods)} forecasted values:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(min(10, forecast_periods)))
    
    # Save forecast to CSV
    os.makedirs(output_dir, exist_ok=True)
    forecast_output = forecast[['product', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    # Sanitize product name for filename (replace spaces/special chars)
    safe_product_name = product_name.replace(' ', '_').replace('/', '_')
    forecast_path = os.path.join(output_dir, f'forecast_{safe_product_name}.csv')
    forecast_output.to_csv(forecast_path, index=False)
    print(f"\nSaved forecast to '{forecast_path}'")
    
    # Plot results (skip in API context)
    if not skip_plots:
        plot_results(model, forecast, product_df, product_name, output_dir)
    
    # Calculate model performance metrics
    metrics = {}
    if len(product_df) > 0:
        # Merge actual values with forecast
        comparison = forecast[['ds', 'yhat']].merge(product_df[['ds', 'y']], on='ds', how='inner')
        if len(comparison) > 0:
            mae = np.mean(np.abs(comparison['y'] - comparison['yhat']))
            rmse = np.sqrt(np.mean((comparison['y'] - comparison['yhat'])**2))
            metrics = {'mae': mae, 'rmse': rmse}
            print(f"\nModel Performance (on training data):")
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
    
    return {
        'product': product_name,
        'forecast': forecast,
        'metrics': metrics,
        'data_points': len(product_df),
        'product_df': product_df  # Include for API to determine future vs historical
    }


def main():
    """
    Main function to run the Prophet forecasting pipeline for multiple SKUs.
    """
    # Configuration
    CSV_PATH = 'data.csv'  # Change this to your CSV file path
    FUTURE_REGRESSORS_PATH = 'future_regressors.csv'  # Path to future regressors file
    FORECAST_PERIODS = 21  # Number of periods to forecast (7 days * 3 observations per day)
    OUTPUT_DIR = 'outputs'  # Directory to save outputs
    
    try:
        # Load and prepare data
        print(f"Loading data from {CSV_PATH}...")
        df, product_names = load_and_prepare_data(CSV_PATH)
        print(f"Loaded {len(df)} rows of data")
        
        # Load future regressors
        print(f"\nLoading future regressors from {FUTURE_REGRESSORS_PATH}...")
        future_regressors = load_future_regressors(FUTURE_REGRESSORS_PATH)
        if future_regressors is not None:
            print(f"Loaded {len(future_regressors)} future regressor rows")
            print(f"Future regressor date range: {future_regressors['ds'].min()} to {future_regressors['ds'].max()}")
        else:
            print("No future regressors file found. Using last known values for all future periods.")
        
        # Process each product
        all_results = []
        all_forecasts = []
        
        for product_name in sorted(product_names):
            result = process_product(df, product_name, FORECAST_PERIODS, OUTPUT_DIR, future_regressors)
            if result:
                all_results.append(result)
                all_forecasts.append(result['forecast'])
        
        # Create combined forecast file in wide format
        if all_forecasts:
            combined_forecast = create_wide_format_forecast(all_forecasts)
            combined_path = os.path.join(OUTPUT_DIR, 'forecast_all_skus.csv')
            combined_forecast.to_csv(combined_path, index=False)
            print(f"\n{'='*60}")
            print(f"Saved combined forecast (wide format) to '{combined_path}'")
        
        # Print summary
        print(f"\n{'='*60}")
        print("FORECASTING SUMMARY")
        print(f"{'='*60}")
        print(f"\nProcessed {len(all_results)} product(s) successfully")
        
        if all_results:
            print("\nPerformance Metrics by Product:")
            print("-" * 60)
            for result in all_results:
                product = result['product']
                metrics = result['metrics']
                data_points = result['data_points']
                print(f"\nProduct: {product}")
                print(f"  Data points: {data_points}")
                if metrics:
                    print(f"  MAE: {metrics['mae']:.2f}")
                    print(f"  RMSE: {metrics['rmse']:.2f}")
        
        print(f"\n{'='*60}")
        print("Forecasting completed successfully!")
        print(f"{'='*60}")
        
    except FileNotFoundError:
        print(f"Error: File '{CSV_PATH}' not found.")
        print("Please ensure the CSV file exists or update CSV_PATH in the script.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

