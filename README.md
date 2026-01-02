# Prophet Forecasting Model with Multiple SKUs

This project implements Meta's Prophet forecasting model with additional regressors for time series prediction. **It supports multiple SKUs and creates independent forecasts for each SKU.**

## Features

- Uses Prophet for time series forecasting
- **Multi-SKU support**: Automatically splits data by SKU and creates separate models
- Supports regressors:
  - **Weather**: Categorical (rainy/sunny/overcast) - encoded as numeric
  - **Festival Events**: Binary indicator (0/1)
- **Future regressors**: Optionally provide future regressor values via `future_regressors.csv`

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Note: Prophet requires additional system dependencies. On macOS:
```bash
brew install cmake
```

On Ubuntu/Debian:
```bash
sudo apt-get install cmake
```

## Usage

1. Prepare your CSV file in **long format** where each row represents a product sale record:
   - `timestamp`: DateTime column (required) - 4-hour intervals: 8AM, 12PM, 4PM
   - `product`: Product/SKU name (required) - e.g., 'Danish', 'Muffins', 'Coffee', 'Tea'
   - `units_sold`: Number of units sold (required) - target variable for forecasting
   - `weather`: Categorical (rainy/sunny/overcast) - regressor
   - `festival_events`: Binary (0/1) or numeric - regressor
   
   **Example format:**
   ```
   timestamp,product,units_sold,weather,festival_events
   2024-01-01 08:00:00,Danish,10,sunny,25,0,8
   2024-01-01 08:00:00,Muffins,20,sunny,25,0,8
   2024-01-01 08:00:00,Coffee,50,sunny,25,0,8
   2024-01-01 08:00:00,Tea,30,sunny,25,0,8
   2024-01-01 12:00:00,Danish,15,sunny,30,1,12
   ...
   ```

2. (Optional) Create a `future_regressors.csv` file with future regressor values:
   - `timestamp`: DateTime column - timestamps for future periods (4-hour intervals)
   - `weather`: Categorical (rainy/sunny/overcast)
   - `festival_events`: Binary (0/1) or numeric
   
   **Example format:**
   ```
   timestamp,weather,festival_events
   2024-01-06 08:00:00,sunny,0
   2024-01-06 12:00:00,sunny,0
   2024-01-06 16:00:00,overcast,0
   ...
   ```
   
   If this file is not provided, the script will use the last known regressor values for all future periods.

3. Update the `CSV_PATH` and `FUTURE_REGRESSORS_PATH` variables in `prophet_forecast.py` if your files have different names.

4. Run the script:
```bash
python prophet_forecast.py
```

## Output

The script generates an `outputs/` directory containing:

- **Per SKU files:**
  - `forecast_{SKU}.csv`: Forecasted values with confidence intervals for each SKU
  - `prophet_forecast_{SKU}.png`: Visualization of the forecast for each SKU
  - `prophet_components_{SKU}.png`: Decomposition of trend, seasonality, and regressor effects for each SKU

- **Combined file:**
  - `forecast_all_skus.csv`: Combined forecast for all SKUs in a single file

## Sample Data

- A sample `data.csv` file is included with products (Danish, Muffins, Coffee, Tea) for testing purposes.
- A sample `future_regressors.csv` file is included with future regressor values for 7 days (21 periods).

## How It Works

1. The script reads the CSV file in long format where each row is a product sale record
2. It automatically identifies all unique products from the `product` column
3. For each product, it:
   - Filters rows for that product
   - Uses `units_sold` as the target variable (`y`)
   - Uses the regressors (weather, festival_events)
   - If `future_regressors.csv` is provided, uses those values for future forecasts; otherwise uses last known values
   - Creates a Prophet-friendly dataset
   - Trains a separate Prophet model
   - Generates independent forecasts
   - Saves outputs with product-specific filenames
4. All forecasts are also combined into a single wide-format file for easy analysis

## Customization

You can modify:
- `FORECAST_PERIODS`: Number of future periods to forecast (default: 21, which is 7 days * 3 observations per day)
- `OUTPUT_DIR`: Directory to save outputs (default: 'outputs')
- Seasonality settings in the `create_prophet_model()` function
- Regressor handling in `load_and_prepare_data()`

