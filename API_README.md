# Prophet Forecasting REST API

REST API for running Prophet forecasting models on product sales data.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the API

Start the Flask server:

```bash
python prophet_api.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### 1. POST /forecast

Upload data file and trigger forecasting.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Form fields:
  - `file` (required): CSV file with sales data
  - `future_regressors` (optional): CSV file with future regressor values

**Example using curl:**
```bash
curl -X POST http://localhost:5000/forecast \
  -F "file=@data.csv" \
  -F "future_regressors=@future_regressors.csv"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": {
    "mode": "per_product",
    "results": {
      "danish": {
        "meta": {
          "rows_in": 84,
          "rows_out": 21,
          "regressors": ["weather", "festival_events"]
        },
        "forecast": [
          {
            "ds": "2024-01-29T08:00:00",
            "yhat": 19.5,
            "yhat_lower": 15.2,
            "yhat_upper": 23.8
          }
        ]
      }
    }
  }
}
```

### 2. GET /forecast/<job_id>

Retrieve forecast results for a specific job.

**Request:**
- Method: `GET`
- URL: `http://localhost:5000/forecast/<job_id>`

**Example using curl:**
```bash
curl http://localhost:5000/forecast/550e8400-e29b-41d4-a716-446655440000
```

**Response:**
Same format as the result in POST response.

### 3. GET /forecast

List all forecast jobs.

**Request:**
- Method: `GET`
- URL: `http://localhost:5000/forecast`

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "created_at": "2024-01-28T10:30:00"
    }
  ]
}
```

### 4. GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## Input File Format

### Data File (required)

CSV file with the following columns:
- `timestamp`: DateTime column (4-hour intervals: 8AM, 12PM, 4PM)
- `product`: Product name
- `units_sold`: Number of units sold
- `weather`: Categorical (rainy/sunny/overcast)
- `festival_events`: Binary (0/1) or numeric

Example:
```csv
timestamp,product,units_sold,weather,temperature,festival_events,time_of_day
2024-01-01 08:00:00,Danish,14,rainy,30,0,8
2024-01-01 08:00:00,Muffins,23,overcast,24,0,8
```

### Future Regressors File (optional)

CSV file with future regressor values:
- `timestamp`: DateTime column
- `weather`: Categorical (rainy/sunny/overcast)
- `festival_events`: Binary (0/1) or numeric

Example:
```csv
timestamp,weather,festival_events
2024-01-29 08:00:00,overcast,0
2024-01-29 12:00:00,rainy,0
```

## Output Format

The API returns forecasts in the following format:

```json
{
  "mode": "per_product",
  "results": {
    "product_name": {
      "meta": {
        "rows_in": 84,
        "rows_out": 21,
        "regressors": ["weather", "festival_events"]
      },
      "input": [
        {
          "timestamp": "2024-01-01T08:00:00",
          "units_sold": 14,
          "weather": "rainy",
          "temperature": 30,
          "festival_events": 0
        }
      ],
      "forecast": [
        {
          "ds": "2024-01-29T08:00:00",
          "yhat": 19.5,
          "yhat_lower": 15.2,
          "yhat_upper": 23.8
        }
      ]
    }
  }
}
```

Where:
- `mode`: Always "per_product" for product-level forecasts
- `results`: Dictionary keyed by product name (lowercase, spaces replaced with underscores)
- `meta.rows_in`: Number of input data points
- `meta.rows_out`: Number of forecast periods
- `meta.regressors`: List of regressor names used
- `input`: Array of input data objects with:
  - `timestamp`: Timestamp in ISO format
  - `units_sold`: Number of units sold
  - `weather`: Weather condition (rainy/sunny/overcast)
  - `temperature`: Temperature value
  - `festival_events`: Festival event indicator (0/1)
- `forecast`: Array of forecast objects with:
  - `ds`: Timestamp in ISO format
  - `yhat`: Forecasted value
  - `yhat_lower`: Lower bound of confidence interval
  - `yhat_upper`: Upper bound of confidence interval

## Error Handling

If an error occurs, the API returns:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "error",
  "error": "Error message here"
}
```

## Notes

- Results are stored in memory and will be lost when the server restarts
- For production use, consider using a database for persistent storage
- File uploads are stored in a temporary directory
- Maximum file size is 16MB

