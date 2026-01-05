"""
REST API for Prophet Forecasting
Provides endpoints to upload data and retrieve forecast results.
"""

# Set matplotlib backend before any other imports that might use it
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web server

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import os
import json
import uuid
from datetime import datetime
import tempfile
import shutil

# Import forecasting functions
from prophet_forecast import (
    load_and_prepare_data,
    load_future_regressors,
    process_product,
    create_wide_format_forecast
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# In-memory storage for results (in production, use a database)
forecast_results = {}

# Configuration
FORECAST_PERIODS = 21  # 7 days * 3 observations per day
OUTPUT_DIR = 'outputs'


def run_forecast(data_file_path, future_regressors_path=None):
    """
    Run forecasting pipeline and return results in API format.
    
    Args:
        data_file_path: Path to input CSV file
        future_regressors_path: Optional path to future regressors CSV
    
    Returns:
        Dictionary with forecast results in API format
    """
    try:
        # Load and prepare data
        df, product_names = load_and_prepare_data(data_file_path)
        
        # Load future regressors if provided
        future_regressors = None
        if future_regressors_path and os.path.exists(future_regressors_path):
            future_regressors = load_future_regressors(future_regressors_path)
        
        # Process each product
        results = {}
        
        for product_name in sorted(product_names):
            # Extract input data for this product before processing
            product_input_df = df[df['product'] == product_name].copy()
            product_input_df = product_input_df.sort_values('ds').reset_index(drop=True)
            
            # Format input data
            input_data = []
            for _, row in product_input_df.iterrows():
                input_data.append({
                    'timestamp': row['ds'].isoformat() if pd.notna(row['ds']) else None,
                    'units_sold': float(row['units_sold']) if pd.notna(row['units_sold']) else None,
                    'weather': row.get('weather', None),
                    'festival_events': int(row['festival_events']) if pd.notna(row.get('festival_events', None)) else None
                })
            
            # Process product (skip plots and CSV in API context)
            result = process_product(df, product_name, FORECAST_PERIODS, OUTPUT_DIR, future_regressors, skip_plots=True, skip_csv=True)
            
            if result:
                # Get regressor names used - these are the regressors we're using
                # Map internal names to user-friendly names
                regressor_mapping = {
                    'weather_encoded': 'weather',
                    'festival_events': 'festival_events'
                }
                # Check which regressors are actually in the forecast
                forecast_cols = set(result['forecast'].columns)
                regressor_cols = []
                for internal_name, display_name in regressor_mapping.items():
                    if internal_name in forecast_cols:
                        regressor_cols.append(display_name)
                
                # Get only future forecast (not historical fit)
                product_df = result.get('product_df', None)
                last_timestamp = None
                if product_df is not None and len(product_df) > 0:
                    last_timestamp = product_df['ds'].max()
                
                # Format forecast data - only include future forecasts
                forecast_data = []
                for _, row in result['forecast'].iterrows():
                    forecast_ds = row['ds']
                    # Skip if this is a historical timestamp
                    if last_timestamp is not None and pd.notna(forecast_ds):
                        if pd.to_datetime(forecast_ds) <= last_timestamp:
                            continue
                    
                    forecast_data.append({
                        'ds': forecast_ds.isoformat() if pd.notna(forecast_ds) else None,
                        'yhat': float(row['yhat']) if pd.notna(row['yhat']) else None,
                        'yhat_lower': float(row['yhat_lower']) if pd.notna(row['yhat_lower']) else None,
                        'yhat_upper': float(row['yhat_upper']) if pd.notna(row['yhat_upper']) else None
                    })
                
                # Store in results
                product_key = product_name.lower().replace(' ', '_')
                results[product_key] = {
                    'input': input_data,
                    'meta': {
                        'rows_in': result['data_points'],
                        'rows_out': len(forecast_data),
                        'regressors': regressor_cols
                    },
                    'forecast': forecast_data
                }
        
        return {
            'mode': 'per_product',
            'results': results
        }
    
    except Exception as e:
        raise Exception(f"Forecasting error: {str(e)}")


@app.route('/forecast', methods=['POST'])
def upload_and_forecast():
    """
    POST endpoint to upload data file and trigger forecasting.
    
    Expected form data:
    - file: CSV file with sales data
    - future_regressors (optional): CSV file with future regressor values
    
    Returns:
        JSON with job_id and status
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        data_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{job_id}_data.csv')
        file.save(data_file_path)
        
        # Handle optional future regressors file
        future_regressors_path = None
        if 'future_regressors' in request.files:
            future_regressors_file = request.files['future_regressors']
            if future_regressors_file.filename != '':
                future_regressors_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], 
                    f'{job_id}_future_regressors.csv'
                )
                future_regressors_file.save(future_regressors_path)
        
        # Run forecast
        try:
            forecast_result = run_forecast(data_file_path, future_regressors_path)
            
            # Store results
            forecast_results[job_id] = {
                'status': 'completed',
                'created_at': datetime.now().isoformat(),
                'result': forecast_result
            }
            
            return jsonify({
                'job_id': job_id,
                'status': 'completed',
                'result': forecast_result
            }), 200
        
        except Exception as e:
            forecast_results[job_id] = {
                'status': 'error',
                'created_at': datetime.now().isoformat(),
                'error': str(e)
            }
            return jsonify({
                'job_id': job_id,
                'status': 'error',
                'error': str(e)
            }), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/forecast/<job_id>', methods=['GET'])
def get_forecast(job_id):
    """
    GET endpoint to retrieve forecast results.
    
    Args:
        job_id: Unique job identifier from POST response
    
    Returns:
        JSON with forecast results in specified format
    """
    if job_id not in forecast_results:
        return jsonify({'error': 'Job ID not found'}), 404
    
    result = forecast_results[job_id]
    
    if result['status'] == 'error':
        return jsonify({
            'status': 'error',
            'error': result.get('error', 'Unknown error')
        }), 500
    
    return jsonify(result['result']), 200


@app.route('/forecast', methods=['GET'])
def list_forecasts():
    """
    GET endpoint to list all forecast jobs.
    
    Returns:
        JSON with list of job IDs and their statuses
    """
    jobs = []
    for job_id, result in forecast_results.items():
        jobs.append({
            'job_id': job_id,
            'status': result['status'],
            'created_at': result['created_at']
        })
    
    return jsonify({'jobs': jobs}), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=8000)

