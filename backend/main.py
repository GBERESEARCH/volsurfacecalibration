from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Annotated, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import the volatility surface calibration class
from vol_surface_calibrator import VolSurfaceCalibrator

app = FastAPI(title="Volatility Surface API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptionData(BaseModel):
    csv_data: str = Field(..., description="CSV data containing option information")
    price_type: str = Field(default="mid", description="Type of price to use (bid, ask, mid, last)")
    reference_date: Optional[str] = Field(default=None, description="Reference date for option quotes (YYYY-MM-DD)")
    fitting_method: str = Field(default="rbf", description="Method used for surface fitting (rbf, svi)")
    spot: float = Field(..., description="Current spot price of the underlying")
    discount_rates: Dict[str, float] = Field(..., description="Discount rates by date")
    repo_rates: Dict[str, float] = Field(..., description="Repo rates by date")
    dividends: Optional[Dict[str, float]] = Field(default=None, description="Dividend amounts by ex-date")
    moneyness_grid: Optional[List[float]] = Field(default=None, description="Custom moneyness grid points")
    time_grid: Optional[List[float]] = Field(default=None, description="Custom time grid points")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "csv_data": "Expiry,Strike,Last Trade Date,Last Price,Bid,Ask,Option Type\n2025-03-21,100,2024-02-28,10.80,10.50,11.00,CALL",
                    "price_type": "mid",
                    "reference_date": "2024-02-28",
                    "fitting_method": "rbf",
                    "spot": 110.0,
                    "discount_rates": {"2025-03-21": 0.05},
                    "repo_rates": {"2025-03-21": 0.01},
                    "dividends": {"2025-05-15": 1.5}
                }
            ]
        }
    }

async def validate_csv_data(data: OptionData) -> pd.DataFrame:
    """Validate and parse CSV data."""
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(data.csv_data))
        
        # Normalize column names (handle case sensitivity)
        df.columns = [col.strip() for col in df.columns]
        column_mapping = {
            'expiry': 'Expiry',
            'strike': 'Strike',
            'last trade date': 'Last Trade Date',
            'last price': 'Last Price',
            'bid': 'Bid',
            'ask': 'Ask',
            'option type': 'Option Type'
        }
        
        # Rename columns if they exist in lowercase
        for lowercase_col, proper_col in column_mapping.items():
            if lowercase_col in df.columns and proper_col not in df.columns:
                df.rename(columns={lowercase_col: proper_col}, inplace=True)
        
        # Validate required columns
        required_cols = ['Expiry', 'Strike', 'Last Trade Date', 'Last Price', 'Bid', 'Ask', 'Option Type']
        
        # Check required columns exist
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Required column '{col}' missing in CSV data")
        
        # Also create lowercase versions for compatibility with the calibrator
        reverse_mapping = {v: k for k, v in column_mapping.items()}
        for proper_col, lowercase_col in reverse_mapping.items():
            if proper_col in df.columns:
                df[lowercase_col] = df[proper_col]
        
        # Data quality checks: Convert to appropriate types
        for col in ['Last Price', 'Bid', 'Ask']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"Warning when converting {col} to numeric: {str(e)}")
        
        # Handle timestamp format in Last Trade Date
        try:
            # First, try parsing as timestamp (2025-02-18 15:19:18+00:00)
            df['Last Trade Date'] = pd.to_datetime(df['Last Trade Date'], errors='coerce')
            
            # Check for NaT (Not a Time) values, which indicate parse failures
            if df['Last Trade Date'].isna().any():
                print(f"Warning: Some Last Trade Date entries could not be parsed as timestamps")
            
            # If all values are NaT, try simpler date format
            if df['Last Trade Date'].isna().all():
                print("Attempting to parse Last Trade Date as simple date format (YYYY-MM-DD)")
                df['Last Trade Date'] = pd.to_datetime(df['Last Trade Date'], format='%Y-%m-%d', errors='coerce')
        except Exception as e:
            print(f"Error parsing Last Trade Date: {str(e)}")
            # Fallback to simple date format
            try:
                df['Last Trade Date'] = pd.to_datetime(df['Last Trade Date'], format='%Y-%m-%d', errors='coerce')
            except Exception as e2:
                print(f"Fallback date parsing also failed: {str(e2)}")
        
        # Drop rows with invalid dates
        initial_rows = len(df)
        df = df.dropna(subset=['Last Trade Date'])
        rows_dropped_dates = initial_rows - len(df)
        if rows_dropped_dates > 0:
            print(f"Dropped {rows_dropped_dates} rows with invalid Last Trade Date")
        
        # Modify the validate_csv_data function in main.py

        # Replace this section in validate_csv_data:
        # ...
        # Remove rows with invalid prices (NaN, None, negative, etc.)
        # rows_before_price_check = len(df)
        # df = df.dropna(subset=['Last Price', 'Bid', 'Ask'])
        # 
        # Remove rows with zero or negative prices
        # df = df[(df['Last Price'] > 0) & (df['Bid'] > 0) & (df['Ask'] > 0)]
        # 
        # Check if Ask > Bid (market sanity check)
        # df = df[df['Ask'] >= df['Bid']]
        # ...

        # With this improved version:
        # Improved price validation that allows options with some zero prices
        rows_before_price_check = len(df)

        # Keep track of which price type we're using
        price_column_map = {
            'bid': 'Bid',
            'ask': 'Ask',
            'mid': 'Mid',
            'last': 'Last Price'
        }
        preferred_column = price_column_map.get(data.price_type.lower(), 'Mid')

        # First, make sure the specified price type is valid
        if preferred_column not in df.columns:
            # If preferred column doesn't exist, look for alternatives
            available_price_columns = [col for col in ['Last Price', 'Bid', 'Ask', 'Mid'] if col in df.columns]
            if not available_price_columns:
                raise HTTPException(status_code=400, detail=f"No price columns found in the data.")
            preferred_column = available_price_columns[0]
            print(f"Preferred price type '{data.price_type}' not available. Using '{preferred_column}' instead.")

        # Remove rows where the preferred price column is invalid
        df = df.dropna(subset=[preferred_column])
        df = df[df[preferred_column] > 0]

        # For 'mid' price type, we need special handling
        if preferred_column == 'Mid':
            # If Mid is directly provided, use it
            if 'Mid' in df.columns:
                pass
            # Otherwise, calculate it from Bid and Ask
            elif 'Bid' in df.columns and 'Ask' in df.columns:
                # Remove rows with NaN in both Bid and Ask
                df = df.dropna(subset=['Bid', 'Ask'], how='all')
                
                # Fill NaN values with existing values or zeros
                df['Bid'] = df['Bid'].fillna(0)
                df['Ask'] = df['Ask'].fillna(0)
                
                # Apply market sanity check only when both Bid and Ask are non-zero
                valid_market = (df['Bid'] == 0) | (df['Ask'] == 0) | (df['Ask'] >= df['Bid'])
                df = df[valid_market]
                
                # Calculate Mid when possible, otherwise use the available price
                df['Mid'] = df.apply(
                    lambda row: (row['Bid'] + row['Ask']) / 2 if row['Bid'] > 0 and row['Ask'] > 0 
                    else row['Ask'] if row['Ask'] > 0 
                    else row['Bid'] if row['Bid'] > 0
                    else 0, 
                    axis=1
                )
                
                # Keep only rows with valid Mid
                df = df[df['Mid'] > 0]
            else:
                # If we can't calculate Mid, use another price
                for alt_col in ['Last Price', 'Bid', 'Ask']:
                    if alt_col in df.columns:
                        df = df.dropna(subset=[alt_col])
                        df = df[df[alt_col] > 0]
                        print(f"No Bid/Ask to calculate Mid, using {alt_col} instead.")
                        break
        else:
            # For non-Mid prices, only apply the market sanity check 
            # when both Bid and Ask are non-zero
            if 'Bid' in df.columns and 'Ask' in df.columns:
                has_both_prices = (df['Bid'] > 0) & (df['Ask'] > 0)
                valid_prices = ~has_both_prices | (df['Ask'] >= df['Bid'])
                df = df[valid_prices]

        # Report how many rows were removed
        rows_removed_prices = rows_before_price_check - len(df)
        if rows_removed_prices > 0:
            print(f"Removed {rows_removed_prices} rows with invalid price data")
        
        # Log data quality information
        rows_removed_prices = rows_before_price_check - len(df)
        if rows_removed_prices > 0:
            print(f"Removed {rows_removed_prices} rows with invalid price data")
            
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No valid option data found after filtering invalid dates and prices")
        
        # Normalize option types to call/put
        df['Option Type'] = df['Option Type'].str.upper()
        df['option_type'] = df['Option Type'].str.lower()  # Create lowercase version
        
        # Validate options types (either 'CALL' or 'PUT')
        valid_option_types = ['CALL', 'PUT', 'C', 'P']
        invalid_types = df[~df['Option Type'].str.upper().isin(valid_option_types)]['Option Type'].unique()
        if len(invalid_types) > 0:
            raise HTTPException(status_code=400, 
                detail=f"Invalid option types found: {', '.join(invalid_types)}. Must be 'CALL', 'PUT', 'C', or 'P'.")
        
        # Standardize option types
        df['Option Type'] = df['Option Type'].replace({'C': 'CALL', 'P': 'PUT'})
        df['option_type'] = df['Option Type'].str.lower()
        
        # Calculate mid price if not provided
        df['Mid'] = (df['Bid'] + df['Ask']) / 2
        
        # Filter by reference date if provided
        if data.reference_date:
            print(f"Filtering by reference date: {data.reference_date}")
            
            try:
                # Print a sample of what we're dealing with
                if len(df) > 0:
                    print(f"Sample Last Trade Date before filtering: {df['Last Trade Date'].iloc[0]}")
                
                # Convert all timestamps to datetime64[ns] without timezone info for comparison
                df['Last_Trade_Date_Normalized'] = pd.to_datetime(df['Last Trade Date']).dt.tz_localize(None).dt.normalize()
                
                # Parse reference date to datetime and normalize to date only
                reference_date = pd.to_datetime(data.reference_date).normalize()
                print(f"Normalized reference date: {reference_date}")
                
                # Filter based on date only (without timezone info)
                df = df[df['Last_Trade_Date_Normalized'] <= reference_date]
                
                # Debug output
                filtered_count = len(df)
                print(f"After reference date filtering: {filtered_count} rows remain")
                
                if len(df) == 0:
                    # Before raising an error, let's print some information for debugging
                    all_dates = pd.to_datetime(df['Last Trade Date']).dt.tz_localize(None).dt.normalize().unique()
                    print(f"Available dates in dataset: {all_dates}")
                    raise HTTPException(status_code=400, 
                        detail=f"No data found for reference date '{data.reference_date}'. Check date format and timezone.")
                        
            except Exception as e:
                print(f"Error during date filtering: {str(e)}")
                
                # As a fallback, try a simple string comparison of just the date portion
                try:
                    # Extract just the date part as string from both sides
                    ref_date_str = data.reference_date.split('T')[0]
                    df['Trade_Date_Str'] = pd.to_datetime(df['Last Trade Date']).dt.strftime('%Y-%m-%d')
                    
                    # Filter using string comparison
                    df = df[df['Trade_Date_Str'] <= ref_date_str]
                    
                    print(f"After fallback date filtering: {len(df)} rows remain")
                    
                    if len(df) == 0:
                        raise HTTPException(status_code=400, 
                            detail=f"No data found for reference date '{data.reference_date}' using string comparison")
                except Exception as e2:
                    print(f"String date comparison also failed: {str(e2)}")
                    raise HTTPException(status_code=400, 
                        detail=f"Date filtering failed. Original error: {str(e)}. Fallback error: {str(e2)}")
                                
        return df
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV data: {str(e)}")
    
# Add this class after the existing OptionData model in main.py

class OptionDiagnosticsRequest(BaseModel):
    """Model for option diagnostics request"""
    csv_data: str = Field(..., description="CSV data containing option information")
    price_type: str = Field(default="mid", description="Type of price to use (bid, ask, mid, last)")
    reference_date: Optional[str] = Field(default=None, description="Reference date for option quotes (YYYY-MM-DD)")
    spot: float = Field(..., description="Current spot price of the underlying")
    discount_rates: Dict[str, float] = Field(..., description="Discount rates by date")
    repo_rates: Dict[str, float] = Field(..., description="Repo rates by date")
    dividends: Optional[Dict[str, float]] = Field(default=None, description="Dividend amounts by ex-date")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "csv_data": "Expiry,Strike,Last Trade Date,Last Price,Bid,Ask,Option Type\n2025-03-21,100,2024-02-28,10.80,10.50,11.00,CALL",
                    "price_type": "mid",
                    "reference_date": "2024-02-28",
                    "spot": 110.0,
                    "discount_rates": {"2025-03-21": 0.05},
                    "repo_rates": {"2025-03-21": 0.01},
                    "dividends": {"2025-05-15": 1.5}
                }
            ]
        }
    }

# Add this endpoint to main.py
@app.post("/diagnostics", description="Get diagnostic information for option data")
async def get_diagnostics(data: OptionDiagnosticsRequest = Body(...)):
    """
    Analyze option data and provide diagnostics on which options are included/excluded and why.
    """
    try:
        # Parse and validate the CSV data
        from io import StringIO
        df = pd.read_csv(StringIO(data.csv_data))
        
        # Use the existing validation function
        validator_data = OptionData(
            csv_data=data.csv_data,
            price_type=data.price_type,
            reference_date=data.reference_date,
            fitting_method='rbf',  # Default, not used for diagnostics
            spot=data.spot,
            discount_rates=data.discount_rates,
            repo_rates=data.repo_rates,
            dividends=data.dividends,
            moneyness_grid=None,
            time_grid=None
        )
        
        df = await validate_csv_data(validator_data)
        
        # Create volatility surface calibrator
        calibrator = VolSurfaceCalibrator(
            df, 
            price_type=data.price_type
        )
        
        # Get diagnostics
        diagnostics = calibrator.diagnose_options(
            data.spot,
            data.discount_rates,
            data.repo_rates,
            data.dividends
        )
        
        # Calculate summary statistics
        included_count = sum(1 for d in diagnostics if d["status"] == "Included")
        total_count = len(diagnostics)
        
        # Final response
        response = {
            "diagnostics": diagnostics,
            "summary": {
                "total_options": total_count,
                "included_options": included_count,
                "excluded_options": total_count - included_count,
                "inclusion_rate": round((included_count / total_count * 100), 2) if total_count > 0 else 0
            }
        }
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # Handle ValueError (often from numerical issues)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch-all for any other errors
        raise HTTPException(status_code=500, detail=f"Diagnostics error: {str(e)}")    

@app.post("/calibrate_surface", description="Calibrate an arbitrage-free volatility surface from option data")
async def calibrate_surface(data: OptionData = Body(...), df: pd.DataFrame = Depends(validate_csv_data)):
    """Calibrate a volatility surface from option data."""
    try:        
        # Store initial data counts for quality metrics
        initial_row_count = len(df)
        
        # Create volatility surface calibrator
        calibrator = VolSurfaceCalibrator(
            df, 
            price_type=data.price_type,
            moneyness_grid=data.moneyness_grid,
            time_grid=data.time_grid
        )
        
        # Calculate implied volatilities
        try:
            vol_df = calibrator.calculate_implied_vols(
                data.spot,
                data.discount_rates,
                data.repo_rates,
                data.dividends
            )
            
            # Calculate metrics for data quality
            valid_rows = len(vol_df)
            invalid_vols = initial_row_count - valid_rows
            
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Error calculating implied volatilities: {str(e)}"
            )
        
        # Check if we have enough valid data points
        if len(vol_df) < 3:
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough valid implied volatilities calculated. Only found {len(vol_df)} valid data points."
            )
        
        # Fit the surface using the specified method
        try:
            surface_data = calibrator.fit_surface(method=data.fitting_method)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Error fitting volatility surface: {str(e)}"
            )
        
        # Convert to Plotly format
        try:
            # Check if SVI method was successfully used
            is_svi = data.fitting_method == 'svi' and 'svi_params' in surface_data
            
            if is_svi:
                # Use special SVI visualization with enhanced grid
                print("Using SVI-specific visualization")
                plotly_json = calibrator.to_plotly_json_svi(surface_data)
            else:
                # Use standard visualization for RBF and other methods
                plotly_json = calibrator.to_plotly_json(surface_data)
        except Exception as e:
            # If visualization fails, log the error and try the standard method
            print(f"Error in visualization: {str(e)}")

            try:
                # Use the enhanced visualization with better error handling and wider strike range
                plotly_json = calibrator.to_plotly_json_enhanced(surface_data)
                
                # Verify valid JSON
                json.loads(plotly_json)
            except Exception as e:
                print(f"Error generating visualization: {str(e)}")
                # Fall back to the most basic visualization possible
                try:
                    # Extract raw points
                    raw_points = surface_data.get('raw_points', {})
                    if 'moneyness' in raw_points and 'time_to_expiry' in raw_points and 'implied_vol' in raw_points:
                        moneyness = raw_points['moneyness']
                        time_to_expiry = raw_points['time_to_expiry']
                        implied_vol = raw_points['implied_vol']
                        
                        # Convert to days and percentage
                        spot_price = data.spot
                        time_days = [t * 365 for t in time_to_expiry]
                        strikes = [m * spot_price for m in moneyness]
                        vol_pct = [v * 100 for v in implied_vol]
                        
                        # Create simple scatter plot
                        fallback_data = {
                            'data': [
                                {
                                    'type': 'scatter3d',
                                    'x': time_days,
                                    'y': strikes,
                                    'z': vol_pct,
                                    'mode': 'markers',
                                    'marker': {
                                        'size': 5,
                                        'color': vol_pct,
                                        'colorscale': 'Viridis',
                                        'opacity': 0.9
                                    },
                                    'name': 'Market Data'
                                }
                            ],
                            'layout': {
                                'title': 'Implied Volatility Data (Surface Generation Failed)',
                                'scene': {
                                    'xaxis': {'title': 'Time to Expiry (days)'},
                                    'yaxis': {'title': 'Strike Price'},
                                    'zaxis': {'title': 'Implied Volatility (%)'}
                                }
                            }
                        }
                        
                        plotly_json = json.dumps(fallback_data)
                    else:
                        raise ValueError("Raw points data not available")
                except Exception as fallback_error:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to create visualization. Original error: {str(e)}. Fallback error: {str(fallback_error)}"
                    )
        
        # Add metadata to response
        response = {
            "surface_data": json.loads(plotly_json),
            "metadata": {
                "num_valid_points": valid_rows,
                "num_invalid_points": invalid_vols,
                "data_quality": {
                    "input_rows": initial_row_count,
                    "filtered_rows": initial_row_count - valid_rows,
                    "percentage_valid": round((valid_rows / initial_row_count * 100), 2) if initial_row_count > 0 else 0
                },
                "moneyness_range": [float(vol_df['moneyness'].min()), float(vol_df['moneyness'].max())],
                "time_range": [float(vol_df['time_to_expiry'].min()), float(vol_df['time_to_expiry'].max())],
                "price_type_used": data.price_type,
                "fitting_method": data.fitting_method,
                "calibration_timestamp": datetime.now().isoformat()
            }
        }
        
        # Add data coverage metrics
        calls = vol_df[vol_df['option_type'] == 'call']
        puts = vol_df[vol_df['option_type'] == 'put']
        response["metadata"]["data_coverage"] = {
            "num_calls": len(calls),
            "num_puts": len(puts),
            "num_expiries": len(vol_df['expiry'].unique()),
            "num_strikes": len(vol_df['strike'].unique())
        }
        
        # Add SVI parameters if available
        if 'svi_params' in surface_data:
            response["metadata"]["svi_params"] = surface_data['svi_params']
        
        # Add warning if present
        if 'warning' in surface_data:
            response['metadata']['warning'] = surface_data['warning']
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # Handle ValueError (often from numerical issues)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch-all for any other errors
        raise HTTPException(status_code=500, detail=f"Calibration error: {str(e)}")

# Add to your main.py file - additional logging to troubleshoot SVI issues

@app.post("/debug_svi", description="Debug SVI parameterization issues")
async def debug_svi(data: OptionData = Body(...), df: pd.DataFrame = Depends(validate_csv_data)):
    """Debug SVI parameterization issues."""
    try:
        # Store initial data counts for quality metrics
        initial_row_count = len(df)
        
        # Create volatility surface calibrator
        calibrator = VolSurfaceCalibrator(
            df, 
            price_type=data.price_type,
            moneyness_grid=data.moneyness_grid,
            time_grid=data.time_grid
        )
        
        # Calculate implied volatilities
        try:
            vol_df = calibrator.calculate_implied_vols(
                data.spot,
                data.discount_rates,
                data.repo_rates,
                data.dividends
            )
            
            # Calculate metrics for data quality
            valid_rows = len(vol_df)
            invalid_vols = initial_row_count - valid_rows
            
            debug_result = {
                "initial_count": initial_row_count,
                "valid_count": valid_rows,
                "invalid_count": invalid_vols,
                "unique_expiries": len(vol_df['expiry'].unique()),
                "points_per_expiry": {},
                "moneyness_range": [float(vol_df['moneyness'].min()), float(vol_df['moneyness'].max())],
                "implied_vol_range": [float(vol_df['implied_vol'].min()), float(vol_df['implied_vol'].max())],
                "implied_vol_stats": {
                    "mean": float(vol_df['implied_vol'].mean()),
                    "std": float(vol_df['implied_vol'].std()),
                    "median": float(vol_df['implied_vol'].median())
                }
            }
            
            # Count data points per expiry
            for expiry in vol_df['expiry'].unique():
                expiry_points = len(vol_df[vol_df['expiry'] == expiry])
                debug_result["points_per_expiry"][str(expiry)] = expiry_points
            
            # Check SVI import
            debug_result["svi_import_status"] = "Checking SVI import..."
            
            try:
                # Add current directory to path if running as main script
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                
                # Try to import SVI functions
                try:
                    from svi_implementation import (
                        fit_svi_surface, 
                        svi_to_surface_data
                    )
                    debug_result["svi_import_status"] = "SVI import successful"
                except ImportError as e:
                    debug_result["svi_import_status"] = f"SVI import failed: {str(e)}"
                    
                    # Check if the file exists
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    svi_file = os.path.join(current_dir, "svi_implementation.py")
                    if os.path.exists(svi_file):
                        debug_result["svi_file_exists"] = True
                        with open(svi_file, 'r') as f:
                            file_content = f.read()
                            debug_result["svi_file_preview"] = file_content[:500] + "..." if len(file_content) > 500 else file_content
                    else:
                        debug_result["svi_file_exists"] = False
                        debug_result["current_directory"] = current_dir
            except Exception as e:
                debug_result["svi_import_status"] = f"Error checking SVI import: {str(e)}"
            
            # Add more debug info about SVI fitting conditions
            debug_result["svi_conditions"] = {
                "enough_unique_expiries": len(vol_df['expiry'].unique()) >= 2,
                "enough_points_per_expiry": all(count >= 3 for count in debug_result["points_per_expiry"].values()),
                "method_requested": data.fitting_method == "svi"
            }
            
            # If SVI is available, try a test fit with minimal data
            if debug_result["svi_import_status"] == "SVI import successful":
                try:
                    from svi_implementation import fit_svi_slice
                    
                    # Create a simple test case
                    test_log_moneyness = np.linspace(-0.2, 0.2, 5)
                    test_total_var = np.array([0.04, 0.035, 0.03, 0.035, 0.04]) # Simple smile pattern
                    
                    # Try to fit
                    test_result = fit_svi_slice(test_log_moneyness, test_total_var)
                    debug_result["svi_test_fit"] = "Successful"
                    debug_result["svi_test_params"] = test_result.tolist()
                except Exception as e:
                    debug_result["svi_test_fit"] = f"Failed: {str(e)}"
            
            return debug_result
            
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Error calculating implied volatilities: {str(e)}"
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch-all for any other errors
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)