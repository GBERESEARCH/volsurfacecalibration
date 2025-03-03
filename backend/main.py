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
            
        # Remove rows with invalid prices (NaN, None, negative, etc.)
        rows_before_price_check = len(df)
        df = df.dropna(subset=['Last Price', 'Bid', 'Ask'])
        
        # Remove rows with zero or negative prices
        df = df[(df['Last Price'] > 0) & (df['Bid'] > 0) & (df['Ask'] > 0)]
        
        # Check if Ask > Bid (market sanity check)
        df = df[df['Ask'] >= df['Bid']]
        
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
            try:
                # Convert reference date to datetime with timezone info
                reference_date = pd.to_datetime(data.reference_date).tz_localize('UTC')
                
                # Ensure Last Trade Date has timezone info for comparison
                if df['Last Trade Date'].dt.tz is None:
                    df['Last Trade Date'] = df['Last Trade Date'].dt.tz_localize('UTC')
                
                # Now filter with normalized timestamps
                df = df[df['Last Trade Date'] <= reference_date]
                
                if len(df) == 0:
                    raise HTTPException(status_code=400, 
                        detail=f"No data found for reference date '{data.reference_date}'")
            except TypeError as e:
                # Handle mixed timezone issue
                print(f"Timezone comparison error: {str(e)}")
                
                # Try converting both to naive timestamps
                reference_date = pd.to_datetime(data.reference_date).tz_localize(None)
                df['Last Trade Date'] = df['Last Trade Date'].dt.tz_localize(None)
                
                df = df[df['Last Trade Date'] <= reference_date]
                if len(df) == 0:
                    raise HTTPException(status_code=400, 
                        detail=f"No data found for reference date '{data.reference_date}'")
        
        return df
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV data: {str(e)}")

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
            plotly_json = calibrator.to_plotly_json(surface_data)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error generating Plotly visualization: {str(e)}"
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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)