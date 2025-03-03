import pandas as pd
import numpy as np
import json
from scipy.stats import norm
from scipy.optimize import newton
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

class VolSurfaceCalibrator:
    def __init__(self, options_df, price_type='mid', moneyness_grid=None, time_grid=None):
        """
        Initialize the volatility surface calibrator
        
        Parameters:
        -----------
        options_df : pandas.DataFrame
            DataFrame containing options data with columns:
            ['expiry', 'strike', 'call_bid', 'call_ask', 'call_last', 'put_bid', 'put_ask', 'put_last', 'option_type']
            'option_type' should be 'call' or 'put'
        price_type : str
            The price type to use ('bid', 'ask', 'mid', 'last')
        moneyness_grid : list or np.array
            Grid points for moneyness axis (K/F)
        time_grid : list or np.array
            Grid points for time axis (in years)
        """
        self.options_df = options_df
        self.price_type = price_type
        
        # Default grids if none provided
        if moneyness_grid is None:
            self.moneyness_grid = np.linspace(0.7, 1.3, 25)
        else:
            self.moneyness_grid = moneyness_grid
            
        if time_grid is None:
            max_time = (pd.to_datetime(options_df['expiry']).max() - datetime.now()).days / 365.0
            self.time_grid = np.linspace(1/365, max_time, 25)
        else:
            self.time_grid = time_grid
            
        # Prepare columns for price selection
        self._prepare_price_columns()
    
    def _prepare_price_columns(self):
        """Prepare price columns based on the CSV format"""
        # Check what price columns are available
        available_price_columns = set()
        for col in ['Bid', 'Ask', 'Mid', 'Last Price']:
            if col in self.options_df.columns:
                available_price_columns.add(col)
        
        # Map price type to column names
        price_column_map = {
            'bid': 'Bid',
            'ask': 'Ask',
            'mid': 'Mid',
            'last': 'Last Price'
        }
        
        # Get the preferred column based on price_type
        preferred_column = price_column_map.get(self.price_type.lower())
        
        # First try to use the preferred price type if available
        if preferred_column in available_price_columns:
            self.options_df['option_price'] = self.options_df[preferred_column]
            missing_mask = self.options_df['option_price'].isna()
            missing_count = missing_mask.sum()
            
            if missing_count > 0:
                print(f"{missing_count} rows have missing {preferred_column} values, looking for alternatives")
                
                # For missing values, try to use other price columns as fallbacks
                for alt_col in ['Mid', 'Last Price', 'Bid', 'Ask']:
                    if alt_col in available_price_columns and alt_col != preferred_column:
                        # Fill missing values from this alternative column
                        still_missing = missing_mask & self.options_df[alt_col].notna()
                        if still_missing.any():
                            print(f"Filling {still_missing.sum()} missing values with {alt_col}")
                            self.options_df.loc[still_missing, 'option_price'] = self.options_df.loc[still_missing, alt_col]
                            missing_mask = self.options_df['option_price'].isna()
                            if not missing_mask.any():
                                break
        else:
            # If preferred column is not available, use the first available price column
            if available_price_columns:
                fallback_col = list(available_price_columns)[0]
                print(f"Preferred price type '{self.price_type}' not available. Using '{fallback_col}' instead.")
                self.options_df['option_price'] = self.options_df[fallback_col]
            else:
                raise ValueError("No price columns found in the data. Cannot calculate implied volatilities.")
        
        # Remove any rows that still have missing prices
        initial_rows = len(self.options_df)
        self.options_df = self.options_df.dropna(subset=['option_price'])
        rows_dropped = initial_rows - len(self.options_df)
        
        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} rows with no available price data after trying all alternatives")
            
        if len(self.options_df) == 0:
            raise ValueError(f"No valid price data found for price type '{self.price_type}' or alternatives")
            
        # Final check: ensure all prices are positive
        self.options_df = self.options_df[self.options_df['option_price'] > 0]

    
    def calculate_forwards(self, spot, discount_rates, repo_rates, dividends=None):
        """
        Calculate forward prices for each expiry
        
        Parameters:
        -----------
        spot : float
            Current spot price
        discount_rates : dict
            Dictionary mapping expiry dates to discount rates
        repo_rates : dict
            Dictionary mapping expiry dates to repo rates
        dividends : dict or None
            Dictionary mapping dividend dates to dividend amounts
            
        Returns:
        --------
        dict
            Dictionary mapping expiry dates to forward prices
        """
        # Convert expiries to datetime if they are strings
        if isinstance(next(iter(discount_rates.keys())), str):
            discount_rates = {pd.to_datetime(k): v for k, v in discount_rates.items()}
        if isinstance(next(iter(repo_rates.keys())), str):
            repo_rates = {pd.to_datetime(k): v for k, v in repo_rates.items()}
            
        forwards = {}
        unique_expiries = pd.to_datetime(self.options_df['expiry'].unique())
        
        for expiry in unique_expiries:
            # Find the nearest discount and repo rates
            discount_date = min(discount_rates.keys(), key=lambda x: abs((x - expiry).total_seconds()))
            repo_date = min(repo_rates.keys(), key=lambda x: abs((x - expiry).total_seconds()))
            
            discount_rate = discount_rates[discount_date]
            repo_rate = repo_rates[repo_date]
            
            # Calculate time to expiry in years
            t = (expiry - datetime.now()).total_seconds() / (365.25 * 24 * 60 * 60)
            
            # Apply continuous compounding
            forward = spot * np.exp((repo_rate - discount_rate) * t)
            
            # Apply dividends if provided
            if dividends:
                for div_date, div_amount in dividends.items():
                    div_date = pd.to_datetime(div_date) if isinstance(div_date, str) else div_date
                    if div_date <= expiry:
                        # Discount the dividend to present value and then compound to the expiry
                        t_div = (div_date - datetime.now()).total_seconds() / (365.25 * 24 * 60 * 60)
                        pv_div = div_amount * np.exp(-discount_rate * t_div)
                        forward -= pv_div * np.exp(discount_rate * t)
            
            forwards[expiry] = forward
            
        return forwards
    
    def black_scholes_price(self, S, K, T, r, q, sigma, option_type):
        """
        Calculate Black-Scholes option price
        
        Parameters:
        -----------
        S : float
            Forward price
        K : float
            Strike price
        T : float
            Time to expiry in years
        r : float
            Risk-free rate
        q : float
            Dividend/repo rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float
            Option price
        """
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    def implied_vol_objective(self, sigma, price, S, K, T, r, q, option_type):
        """
        Objective function for implied volatility calculation
        
        Parameters:
        -----------
        sigma : float
            Volatility to solve for
        price : float
            Market price
        Other parameters are the same as in black_scholes_price
            
        Returns:
        --------
        float
            Difference between model price and market price
        """
        return self.black_scholes_price(S, K, T, r, q, sigma, option_type) - price
    
    def calculate_implied_vols(self, spot, discount_rates, repo_rates, dividends=None):
        """
        Calculate implied volatilities for all options
        
        Parameters:
        -----------
        Same as calculate_forwards method
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with implied volatilities added
        """
        # Store spot price for use in visualization
        self.spot_price = spot
        
        # Calculate forwards
        forwards = self.calculate_forwards(spot, discount_rates, repo_rates, dividends)
        
        # Convert expiries to datetime if needed
        if isinstance(next(iter(discount_rates.keys())), str):
            discount_rates = {pd.to_datetime(k): v for k, v in discount_rates.items()}
        
        # Create standardized column names (lowercase)
        column_mapping = {
            'Expiry': 'expiry',
            'Strike': 'strike',
            'Option Type': 'option_type'
        }
        
        # Create lowercase versions of required columns if they don't exist
        # This ensures backward compatibility and consistent column naming
        for original_col, lowercase_col in column_mapping.items():
            if original_col in self.options_df.columns and lowercase_col not in self.options_df.columns:
                self.options_df[lowercase_col] = self.options_df[original_col]
        
        # Ensure option_type is lowercase for consistency
        if 'option_type' in self.options_df.columns:
            self.options_df['option_type'] = self.options_df['option_type'].str.lower()
        
        # Add columns for analysis
        # Use consistent lowercase column names for calculations
        self.options_df['forward'] = self.options_df['expiry'].map(lambda x: forwards[pd.to_datetime(x)])
        self.options_df['moneyness'] = self.options_df['strike'] / self.options_df['forward']
        self.options_df['time_to_expiry'] = (pd.to_datetime(self.options_df['expiry']) - datetime.now()).dt.total_seconds() / (365.25 * 24 * 60 * 60)
        
        # Add discount rate column by finding the nearest date
        self.options_df['discount_rate'] = self.options_df['expiry'].apply(
            lambda x: discount_rates[min(discount_rates.keys(), key=lambda k: abs((k - pd.to_datetime(x)).total_seconds()))]
        )
        
        # Calculate implied volatilities
        implied_vols = []
        
        for idx, row in self.options_df.iterrows():
            price = row['option_price']
            option_type = row['option_type'].lower()
            
            try:
                implied_vol = newton(
                    self.implied_vol_objective, 
                    x0=0.2,  # Initial guess
                    args=(price, row['forward'], row['strike'], row['time_to_expiry'], 
                          row['discount_rate'], 0, option_type)
                )
                # Sanity check the output
                if 0.001 <= implied_vol <= 2.0:
                    implied_vols.append(implied_vol)
                else:
                    implied_vols.append(np.nan)
            except:
                implied_vols.append(np.nan)
        
        self.options_df['implied_vol'] = implied_vols
        
        # Remove rows with invalid implied vols
        self.options_df = self.options_df.dropna(subset=['implied_vol'])
        
        return self.options_df
    
    def fit_surface(self, method='rbf'):
        """
        Fit the volatility surface
        
        Parameters:
        -----------
        method : str
            Method to use for surface fitting ('rbf', 'svi', etc.)
            
        Returns:
        --------
        dict
            Fitted volatility surface parameters
        """
        from scipy.interpolate import Rbf, griddata
        
        # Extract data for fitting
        moneyness = self.options_df['moneyness'].values
        time_to_expiry = self.options_df['time_to_expiry'].values
        implied_vol = self.options_df['implied_vol'].values
        
        # Check if we have enough data points
        if len(moneyness) < 4:
            raise ValueError(f"Not enough valid data points for surface fitting. Found only {len(moneyness)} points.")
        
        try:
            # Handle different fitting methods
            if method == 'svi':
                # Use SVI parameterization
                try:
                    # Try to import locally first (for development)
                    import sys
                    import os
                    # Add current directory to path if running as main script
                    if __name__ == "__main__":
                        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                    
                    # Import SVI functions
                    try:
                        from svi_implementation import (
                            fit_svi_surface, 
                            svi_to_surface_data
                        )
                    except ImportError:
                        # If still can't import, check if svi_implementation.py exists in current dir
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        svi_file = os.path.join(current_dir, "svi_implementation.py")
                        if os.path.exists(svi_file):
                            raise ImportError(f"SVI implementation file exists at {svi_file} but cannot be imported.")
                        else:
                            raise ImportError(f"SVI implementation file not found in {current_dir}.")
                except ImportError as e:
                    print(f"Error importing SVI implementation: {str(e)}")
                    print("Falling back to RBF interpolation.")
                    method = 'rbf'  # Fallback to RBF
                
                if method == 'svi':  # Only proceed if import was successful
                    # Group data by expiry time
                    unique_times = np.unique(time_to_expiry)
                    unique_times.sort()  # Ensure times are in ascending order
                    
                    if len(unique_times) < 2:
                        raise ValueError(f"SVI parameterization requires at least 2 expiry times. Found {len(unique_times)}.")
                    
                    # Create time grid based on unique expiry times
                    time_grid = np.array(unique_times)
                    
                    # Create moneyness grid
                    if self.moneyness_grid is None:
                        min_moneyness = max(0.5, min(moneyness) * 0.9)
                        max_moneyness = min(2.0, max(moneyness) * 1.1)
                        self.moneyness_grid = np.linspace(min_moneyness, max_moneyness, 25)
                    moneyness_grid = np.array(self.moneyness_grid)
                    
                    # Create a vol matrix for fitting
                    interp_vol_matrix = np.zeros((len(time_grid), len(moneyness_grid)))
                    
                    # For each time slice, create an RBF interpolation to fill the moneyness grid
                    for i, t in enumerate(time_grid):
                        mask = np.isclose(time_to_expiry, t)
                        if np.sum(mask) < 3:
                            # Not enough points in this slice, use nearby points
                            # Find closest time with enough points
                            times_with_enough = [j for j, time in enumerate(time_grid) 
                                                if np.sum(np.isclose(time_to_expiry, time)) >= 3]
                            if not times_with_enough:
                                raise ValueError("Not enough data points in any time slice for SVI fitting.")
                            closest_time_idx = min(times_with_enough, key=lambda j: abs(time_grid[j] - t))
                            t_mask = np.isclose(time_to_expiry, time_grid[closest_time_idx])
                            t_moneyness = moneyness[t_mask]
                            t_vols = implied_vol[t_mask]
                        else:
                            # Use points from this time slice
                            t_moneyness = moneyness[mask]
                            t_vols = implied_vol[mask]
                        
                        # Create RBF interpolation for this slice
                        rbf = Rbf(t_moneyness, t_vols, function='linear')
                        interp_vol_matrix[i] = rbf(moneyness_grid)
                    
                    # Fit SVI surface to the interpolated volatility matrix
                    svi_params = fit_svi_surface(moneyness_grid, time_grid, interp_vol_matrix)
                    
                    # Convert SVI parameters to surface data
                    surface_data = svi_to_surface_data(moneyness_grid, time_grid, svi_params)
                    
                    # Add raw points for visualization
                    surface_data['raw_points'] = {
                        'moneyness': moneyness.tolist(),
                        'time_to_expiry': time_to_expiry.tolist(),
                        'implied_vol': implied_vol.tolist()
                    }
                    
                    # Add SVI parameters to surface data for reference
                    surface_data['svi_params'] = [params.tolist() for params in svi_params]
                    
                    return surface_data
            
            # If method is not SVI or SVI import failed, proceed with RBF
            if method == 'rbf':
                # Use Radial Basis Function interpolation
                rbf_functions = ['thin_plate', 'multiquadric', 'gaussian', 'linear']
                
                for func in rbf_functions:
                    try:
                        # Try with regularization
                        rbf = Rbf(moneyness, time_to_expiry, implied_vol, function=func, epsilon=1.0)
                        # Test if the RBF works on a simple grid
                        test_x = [min(moneyness), max(moneyness)]
                        test_y = [min(time_to_expiry), max(time_to_expiry)]
                        rbf(test_x, test_y)
                        # If we get here, the RBF works
                        break
                    except Exception as e:
                        print(f"RBF with {func} failed: {str(e)}")
                        if func == rbf_functions[-1]:
                            # If we've tried all RBF functions, use grid interpolation instead
                            method = 'grid'
                
                if method == 'rbf':
                    # Create grid for surface
                    X, Y = np.meshgrid(self.moneyness_grid, self.time_grid)
                    Z = rbf(X, Y)
                else:
                    # Fall back to grid interpolation
                    points = np.column_stack((moneyness, time_to_expiry))
                    X, Y = np.meshgrid(self.moneyness_grid, self.time_grid)
                    grid_points = np.vstack([X.ravel(), Y.ravel()]).T
                    Z = griddata(points, implied_vol, grid_points, method='cubic', fill_value=0.2)
                    Z = Z.reshape(X.shape)
            else:
                # Fall back to grid interpolation
                points = np.column_stack((moneyness, time_to_expiry))
                X, Y = np.meshgrid(self.moneyness_grid, self.time_grid)
                grid_points = np.vstack([X.ravel(), Y.ravel()]).T
                Z = griddata(points, implied_vol, grid_points, method='cubic', fill_value=0.2)
                Z = Z.reshape(X.shape)
            
            # Ensure positive volatilities and apply reasonable bounds
            Z = np.maximum(Z, 0.001)
            Z = np.minimum(Z, 2.0)
            
            # Check for arbitrage (this is a simple check, more sophisticated methods exist)
            # Ensure volatility smile has convexity
            for i in range(len(self.time_grid)):
                # Apply a simple smoothing if needed
                if np.any(np.diff(np.diff(Z[i])) < 0):
                    from scipy.signal import savgol_filter
                    # Check if we have enough points for savgol
                    if len(self.moneyness_grid) >= 5:
                        Z[i] = savgol_filter(Z[i], min(9, len(self.moneyness_grid)), 3)
                    else:
                        # Simple smoothing for small number of points
                        Z[i] = np.convolve(Z[i], np.ones(3)/3, mode='same')
            
            # Check for calendar arbitrage
            for j in range(len(self.moneyness_grid)):
                total_var = Z[:, j]**2 * self.time_grid
                if np.any(np.diff(total_var) < 0):
                    # Adjust to ensure total variance is increasing
                    for i in range(1, len(self.time_grid)):
                        min_var = total_var[i-1]
                        if total_var[i] < min_var:
                            Z[i, j] = np.sqrt(min_var / self.time_grid[i])
            
            # Create output data structure
            surface_data = {
                'x': self.time_grid.tolist(),
                'y': self.moneyness_grid.tolist(),
                'z': Z.tolist(),
                'raw_points': {
                    'moneyness': moneyness.tolist(),
                    'time_to_expiry': time_to_expiry.tolist(),
                    'implied_vol': implied_vol.tolist()
                }
            }
            
            return surface_data
            
        except Exception as e:
            # If RBF or grid methods fail, create a simple average surface
            print(f"Surface fitting failed with error: {str(e)}. Using simple averaging method.")
            
            # Group by expiry and calculate average implied vol for each expiry
            unique_expiries = np.sort(np.unique(time_to_expiry))
            avg_vols = []
            
            for expiry in unique_expiries:
                mask = np.isclose(time_to_expiry, expiry)
                avg_vols.append(np.mean(implied_vol[mask]))
            
            # Create a flat surface based on average vols at each expiry
            X, Y = np.meshgrid(self.moneyness_grid, self.time_grid)
            Z = np.zeros_like(X)
            
            # Fill Z with interpolated values based on time_to_expiry
            for i, t in enumerate(self.time_grid):
                # Find nearest expiry
                nearest_idx = np.argmin(np.abs(unique_expiries - t))
                Z[i, :] = avg_vols[nearest_idx]
            
            # Create output data structure
            surface_data = {
                'x': self.time_grid.tolist(),
                'y': self.moneyness_grid.tolist(),
                'z': Z.tolist(),
                'raw_points': {
                    'moneyness': moneyness.tolist(),
                    'time_to_expiry': time_to_expiry.tolist(),
                    'implied_vol': implied_vol.tolist()
                },
                'warning': "Using simplified surface due to numerical issues with full fitting."
            }
            
            return surface_data
    
    def to_plotly_json(self, surface_data=None):
        """
        Convert the volatility surface to Plotly-compatible JSON
        
        Parameters:
        -----------
        surface_data : dict or None
            Volatility surface data, if None, fit_surface() will be called
            
        Returns:
        --------
        str
            JSON string for Plotly
        """
        if surface_data is None:
            surface_data = self.fit_surface()
        
        # Format for Plotly
        
        # Convert time to days and reverse the axis
        time_in_days = [t * 365 for t in surface_data['x']]
        time_in_days.reverse()  # Reverse the time axis
        
        # Convert z values (volatility) to percentage
        z_values = [[vol * 100 for vol in row] for row in surface_data['z']]
        
        # Reverse each row of z to match the reversed x axis
        z_values = [row[::-1] for row in z_values]
        
        # Calculate actual strike prices using moneyness and spot price
        # We need the latest spot price from the calibration
        spot_price = getattr(self, 'spot_price', 100.0)  # Default to 100 if not set
        
        # Calculate actual strikes from moneyness grid
        strikes = [m * spot_price for m in surface_data['y']]
        
        # Raw points with converted values
        raw_points_time = [t * 365 for t in surface_data['raw_points']['time_to_expiry']]
        raw_points_strikes = [m * spot_price for m in surface_data['raw_points']['moneyness']]
        raw_points_vol = [vol * 100 for vol in surface_data['raw_points']['implied_vol']]
        
        plotly_data = {
            'data': [
                {
                    'type': 'surface',
                    'x': time_in_days,
                    'y': strikes,
                    'z': z_values,
                    'colorscale': 'Viridis',
                    'showscale': True,
                    'colorbar': {
                        'title': 'Implied Vol (%)',
                        'titleside': 'right'
                    },
                    'name': 'Vol Surface'
                },
                {
                    'type': 'scatter3d',
                    'x': raw_points_time,
                    'y': raw_points_strikes,
                    'z': raw_points_vol,
                    'mode': 'markers',
                    'marker': {
                        'size': 4,
                        'color': 'red',
                        'opacity': 0.8
                    },
                    'name': 'Market Data'
                }
            ],
            'layout': {
                'title': 'Implied Volatility Surface',
                'scene': {
                    'xaxis': {
                        'title': 'Time to Expiry (days)',
                        'autorange': 'reversed'  # Reverse the axis
                    },
                    'yaxis': {
                        'title': 'Strike Price'
                    },
                    'zaxis': {
                        'title': 'Implied Volatility (%)'
                    }
                }
            }
        }
        
        return json.dumps(plotly_data)

# Example usage
if __name__ == "__main__":
    # Mock data in the new format
    data = {
        'Expiry': ['2025-03-21', '2025-03-21', '2025-03-21', '2025-03-21', '2025-06-20', '2025-06-20'],
        'Strike': [100, 110, 120, 100, 110, 120],
        'Last Trade Date': ['2024-02-28', '2024-02-28', '2024-02-28', '2024-02-28', '2024-02-28', '2024-02-28'],
        'Last Price': [10.8, 5.3, 2.2, 1.6, 8.0, 15.8],
        'Bid': [10.5, 5.2, 2.1, 1.5, 7.8, 15.5],
        'Ask': [11.0, 5.5, 2.3, 1.8, 8.2, 16.0],
        'Option Type': ['CALL', 'CALL', 'CALL', 'PUT', 'PUT', 'PUT']
    }
    
    options_df = pd.DataFrame(data)
    
    # Setup calibrator
    calibrator = VolSurfaceCalibrator(options_df, price_type='mid')
    
    # Mock market data
    spot = 110.0
    discount_rates = {'2025-03-21': 0.05, '2025-06-20': 0.052}
    repo_rates = {'2025-03-21': 0.01, '2025-06-20': 0.012}
    dividends = {'2025-05-15': 1.5}
    
    # Calculate implied vols
    calibrator.calculate_implied_vols(spot, discount_rates, repo_rates, dividends)
    
    # Fit surface
    surface_data = calibrator.fit_surface()
    
    # Get Plotly JSON
    plotly_json = calibrator.to_plotly_json(surface_data)
    print(plotly_json)