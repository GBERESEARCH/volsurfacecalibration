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
    
    # Add this at the beginning of the fit_surface method:
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
        
        # Print detailed diagnostic info
        print(f"\nFitting surface with method: {method}")
        print(f"Data points available: {len(moneyness)}")
        unique_expiries = np.unique(time_to_expiry)
        print(f"Unique expiries: {len(unique_expiries)}")
        for t in unique_expiries:
            mask = np.isclose(time_to_expiry, t)
            print(f"  Expiry {t:.6f} years: {np.sum(mask)} data points")
        
        try:
            # Handle different fitting methods
            if method == 'svi':
                # Use SVI parameterization
                print("\nAttempting SVI parameterization...")
                
                try:
                    # Try to import locally first (for development)
                    import sys
                    import os
                    # Add current directory to path if running as main script
                    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                    
                    # Import SVI functions
                    try:
                        from svi_implementation import (
                            fit_svi_surface, 
                            svi_to_surface_data,
                            fit_svi_surface_direct    # Make sure this is imported too
                        )
                        print("SVI implementation successfully imported")
                    except ImportError as e:
                        # If still can't import, check if svi_implementation.py exists in current dir
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        svi_file = os.path.join(current_dir, "svi_implementation.py")
                        if os.path.exists(svi_file):
                            raise ImportError(f"SVI implementation file exists at {svi_file} but cannot be imported: {str(e)}")
                        else:
                            raise ImportError(f"SVI implementation file not found in {current_dir}: {str(e)}")
                        
                except ImportError as e:
                    print(f"Error importing SVI implementation: {str(e)}")
                    print("Falling back to RBF interpolation.")
                    method = 'rbf'  # Fallback to RBF
                
                if method == 'svi':  # Only proceed if import was successful
                    # First get the SVI parameters
                    try:
                        # Standard SVI fitting with interpolation
                        # Group data by expiry time
                        unique_times = np.unique(time_to_expiry)
                        unique_times.sort()  # Ensure times are in ascending order
                        
                        print(f"Found {len(unique_times)} unique expiry times for SVI fitting")
                        
                        if len(unique_times) < 2:
                            print(f"SVI parameterization requires at least 2 expiry times, but found {len(unique_times)}.")
                            print("Falling back to RBF interpolation.")
                            method = 'rbf'
                        else:
                            # Continue with SVI implementation
                            # Create time grid based on unique expiry times
                            time_grid = np.array(unique_times)
                            
                            # Create moneyness grid
                            if self.moneyness_grid is None:
                                min_moneyness = max(0.5, min(moneyness) * 0.9)
                                max_moneyness = min(2.0, max(moneyness) * 1.1)
                                print(f"Creating moneyness grid from {min_moneyness} to {max_moneyness}")
                                self.moneyness_grid = np.linspace(min_moneyness, max_moneyness, 25)
                            moneyness_grid = np.array(self.moneyness_grid)
                            
                            # For each time slice, interpolate to the moneyness grid
                            interp_vol_matrix = np.zeros((len(time_grid), len(moneyness_grid)))
                            
                            # Check if we have enough points per slice
                            enough_points_per_slice = True
                            for i, t in enumerate(time_grid):
                                mask = np.isclose(time_to_expiry, t)
                                points_in_slice = np.sum(mask)
                                print(f"Time slice {t:.6f} years: {points_in_slice} points")
                                
                                if points_in_slice < 3:
                                    print(f"Not enough points for time slice {t:.6f} years. Need at least 3, found {points_in_slice}.")
                                    enough_points_per_slice = False
                            
                            if not enough_points_per_slice:
                                print("At least one time slice doesn't have enough points for SVI fitting.")
                                print("Trying direct SVI fitting without interpolation...")
                                
                                try:
                                    # Direct SVI fitting (bypass the problematic interpolation)
                                    svi_params = fit_svi_surface_direct(moneyness, time_to_expiry, implied_vol)
                                    print("Successfully fit SVI surface directly!")
                                    
                                    # Store SVI parameters for reference
                                    svi_params_list = []
                                    for params in svi_params:
                                        # Handle different potential types of params
                                        if hasattr(params, 'tolist'):
                                            svi_params_list.append(params.tolist())
                                        elif isinstance(params, tuple):
                                            svi_params_list.append(list(params))
                                        else:
                                            svi_params_list.append(list(params))
                                    
                                except Exception as e:
                                    print(f"Direct SVI fitting failed: {str(e)}")
                                    print("Falling back to RBF interpolation.")
                                    method = 'rbf'
                                    # Exit the SVI section
                                    
                            else:
                                # Interpolate each time slice
                                for i, t in enumerate(time_grid):
                                    mask = np.isclose(time_to_expiry, t)
                                    t_moneyness = moneyness[mask]
                                    t_vols = implied_vol[mask]
                                    
                                    print(f"Fitting time slice {t:.6f} years with {len(t_moneyness)} points")
                                    
                                    # Use robust interpolation for this slice
                                    try:
                                        interp_vol_matrix[i] = self.robust_slice_interpolation(t_moneyness, t_vols, moneyness_grid)
                                        print(f"Successfully interpolated slice {t:.6f}")
                                    except Exception as e:
                                        print(f"Error interpolating slice {t:.6f}: {str(e)}")
                                        raise ValueError(f"Cannot interpolate time slice {t:.6f} for SVI fitting: {str(e)}")
                                
                                # Fit SVI surface to the interpolated volatility matrix
                                try:
                                    svi_params = fit_svi_surface(moneyness_grid, time_grid, interp_vol_matrix)
                                    print("Successfully fit SVI surface!")
                                    
                                    # Store SVI parameters for reference
                                    svi_params_list = []
                                    for params in svi_params:
                                        # Handle different potential types of params
                                        if hasattr(params, 'tolist'):
                                            svi_params_list.append(params.tolist())
                                        elif isinstance(params, tuple):
                                            svi_params_list.append(list(params))
                                        else:
                                            svi_params_list.append(list(params))
                                    
                                except Exception as e:
                                    print(f"SVI fitting failed: {str(e)}")
                                    print("Falling back to RBF interpolation.")
                                    method = 'rbf'
                                    # Exit the SVI section
                    except Exception as e:
                        print(f"SVI preparation failed: {str(e)}")
                        print("Falling back to RBF interpolation.")
                        method = 'rbf'
                        # Exit the SVI section
                    
                    # If we're still in the SVI method, generate the surface data
                    if method == 'svi':
                        try:
                            # Generate surface data
                            surface_data = svi_to_surface_data(moneyness_grid, time_grid, svi_params)
                            
                            # Add SVI parameters to surface data
                            surface_data['svi_params'] = svi_params_list
                            
                            # Add raw points for visualization
                            surface_data['raw_points'] = {
                                'moneyness': moneyness.tolist(),
                                'time_to_expiry': time_to_expiry.tolist(),
                                'implied_vol': implied_vol.tolist()
                            }
                            
                            # Print summary of the surface
                            print("\nSVI Surface Summary:")
                            print(f"Time range: {min(time_grid):.4f} to {max(time_grid):.4f} years")
                            print(f"Moneyness range: {min(moneyness_grid):.4f} to {max(moneyness_grid):.4f}")
                            print(f"Surface grid: {len(surface_data['y'])} x {len(surface_data['x'])} points")
                            print(f"Min volatility: {min([min(row) for row in surface_data['z']]):.4f}")
                            print(f"Max volatility: {max([max(row) for row in surface_data['z']]):.4f}")
                            
                            return surface_data
                            
                        except Exception as e:
                            print(f"SVI surface data generation failed: {str(e)}")
                            print("Falling back to RBF interpolation.")
                            method = 'rbf'    

        except Exception as e:
            print(f"Standard SVI fitting failed: {str(e)}")
            print("Trying direct SVI fitting without interpolation...")
            
            try:
                # Import direct fitting function
                from svi_implementation import fit_svi_surface_direct, svi_to_surface_data
                
                # Fit SVI directly to scattered data
                svi_params = fit_svi_surface_direct(moneyness, time_to_expiry, implied_vol)
                print("Successfully fit SVI surface directly!")
                
                # Create moneyness grid if not provided
                if self.moneyness_grid is None:
                    min_moneyness = max(0.5, min(moneyness) * 0.9)
                    max_moneyness = min(2.0, max(moneyness) * 1.1)
                    self.moneyness_grid = np.linspace(min_moneyness, max_moneyness, 25)
                
                # Create time grid based on unique expiry times
                time_grid = np.array(np.unique(time_to_expiry))
                time_grid.sort()
                    
                # Convert SVI parameters to surface data
                surface_data = svi_to_surface_data(self.moneyness_grid, time_grid, svi_params)
                
                # Add raw points for visualization
                surface_data['raw_points'] = {
                    'moneyness': moneyness.tolist(),
                    'time_to_expiry': time_to_expiry.tolist(),
                    'implied_vol': implied_vol.tolist()
                }
                
                # Add SVI parameters to surface data for reference
                surface_data['svi_params'] = []
                for params in svi_params:
                    # Handle different potential types of params
                    if hasattr(params, 'tolist'):
                        surface_data['svi_params'].append(params.tolist())
                    elif isinstance(params, tuple):
                        surface_data['svi_params'].append(list(params))
                    else:
                        surface_data['svi_params'].append(list(params))
                
                return surface_data
                
            except Exception as e:
                print(f"Direct SVI fitting also failed: {str(e)}")
                print("Falling back to RBF interpolation.")
                method = 'rbf'
            
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
    
    # Add this method to the VolSurfaceCalibrator class in vol_surface_calibrator.py

    def diagnose_options(self, spot, discount_rates, repo_rates, dividends=None):
        """
        Analyze options to identify which are included/excluded from calibration and why
        
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
        list
            List of diagnostic records for each option
        """
        # Save original DataFrame before any filtering
        original_df = self.options_df.copy()
        
        # Store spot price for use in visualization
        self.spot_price = spot
        
        # Create standardized column names (lowercase)
        column_mapping = {
            'Expiry': 'expiry',
            'Strike': 'strike',
            'Option Type': 'option_type'
        }
        
        # Create lowercase versions of required columns if they don't exist
        for original_col, lowercase_col in column_mapping.items():
            if original_col in original_df.columns and lowercase_col not in original_df.columns:
                original_df[lowercase_col] = original_df[original_col]
        
        # Ensure option_type is lowercase for consistency
        if 'option_type' in original_df.columns:
            original_df['option_type'] = original_df['option_type'].str.lower()
        
        # Calculate forwards for diagnostics
        forwards = self.calculate_forwards(spot, discount_rates, repo_rates, dividends)
        
        # Convert expiries to datetime if needed
        if isinstance(next(iter(discount_rates.keys())), str):
            discount_rates = {pd.to_datetime(k): v for k, v in discount_rates.items()}
        
        # Initialize diagnostic records
        diagnostics = []
        
        # Process each option for diagnostics
        for idx, row in original_df.iterrows():
            diagnostic = {
                "strike": float(row['strike']),
                "expiry": row['expiry'],
                "option_type": row['option_type'].lower(),
                "price": float(row['option_price']) if 'option_price' in row else float(row.get('Mid', row.get('Bid', 0))),
                "status": "Excluded",
                "implied_vol": None,
                "reason": ""
            }
            
            # Add Black-Scholes inputs for reference
            try:
                # Calculate forward price for this expiry
                expiry_datetime = pd.to_datetime(row['expiry'])
                forward = forwards.get(expiry_datetime)
                
                if forward is None:
                    diagnostic["reason"] = "Failed to calculate forward price"
                    diagnostics.append(diagnostic)
                    continue
                    
                # Calculate time to expiry
                time_to_expiry = (expiry_datetime - datetime.now()).total_seconds() / (365.25 * 24 * 60 * 60)
                
                if time_to_expiry <= 0:
                    diagnostic["reason"] = "Option is expired"
                    diagnostics.append(diagnostic)
                    continue
                    
                # Find discount rate for this expiry
                discount_date = min(discount_rates.keys(), key=lambda x: abs((x - expiry_datetime).total_seconds()))
                discount_rate = discount_rates[discount_date]
                
                # Check option price validity
                option_price = diagnostic["price"]
                if option_price <= 0:
                    diagnostic["reason"] = "Invalid price (≤ 0)"
                    diagnostics.append(diagnostic)
                    continue
                    
                # Check strike price validity
                strike_price = diagnostic["strike"]
                if strike_price <= 0:
                    diagnostic["reason"] = "Invalid strike (≤ 0)"
                    diagnostics.append(diagnostic)
                    continue
                    
                # Check moneyness is reasonable
                moneyness = strike_price / forward
                if moneyness < 0.1 or moneyness > 10:
                    diagnostic["reason"] = f"Extreme moneyness ({moneyness:.2f})"
                    diagnostics.append(diagnostic)
                    continue
                    
                # Check option type
                option_type = diagnostic["option_type"]
                if option_type not in ['call', 'put']:
                    diagnostic["reason"] = f"Invalid option type: {option_type}"
                    diagnostics.append(diagnostic)
                    continue
                    
                # Check for arbitrage violations
                if option_type == 'call':
                    intrinsic = max(0, spot - strike_price)
                    if option_price < intrinsic:
                        diagnostic["reason"] = f"Call price below intrinsic value: {option_price} < {intrinsic}"
                        diagnostics.append(diagnostic)
                        continue
                else:  # put
                    intrinsic = max(0, strike_price - spot)
                    if option_price < intrinsic:
                        diagnostic["reason"] = f"Put price below intrinsic value: {option_price} < {intrinsic}"
                        diagnostics.append(diagnostic)
                        continue
                        
                # Try to calculate implied volatility
                try:
                    implied_vol = newton(
                        self.implied_vol_objective, 
                        x0=0.2,  # Initial guess
                        args=(option_price, forward, strike_price, time_to_expiry, 
                            discount_rate, 0, option_type)
                    )
                    
                    # Sanity check the implied vol
                    if 0.001 <= implied_vol <= 2.0:
                        diagnostic["implied_vol"] = float(implied_vol)
                        diagnostic["status"] = "Included"
                    else:
                        diagnostic["implied_vol"] = float(implied_vol)
                        diagnostic["reason"] = f"Unreasonable volatility value: {implied_vol:.2%}"
                except Exception as e:
                    diagnostic["reason"] = f"IV calculation failed: {str(e)}"
                    
                diagnostics.append(diagnostic)
                
            except Exception as e:
                diagnostic["reason"] = f"Error in diagnostic process: {str(e)}"
                diagnostics.append(diagnostic)
        
        return diagnostics


    def robust_slice_interpolation(self, x_data, y_data, x_grid):
        """
        Perform robust interpolation of a volatility slice, with multiple fallback methods.
        
        Parameters:
        -----------
        x_data : array-like
            Moneyness values from market data
        y_data : array-like
            Implied volatility values from market data
        x_grid : array-like
            Target moneyness grid for interpolation
            
        Returns:
        --------
        array-like
            Interpolated volatility values for the target grid
        """
        from scipy.interpolate import Rbf, interp1d, UnivariateSpline, griddata
        import numpy as np
        
        # Try multiple interpolation methods with error handling
        methods = [
            # Method 1: RBF with different function types
            lambda: Rbf(x_data, y_data, function='linear')(x_grid),
            lambda: Rbf(x_data, y_data, function='multiquadric', epsilon=1.0)(x_grid),
            lambda: Rbf(x_data, y_data, function='gaussian', epsilon=2.0)(x_grid),
            
            # Method 2: Linear interpolation with extrapolation
            lambda: interp1d(x_data, y_data, bounds_error=False, fill_value='extrapolate')(x_grid),
            
            # Method 3: Cubic spline with smoothing
            lambda: UnivariateSpline(x_data, y_data, k=3, s=0.1)(x_grid),
            
            # Method 4: Grid interpolation
            lambda: griddata(x_data.reshape(-1, 1), y_data, x_grid.reshape(-1, 1), method='linear').flatten(),
            lambda: griddata(x_data.reshape(-1, 1), y_data, x_grid.reshape(-1, 1), method='cubic').flatten(),
            
            # Method 5: Polynomial fit
            lambda: np.polyval(np.polyfit(x_data, y_data, min(3, len(x_data)-1)), x_grid)
        ]
        
        # Try each method until one works
        last_error = None
        for i, method in enumerate(methods):
            try:
                result = method()
                
                # Check if the result contains NaN or inf values
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    raise ValueError(f"Method {i+1} produced NaN or inf values")
                    
                # Check for unreasonable volatility values
                if np.any(result < 0.001) or np.any(result > 2.0):
                    print(f"Warning: Method {i+1} produced values outside reasonable volatility range. Clamping.")
                    result = np.clip(result, 0.001, 2.0)
                    
                print(f"Successfully interpolated using method {i+1}")
                return result
            except Exception as e:
                last_error = e
                print(f"Method {i+1} failed: {str(e)}")
        
        # If all methods fail, use nearest neighbor as last resort
        print("All interpolation methods failed. Using nearest neighbor as last resort.")
        result = np.zeros_like(x_grid)
        for i, x in enumerate(x_grid):
            # Find the nearest data point
            nearest_idx = np.argmin(np.abs(x_data - x))
            result[i] = y_data[nearest_idx]
        
        return result

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
                    'colorscale': 'Blues',
                    'opacity': 0.7, 
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
                        'size': 2,
                        'color': 'red',
                        'opacity': 0.9
                    },
                    'name': 'Market Data'
                }
            ],
            'layout': {
                'title': 'Implied Volatility Surface',
                'scene': {
                    'xaxis': {
                        'title': 'Time to Expiry (days)',
                        'autorange': 'reversed',  # Reverse the axis
                        'backgroundcolor': "rgb(200, 200, 230)",
                        'gridcolor': "white",
                        'showbackground': True,
                        'zerolinecolor': "white"
                    },
                    'yaxis': {
                        'title': 'Strike Price',
                        'backgroundcolor': "rgb(230, 200, 230)",
                        'gridcolor': "white",
                        'showbackground': True,
                        'zerolinecolor': "white"
                    },
                    'zaxis': {
                        'title': 'Implied Volatility (%)',
                        'backgroundcolor': "rgb(230, 230, 200)",
                        'gridcolor': "white",
                        'showbackground': True,
                        'zerolinecolor': "white"
                    },
                    'aspectmode': 'cube'
                }
            }
        }
        
        return json.dumps(plotly_data)
    
    # Add this function to your vol_surface_calibrator.py to better handle SVI visualization

    def to_plotly_json_svi(self, surface_data=None):
        """
        Special version of to_plotly_json specifically for SVI surfaces
        that ensures proper grid coverage and visualization
        
        Parameters:
        -----------
        surface_data : dict or None
            Volatility surface data from SVI fitting
            
        Returns:
        --------
        str
            JSON string for Plotly
        """
        if surface_data is None:
            raise ValueError("surface_data is required for SVI visualization")
        
        # Extract key data
        time_grid = np.array(surface_data['x'])
        moneyness_grid = np.array(surface_data['y'])
        vol_matrix = np.array(surface_data['z'])
        raw_points = surface_data['raw_points']
        
        # Create a wider & denser moneyness grid to properly show the surface
        # This is especially important for SVI which has good extrapolation properties
        min_moneyness = min(0.7, min(moneyness_grid) * 0.9)
        max_moneyness = max(1.3, max(moneyness_grid) * 1.1)
        dense_moneyness_grid = np.linspace(min_moneyness, max_moneyness, 100)
        
        # Print diagnostics
        print(f"Original moneyness grid: {min(moneyness_grid)} to {max(moneyness_grid)}")
        print(f"Enhanced moneyness grid: {min_moneyness} to {max_moneyness}")
        
        # Create a denser time grid too
        min_time = min(time_grid)
        max_time = max(time_grid)
        dense_time_grid = np.linspace(min_time, max_time, 100)
        
        # Make new grids
        X, Y = np.meshgrid(dense_time_grid, dense_moneyness_grid)
        
        # Get SVI parameters if available
        svi_params = surface_data.get('svi_params', None)
        
        if svi_params:
            # If we have SVI parameters, regenerate the surface with denser grid
            print("Regenerating surface with SVI parameters and denser grid")
            try:
                from svi_implementation import svi_surface_to_vol_matrix
                
                # Process SVI parameters to ensure correct format
                processed_params = []
                for params in svi_params:
                    if isinstance(params, list):
                        processed_params.append(np.array(params))
                    else:
                        processed_params.append(params)
                
                # Interpolate SVI parameters for the dense time grid
                from scipy.interpolate import interp1d
                
                # For each SVI parameter, create an interpolation function across time
                param_interp_funcs = []
                for i in range(5):  # 5 SVI parameters (a, b, rho, m, sigma)
                    param_values = [p[i] for p in processed_params]
                    
                    # Create interpolation function (with constant extrapolation)
                    interp_func = interp1d(
                        time_grid, param_values, 
                        kind='linear', 
                        bounds_error=False,
                        fill_value=(param_values[0], param_values[-1])
                    )
                    param_interp_funcs.append(interp_func)
                
                # Generate dense SVI parameters
                dense_svi_params = []
                for t in dense_time_grid:
                    params = [f(t) for f in param_interp_funcs]
                    dense_svi_params.append(np.array(params))
                
                # Generate volatility surface with dense grid
                log_moneyness_grid = np.log(dense_moneyness_grid)
                vol_matrix = svi_surface_to_vol_matrix(log_moneyness_grid, dense_time_grid, dense_svi_params)
                
                # Safety check for NaN or Inf values
                vol_matrix = np.nan_to_num(vol_matrix, nan=0.2, posinf=2.0, neginf=0.001)
                vol_matrix = np.clip(vol_matrix, 0.001, 2.0)
                
                # Update grid dimensions
                X, Y = np.meshgrid(dense_time_grid, dense_moneyness_grid)
                
            except Exception as e:
                print(f"Error regenerating SVI surface: {str(e)}")
                print("Proceeding with original grid")
        
        # Format for Plotly - similar to the original to_plotly_json method
        
        # Convert time to days and reverse the axis
        time_in_days = [t * 365 for t in dense_time_grid]
        time_in_days.reverse()  # Reverse the time axis
        
        # Convert z values (volatility) to percentage
        z_values = [[vol * 100 for vol in row] for row in vol_matrix.T]  # Note the transpose here
        
        # Reverse each row of z to match the reversed x axis
        z_values = [row[::-1] for row in z_values]
        
        # Calculate actual strike prices using moneyness and spot price
        spot_price = getattr(self, 'spot_price', 100.0)  # Default to 100 if not set
        
        # Calculate actual strikes from moneyness grid
        strikes = [m * spot_price for m in dense_moneyness_grid]
        
        # Raw points with converted values
        raw_points_time = [t * 365 for t in raw_points['time_to_expiry']]
        raw_points_strikes = [m * spot_price for m in raw_points['moneyness']]
        raw_points_vol = [vol * 100 for vol in raw_points['implied_vol']]
        
        plotly_data = {
            'data': [
                {
                    'type': 'surface',
                    'x': time_in_days,
                    'y': strikes,
                    'z': z_values,
                    'colorscale': 'Blues',  # Changed from Viridis to Blues
                    'opacity': 0.7,         # Added opacity
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
                        'size': 2,
                        'color': 'red',
                        'opacity': 0.9
                    },
                    'name': 'Market Data'
                }
            ],
            'layout': {
                'title': 'SVI Implied Volatility Surface',
                'scene': {
                    'xaxis': {
                        'title': 'Time to Expiry (days)',
                        'autorange': 'reversed',  # Reverse the axis
                        'backgroundcolor': "rgb(200, 200, 230)",
                        'gridcolor': "white",
                        'showbackground': True,
                        'zerolinecolor': "white"
                    },
                    'yaxis': {
                        'title': 'Strike Price',
                        'backgroundcolor': "rgb(230, 200, 230)",
                        'gridcolor': "white",
                        'showbackground': True,
                        'zerolinecolor': "white"
                    },
                    'zaxis': {
                        'title': 'Implied Volatility (%)',
                        'backgroundcolor': "rgb(230, 230, 200)",
                        'gridcolor': "white",
                        'showbackground': True,
                        'zerolinecolor': "white"
                    },
                    'aspectmode': 'cube'
                }
            }
        }
        
        return json.dumps(plotly_data)
    
    # Let's create a simpler, more reliable approach to the SVI visualization
    # Add this method to vol_surface_calibrator.py

    def to_plotly_json_enhanced(self, surface_data=None):
        """
        Enhanced version of to_plotly_json with improved error handling
        and better handling of strike price range
        
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
            raise ValueError("surface_data is required")
        
        import numpy as np
        import json
        from scipy.interpolate import griddata
        
        try:
            # Extract data
            time_grid = np.array(surface_data['x'])
            moneyness_grid = np.array(surface_data['y'])
            vol_matrix = np.array(surface_data['z'])
            raw_points = surface_data['raw_points']
            
            # Create a wider moneyness grid to show more strikes
            spot_price = getattr(self, 'spot_price', 100.0)
            min_moneyness = 0.5  # Show strikes down to 50% of spot
            max_moneyness = 2.0  # Show strikes up to 200% of spot
            
            # Create denser grids for better visualization
            dense_moneyness_grid = np.linspace(min_moneyness, max_moneyness, 100)
            dense_time_grid = np.linspace(min(time_grid), max(time_grid), 100)
            
            # Convert to actual strike prices
            dense_strikes = dense_moneyness_grid * spot_price
            
            # Create new coordinate grid
            X, Y = np.meshgrid(dense_time_grid, dense_moneyness_grid)
            
            # Interpolate the volatility surface to the new grid
            # First create a grid of the original points
            orig_X, orig_Y = np.meshgrid(time_grid, moneyness_grid)
            
            # Reshape for griddata
            points = np.column_stack([orig_X.flatten(), orig_Y.flatten()])
            values = vol_matrix.flatten()
            
            # Create target grid
            xi = np.column_stack([X.flatten(), Y.flatten()])
            
            # Interpolate
            interpolated = griddata(points, values, xi, method='linear')
            
            # Handle NaN values if any
            mask = np.isnan(interpolated)
            if np.any(mask):
                interpolated[mask] = griddata(points, values, xi[mask], method='nearest')
            
            # Reshape to grid
            Z = interpolated.reshape(X.shape)
            
            # Clip to reasonable volatility range
            Z = np.clip(Z, 0.001, 2.0)
            
            # Format for Plotly
            
            # Convert time to days and reverse the axis for Plotly
            time_in_days = [t * 365 for t in dense_time_grid]
            time_in_days.reverse()  # Reverse for visualization
            
            # Convert volatility to percentage
            dense_z_values = [[vol * 100 for vol in row] for row in Z.T]  # Transpose for Plotly
            
            # Reverse each row of z to match the reversed x axis
            dense_z_values = [row[::-1] for row in dense_z_values]
            
            # Raw points with converted values
            raw_points_time = [t * 365 for t in raw_points['time_to_expiry']]
            raw_points_strikes = [m * spot_price for m in raw_points['moneyness']]
            raw_points_vol = [vol * 100 for vol in raw_points['implied_vol']]
            
            # Create plot data
            plotly_data = {
                'data': [
                    {
                        'type': 'surface',
                        'x': time_in_days,
                        'y': dense_strikes.tolist(),  # Actual strike prices
                        'z': dense_z_values,
                        'colorscale': 'Blues',  # Changed from Viridis to Blues
                        'opacity': 0.7,         # Added opacity
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
                            'size': 2,           # Changed from 4/5 to 2
                            'color': 'red',      
                            'opacity': 0.9       # Changed from 0.8 to 0.9
                        },
                        'name': 'Market Data'
                    }
                ],
                'layout': {
                    'title': 'Implied Volatility Surface',
                    'scene': {
                        'xaxis': {
                            'title': 'Time to Expiry (days)',
                            'autorange': 'reversed',  # Reverse the axis
                            'backgroundcolor': "rgb(200, 200, 230)",
                            'gridcolor': "white",
                            'showbackground': True,
                            'zerolinecolor': "white"
                        },
                        'yaxis': {
                            'title': 'Strike Price',
                            'backgroundcolor': "rgb(230, 200, 230)",
                            'gridcolor': "white",
                            'showbackground': True,
                            'zerolinecolor': "white"
                        },
                        'zaxis': {
                            'title': 'Implied Volatility (%)',
                            'backgroundcolor': "rgb(230, 230, 200)",
                            'gridcolor': "white",
                            'showbackground': True,
                            'zerolinecolor': "white"
                        },
                        'aspectmode': 'cube'
                    }
                }
            }
            
            return json.dumps(plotly_data)
            
        except Exception as e:
            # If all else fails, create a simple scatter plot of the raw data
            try:
                # Raw points
                raw_points = surface_data['raw_points']
                raw_points_time = [t * 365 for t in raw_points['time_to_expiry']]
                raw_points_strikes = [m * getattr(self, 'spot_price', 100.0) for m in raw_points['moneyness']]
                raw_points_vol = [vol * 100 for vol in raw_points['implied_vol']]
                
                # Simple scatter plot
                fallback_data = {
                    'data': [
                        {
                            'type': 'scatter3d',
                            'x': raw_points_time,
                            'y': raw_points_strikes,
                            'z': raw_points_vol,
                            'mode': 'markers',
                            'marker': {
                                'size': 5,
                                'color': raw_points_vol,
                                'colorscale': 'Viridis',
                                'opacity': 0.9,
                                'showscale': True,
                                'colorbar': {
                                    'title': 'Implied Vol (%)',
                                    'titleside': 'right'
                                }
                            },
                            'name': 'Market Data'
                        }
                    ],
                    'layout': {
                        'title': 'Implied Volatility Data (Surface Generation Failed)',
                        'scene': {
                            'xaxis': {
                                'title': 'Time to Expiry (days)',
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
                
                return json.dumps(fallback_data)
                
            except Exception as fallback_error:
                # Really failed - return error message
                raise ValueError(f"Failed to create visualization: {str(e)}. Fallback error: {str(fallback_error)}")    


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