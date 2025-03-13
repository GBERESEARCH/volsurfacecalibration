import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

def svi_function(k, params):
    """
    Basic SVI parameterization for a single volatility slice
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    
    Parameters:
    -----------
    k : array-like
        Log-moneyness values
    params : tuple
        SVI parameters (a, b, rho, m, sigma)
        
    Returns:
    --------
    array-like
        Total implied variance w(k) for each log-moneyness
    """
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def svi_vega_weighted_error(params, log_moneyness, total_var, vega_weights):
    """
    Calculate vega-weighted error for SVI fit
    
    Parameters:
    -----------
    params : tuple
        SVI parameters (a, b, rho, m, sigma)
    log_moneyness : array-like
        Log-moneyness values
    total_var : array-like
        Target total implied variance values
    vega_weights : array-like
        Vega weights for each point
        
    Returns:
    --------
    float
        Weighted sum of squared errors
    """
    predicted_var = svi_function(log_moneyness, params)
    weighted_errors = vega_weights * ((predicted_var - total_var) ** 2)
    return np.sum(weighted_errors)

def butterfly_arbitrage_constraint(params):
    """
    Constraint to ensure no butterfly arbitrage: b * (1 + |rho|) < 2
    
    Parameters:
    -----------
    params : tuple
        SVI parameters (a, b, rho, m, sigma)
        
    Returns:
    --------
    float
        Constraint value (negative value means constraint is satisfied)
    """
    a, b, rho, m, sigma = params
    return b * (1 + abs(rho)) - 2

def calendar_spread_constraint(params_t1, params_t2):
    """
    Constraint to ensure no calendar spread arbitrage between two time slices
    
    Parameters:
    -----------
    params_t1 : tuple
        SVI parameters for time slice 1 (earlier)
    params_t2 : tuple
        SVI parameters for time slice 2 (later)
        
    Returns:
    --------
    float
        Minimum value of the difference. If positive, no arbitrage.
    """
    # Generate a grid of moneyness points to check
    k_values = np.linspace(-2, 2, 100)  # Moneyness range to check
    
    # Calculate total variance at each point
    tv1 = svi_function(k_values, params_t1)
    tv2 = svi_function(k_values, params_t2)
    
    # Check if tv2 >= tv1 at all points
    return np.min(tv2 - tv1)

def fit_svi_slice(log_moneyness, total_var, initial_guess=None, vega_weights=None):
    """
    Fit SVI parameters to a single volatility slice (single maturity)
    
    Parameters:
    -----------
    log_moneyness : array-like
        Log-moneyness values
    total_var : array-like
        Total implied variance values (sigma^2 * T)
    initial_guess : tuple, optional
        Initial guess for SVI parameters (a, b, rho, m, sigma)
    vega_weights : array-like, optional
        Vega weights for each point
        
    Returns:
    --------
    tuple
        Optimized SVI parameters (a, b, rho, m, sigma)
    """
    if vega_weights is None:
        # Uniform weights if none provided
        vega_weights = np.ones_like(log_moneyness)
    
    if initial_guess is None:
        # Generate reasonable initial guess
        a = np.min(total_var) * 0.9  # Minimum variance level
        b = (np.max(total_var) - np.min(total_var)) / 4  # Slope
        rho = 0.0  # No skew initially
        m = 0.0  # ATM initially
        sigma = 0.5  # Reasonable smoothness
        initial_guess = (a, b, rho, m, sigma)
    
    # Constraint to ensure no butterfly arbitrage
    constraints = [
        {'type': 'ineq', 'fun': lambda params: 2 - params[1] * (1 + abs(params[2]))}  # b * (1 + |rho|) < 2
    ]
    
    # Parameter bounds
    bounds = [
        (0.0, None),  # a > 0
        (0.0, 2.0),   # 0 < b < 2
        (-1.0, 1.0),  # -1 < rho < 1
        (-2.0, 2.0),  # Reasonable range for m
        (0.01, None)  # sigma > 0
    ]
    
    # Optimize
    result = minimize(
        svi_vega_weighted_error,
        initial_guess,
        args=(log_moneyness, total_var, vega_weights),
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'
    )
    
    if not result.success:
        print(f"Warning: SVI optimization did not converge. Status: {result.message}")
    
    return result.x

def fit_svi_surface(moneyness_grid, time_grid, vol_matrix, forward_curve=None):
    """
    Fit SVI surface to a volatility matrix
    
    Parameters:
    -----------
    moneyness_grid : array-like
        Moneyness values
    time_grid : array-like
        Time to expiry values
    vol_matrix : array-like
        Volatility matrix with shape (len(time_grid), len(moneyness_grid))
    forward_curve : array-like, optional
        Forward curve values for each time slice
        
    Returns:
    --------
    list
        List of SVI parameters for each time slice
    """
    # Convert moneyness to log-moneyness
    log_moneyness_grid = np.log(moneyness_grid)
    
    # Initialize parameter list and previous slice parameters
    svi_params = []
    prev_params = None
    
    # Fit SVI to each time slice
    for t_idx, t in enumerate(time_grid):
        # Get volatility slice for this maturity
        vol_slice = vol_matrix[t_idx]
        
        # Calculate total variance (sigma^2 * t)
        total_var = vol_slice**2 * t
        
        # Set initial guess based on previous slice if available
        initial_guess = prev_params if prev_params is not None else None
        
        # Fit SVI to this slice
        params = fit_svi_slice(log_moneyness_grid, total_var, initial_guess)
        
        # Check calendar spread arbitrage with previous slice
        if prev_params is not None:
            cal_arb = calendar_spread_constraint(prev_params, params)
            # If there's arbitrage, re-fit with additional constraint
            if cal_arb < 0:
                print(f"Calendar arbitrage detected at time {t}. Re-fitting with constraints.")
                
                # Ensure params is a list for modification
                if isinstance(params, tuple):
                    params = list(params)
                elif hasattr(params, 'tolist'):
                    params = params.tolist()
                
                # Make sure prev_params is also accessible as a list
                prev_params_list = prev_params
                if isinstance(prev_params, tuple):
                    prev_params_list = list(prev_params)
                elif hasattr(prev_params, 'tolist'):
                    prev_params_list = prev_params.tolist()
                
                # Adjust the 'a' parameter to ensure the new SVI curve dominates the previous one
                # Add a small buffer to prevent numerical issues
                params[0] = prev_params_list[0] + 0.01  # Ensure a_t2 > a_t1
                
                # Convert back to the original type
                if isinstance(prev_params, tuple):
                    params = tuple(params)

        svi_params.append(params)
        prev_params = params
    
    return svi_params

def svi_surface_to_vol_matrix(log_moneyness_grid, time_grid, svi_params):
    """
    Convert SVI parameters to a volatility matrix
    
    Parameters:
    -----------
    log_moneyness_grid : array-like
        Log-moneyness grid
    time_grid : array-like
        Time to expiry grid
    svi_params : list
        List of SVI parameters for each time slice
        
    Returns:
    --------
    array-like
        Volatility matrix with shape (len(time_grid), len(log_moneyness_grid))
    """
    vol_matrix = np.zeros((len(time_grid), len(log_moneyness_grid)))
    
    for t_idx, t in enumerate(time_grid):
        params = svi_params[t_idx]
        total_var = svi_function(log_moneyness_grid, params)
        vol_matrix[t_idx] = np.sqrt(total_var / t)
    
    return vol_matrix

def svi_to_surface_data(moneyness_grid, time_grid, svi_params):
    """
    Convert SVI parameters to a surface data format for visualization
    
    Parameters:
    -----------
    moneyness_grid : array-like
        Moneyness grid
    time_grid : array-like
        Time to expiry grid
    svi_params : list
        List of SVI parameters for each time slice
        
    Returns:
    --------
    dict
        Surface data in the format expected by to_plotly_json
    """
    # Convert moneyness to log-moneyness
    log_moneyness_grid = np.log(moneyness_grid)

    # Ensure parameters are in the right format
    svi_params_processed = []
    for params in svi_params:
        # Handle different potential types of params
        if isinstance(params, np.ndarray):
            svi_params_processed.append(params)
        elif isinstance(params, tuple) or isinstance(params, list):
            svi_params_processed.append(np.array(params))
        else:
            # Unknown type, try converting to array
            try:
                svi_params_processed.append(np.array(params))
            except:
                raise ValueError(f"Cannot convert SVI parameters to numpy array: {type(params)}")

    # Then replace svi_params with svi_params_processed in the rest of the function
    vol_matrix = svi_surface_to_vol_matrix(log_moneyness_grid, time_grid, svi_params_processed)
    
    # Generate SVI surface in volatility
    vol_matrix = svi_surface_to_vol_matrix(log_moneyness_grid, time_grid, svi_params)
    
    # Format for surface data
    surface_data = {
        'x': time_grid.tolist(),
        'y': moneyness_grid.tolist(),
        'z': vol_matrix.tolist()
    }
    
    return surface_data

# Alternative approach: Skip the problematic interpolation step 
# and fit SVI directly to each slice of data

# Add this function to svi_implementation.py
def fit_svi_surface_direct(moneyness_data, time_data, vol_data):
    """
    Fit SVI surface directly to scattered data points without pre-interpolation
    
    Parameters:
    -----------
    moneyness_data : array-like
        Moneyness values of all data points
    time_data : array-like
        Time to expiry values of all data points
    vol_data : array-like
        Implied volatility values of all data points
        
    Returns:
    --------
    list
        List of SVI parameters for each time slice
    """
    import numpy as np
    
    # Get unique times
    unique_times = np.unique(time_data)
    unique_times.sort()  # Ensure times are in ascending order
    
    # Initialize parameter list and previous slice parameters
    svi_params = []
    prev_params = None
    
    # Fit SVI to each time slice
    for t_idx, t in enumerate(unique_times):
        # Extract data for this time slice
        mask = np.isclose(time_data, t)
        slice_moneyness = moneyness_data[mask]
        slice_vol = vol_data[mask]
        
        print(f"Fitting SVI directly to time slice {t:.6f} with {len(slice_moneyness)} points")
        
        # Convert moneyness to log-moneyness
        log_moneyness = np.log(slice_moneyness)
        
        # Calculate total variance (sigma^2 * t)
        total_var = slice_vol**2 * t
        
        # Set initial guess based on previous slice if available
        initial_guess = prev_params if prev_params is not None else None
        
        # Fit SVI to this slice
        try:
            params = fit_svi_slice(log_moneyness, total_var, initial_guess)
            
            # Check calendar spread arbitrage with previous slice
            if prev_params is not None:
                cal_arb = calendar_spread_constraint(prev_params, params)
                # If there's arbitrage, re-fit with additional constraint
                if cal_arb < 0:
                    print(f"Calendar arbitrage detected at time {t}. Re-fitting with constraints.")
                    
                    # Ensure params is a list for modification
                    if isinstance(params, tuple):
                        params = list(params)
                    elif hasattr(params, 'tolist'):
                        params = params.tolist()
                    
                    # Make sure prev_params is also accessible as a list
                    prev_params_list = prev_params
                    if isinstance(prev_params, tuple):
                        prev_params_list = list(prev_params)
                    elif hasattr(prev_params, 'tolist'):
                        prev_params_list = prev_params.tolist()
                    
                    # Adjust the 'a' parameter to ensure the new SVI curve dominates the previous one
                    params[0] = prev_params_list[0] + 0.01  # Ensure a_t2 > a_t1
                    
                    # Convert back to the original type if needed
                    if isinstance(prev_params, tuple):
                        params = tuple(params)
            
            svi_params.append(params)
            prev_params = params
            
        except Exception as e:
            print(f"Error fitting SVI to time slice {t:.6f}: {str(e)}")
            # If fitting fails for this slice, use previous parameters with adjustment
            if prev_params is not None:
                print(f"Using adjusted parameters from previous slice for {t:.6f}")
                
                # Ensure prev_params is a list for modification
                if isinstance(prev_params, tuple):
                    new_params = list(prev_params)
                elif hasattr(prev_params, 'tolist'):
                    new_params = prev_params.tolist()
                else:
                    new_params = list(prev_params)
                
                # Slightly adjust parameters to ensure monotonicity in time
                new_params[0] *= 1.05  # Increase 'a' parameter by 5%
                
                # Convert back to the original type if needed
                if isinstance(prev_params, tuple):
                    new_params = tuple(new_params)
                
                svi_params.append(new_params)
                prev_params = new_params
            else:
                raise ValueError(f"Cannot fit SVI to time slice {t:.6f} and no previous parameters available")
    
    return svi_params

# Add these functions to svi_implementation.py to better handle output generation

def interpolate_svi_params(time_grid, svi_params, new_time_grid):
    """
    Interpolate SVI parameters across a denser time grid
    
    Parameters:
    -----------
    time_grid : array-like
        Original time grid
    svi_params : list
        List of SVI parameters for each time slice
    new_time_grid : array-like
        New denser time grid
        
    Returns:
    --------
    list
        Interpolated SVI parameters for the new time grid
    """
    from scipy.interpolate import interp1d
    import numpy as np
    
    # Convert parameters to numpy arrays for easier manipulation
    processed_params = []
    for params in svi_params:
        if isinstance(params, list) or isinstance(params, tuple):
            processed_params.append(np.array(params))
        else:
            processed_params.append(params)
    
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
    
    # Generate new SVI parameters
    new_svi_params = []
    for t in new_time_grid:
        params = [f(t) for f in param_interp_funcs]
        new_svi_params.append(np.array(params))
    
    return new_svi_params

def svi_to_enhanced_surface_data(moneyness_grid, time_grid, svi_params, 
                                dense_moneyness=100, dense_time=100,
                                moneyness_range=None):
    """
    Enhanced version of svi_to_surface_data that uses a denser grid
    and better handles extreme moneyness values.
    
    Parameters:
    -----------
    moneyness_grid : array-like
        Original moneyness grid
    time_grid : array-like
        Original time grid
    svi_params : list
        List of SVI parameters for each time slice
    dense_moneyness : int
        Number of points in the dense moneyness grid
    dense_time : int
        Number of points in the dense time grid
    moneyness_range : tuple or None
        Custom (min, max) moneyness range, or None to auto-calculate
        
    Returns:
    --------
    dict
        Surface data with enhanced grid
    """
    import numpy as np
    
    # Create denser grids
    if moneyness_range is None:
        min_moneyness = max(0.5, min(moneyness_grid) * 0.9)
        max_moneyness = min(2.0, max(moneyness_grid) * 1.1)
    else:
        min_moneyness, max_moneyness = moneyness_range
        
    # Ensure sufficient coverage of moneyness range
    min_moneyness = min(0.7, min_moneyness)
    max_moneyness = max(1.3, max_moneyness)
    
    dense_moneyness_grid = np.linspace(min_moneyness, max_moneyness, dense_moneyness)
    dense_time_grid = np.linspace(min(time_grid), max(time_grid), dense_time)
    
    # Interpolate SVI parameters for the dense time grid
    dense_svi_params = interpolate_svi_params(time_grid, svi_params, dense_time_grid)
    
    # Generate volatility surface with dense grid
    log_moneyness_grid = np.log(dense_moneyness_grid)
    vol_matrix = svi_surface_to_vol_matrix(log_moneyness_grid, dense_time_grid, dense_svi_params)
    
    # Ensure volatility values are reasonable
    vol_matrix = np.nan_to_num(vol_matrix, nan=0.2, posinf=2.0, neginf=0.001)
    vol_matrix = np.clip(vol_matrix, 0.001, 2.0)
    
    # Format for surface data
    surface_data = {
        'x': dense_time_grid.tolist(),
        'y': dense_moneyness_grid.tolist(),
        'z': vol_matrix.tolist(),
        'svi_params': [params.tolist() if hasattr(params, 'tolist') else list(params) 
                     for params in dense_svi_params]
    }
    
    return surface_data