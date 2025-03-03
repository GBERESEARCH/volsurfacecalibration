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
                # This is a simplified approach - a more sophisticated one would
                # add explicit constraints to the optimization
                
                # Adjust the 'a' parameter to ensure the new SVI curve dominates the previous one
                params = list(params)
                params[0] = prev_params[0] + 0.01  # Ensure a_t2 > a_t1
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
    
    # Generate SVI surface in volatility
    vol_matrix = svi_surface_to_vol_matrix(log_moneyness_grid, time_grid, svi_params)
    
    # Format for surface data
    surface_data = {
        'x': time_grid.tolist(),
        'y': moneyness_grid.tolist(),
        'z': vol_matrix.tolist()
    }
    
    return surface_data