# Volatility Surface Calibration

A comprehensive application for calibrating and visualizing arbitrage-free volatility surfaces from options market data.

## Project Structure

This project consists of a Python FastAPI backend and a React frontend:

- `backend/`: Python-based FastAPI service for volatility surface calibration
- `frontend/`: React/Redux application with Plotly for 3D visualization

## Running the Application

### Backend (Python)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Run the backend server:
   ```bash
   uvicorn main:app --reload
   ```

The backend will be available at http://localhost:8000

### Frontend (React)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at http://localhost:5173

## Project Features

- Calibration of arbitrage-free volatility surfaces from market data
- 3D visualization of volatility surfaces using Plotly
- Form inputs for market data, discount rates, repo rates, and dividends
- Calculation of implied volatilities using Newton-Raphson method
- Support for different price types (bid, ask, mid, last)
- Forward curve calculation with discount rates and dividends

## Technology Stack

### Backend
- Python 3.13+
- FastAPI
- Pandas, NumPy, SciPy
- Implied volatility calculations

### Frontend
- React 18
- Redux Toolkit for state management
- Plotly.js for 3D visualization
- Axios for API communication
- Traditional CSS for styling

## Fixing Common Issues

### StringIO Import Error
If you encounter `module 'pandas' has no attribute 'StringIO'` error:
- Edit `main.py` to add: `from io import StringIO` 
- Replace `pd.StringIO` with `StringIO`

### Plotly.js Errors
If you encounter Plotly errors:
- Make sure to create a deep copy of the data before passing to Plotly:
  ```js
  const surfaceJson = JSON.parse(JSON.stringify(surfaceData));
  ```

## Development Notes

- The FastAPI backend exposes a `/calibrate_surface` endpoint that accepts CSV data and returns the calibrated surface
- The frontend makes API calls through a proxy at `/api` that redirects to the backend server
- Redux is used for state management and API communication
- CSS uses a traditional class-based approach with semantic naming