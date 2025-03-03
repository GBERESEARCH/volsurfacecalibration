# Equity Derivatives Volatility Surface Calibration

A comprehensive application for calibrating and visualizing arbitrage-free volatility surfaces from options market data.

## Architecture

This application follows a modern three-tier architecture:

1. **Frontend**: React.js with Redux for state management
2. **Backend**: FastAPI Python server
3. **Data Processing**: Python-based volatility surface calibration engine

![Architecture Diagram](https://placeholder.com/architecture-diagram)

### Key Components

#### Backend (Python FastAPI)

- RESTful API for data processing
- Volatility surface calibration using arbitrage-free models
- Implied volatility calculation with Newton-Raphson method
- Forward curve construction with discount rates, repo rates, and dividends

#### Frontend (React + Redux)

- Interactive 3D visualization of volatility surfaces using Plotly
- Form-based data entry for market parameters
- Redux state management for application state
- Responsive UI with Tailwind CSS

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git
- Python 3.13+ (if running without Docker)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/volatility-surface-calibration.git
   cd volatility-surface-calibration
   ```

2. Start the application using Docker Compose:
   ```
   docker-compose up -d
   ```

3. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/docs

## Usage

1. **Enter Option Data**: 
   - Input your option data in CSV format
   - Required columns: expiry, strike, option_type
   - Price columns: call_bid, call_ask, call_last, put_bid, put_ask, put_last

2. **Configure Market Data**:
   - Set the spot price
   - Choose the price type (bid, ask, mid, or last)
   - Enter discount rates, repo rates, and dividends

3. **Calibrate Surface**:
   - Click the "Calibrate Volatility Surface" button
   - The application will fit an arbitrage-free volatility surface
   - View the 3D visualization of the surface

## Technical Details

### System Requirements

- **Backend**: Python 3.13 with FastAPI
- **Frontend**: Node.js 18+ with React and Redux
- **Deployment**: Docker and Docker Compose

### Volatility Surface Calibration

The calibration process follows these steps:

1. **Data Preparation**:
   - Parse option data from CSV
   - Calculate forward prices using discount rates, repo rates, and dividends

2. **Implied Volatility Calculation**:
   - Use Newton-Raphson method to solve for implied volatilities
   - Apply bounds to ensure sensible results

3. **Surface Fitting**:
   - Radial Basis Function (RBF) interpolation
   - Ensure absence of butterfly arbitrage (convexity in strikes)
   - Ensure absence of calendar arbitrage (monotonicity in time)

4. **Visualization**:
   - Create 3D surface with moneyness on the y-axis and time to maturity on the x-axis
   - Display original market data points

## API Documentation

The FastAPI backend provides automatic documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Directory Structure

```
.
├── backend/
│   ├── main.py               # FastAPI application
│   ├── vol_surface_calibrator.py  # Volatility surface calibration class
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── store/            # Redux store and slices
│   │   └── App.js            # Main application
│   ├── public/
│   ├── package.json
│   └── Dockerfile
└── docker-compose.yml
```
