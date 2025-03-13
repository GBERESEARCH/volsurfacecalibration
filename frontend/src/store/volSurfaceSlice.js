import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';
import { fetchOptionDiagnostics } from './diagnosticSlice';

// Use environment variables or default to API proxy
const API_URL = import.meta.env.VITE_API_URL || '/api';

// Async thunk for calibrating vol surface
export const calibrateVolSurface = createAsyncThunk(
  'volSurface/calibrate',
  async (calibrationData, { dispatch, rejectWithValue }) => {
    try {
      const response = await axios.post(`${API_URL}/calibrate_surface`, calibrationData);
      
      // After successful calibration, also fetch diagnostics
      dispatch(fetchOptionDiagnostics(calibrationData));
      
      return response.data;
    } catch (error) {
      return rejectWithValue(
        error.response?.data?.detail || 'Failed to calibrate volatility surface'
      );
    }
  }
);

const initialState = {
  surfaceData: null,
  marketData: {
    spot: 110.0,
    discountRates: { '2025-03-21': 0.05, '2025-06-20': 0.052 },
    repoRates: { '2025-03-21': 0.01, '2025-06-20': 0.012 },
    dividends: { '2025-05-15': 1.5 }
  },
  priceType: 'mid',
  referenceDate: new Date().toISOString().split('T')[0], // Today's date as default
  fittingMethod: 'rbf',
  csvData: `Expiry,Strike,Last Trade Date,Last Price,Bid,Ask,Option Type
2025-03-21,100,2024-02-28,10.80,10.50,11.00,CALL
2025-03-21,110,2024-02-28,5.30,5.20,5.50,CALL
2025-03-21,120,2024-02-28,2.20,2.10,2.30,CALL
2025-03-21,100,2024-02-28,1.60,1.50,1.80,PUT
2025-03-21,110,2024-02-28,6.20,6.00,6.40,PUT
2025-03-21,120,2024-02-28,13.00,12.80,13.20,PUT
2025-06-20,100,2024-02-28,12.50,12.30,12.80,CALL
2025-06-20,110,2024-02-28,8.00,7.80,8.20,CALL
2025-06-20,120,2024-02-28,4.70,4.50,4.90,CALL
2025-06-20,100,2024-02-28,4.00,3.80,4.20,PUT
2025-06-20,110,2024-02-28,9.40,9.20,9.60,PUT
2025-06-20,120,2024-02-28,15.80,15.50,16.00,PUT`,
  status: 'idle', // 'idle' | 'loading' | 'succeeded' | 'failed'
  error: null
};

const volSurfaceSlice = createSlice({
  name: 'volSurface',
  initialState,
  reducers: {
    updateSpot(state, action) {
      state.marketData.spot = action.payload;
    },
    updatePriceType(state, action) {
      state.priceType = action.payload;
    },
    updateReferenceDate(state, action) {
      state.referenceDate = action.payload;
    },
    updateFittingMethod(state, action) {
      state.fittingMethod = action.payload;
    },
    updateCsvData(state, action) {
      state.csvData = action.payload;
    },
    updateDiscountRates(state, action) {
      // Make sure we're not mutating the original object reference
      state.marketData.discountRates = { ...action.payload };
      console.log('Redux state updated with discount rates:', state.marketData.discountRates);
    },
    updateRepoRates(state, action) {
      // Make sure we're not mutating the original object reference
      state.marketData.repoRates = { ...action.payload };
      console.log('Redux state updated with repo rates:', state.marketData.repoRates);
    },
    updateDividends(state, action) {
      // Make sure we're not mutating the original object reference
      state.marketData.dividends = action.payload ? { ...action.payload } : {};
      console.log('Redux state updated with dividends:', state.marketData.dividends);
    },
    resetCalibration(state) {
      state.surfaceData = null;
      state.status = 'idle';
      state.error = null;
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(calibrateVolSurface.pending, (state) => {
        state.status = 'loading';
        state.error = null;
      })
      .addCase(calibrateVolSurface.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.surfaceData = action.payload.surface_data;
      })
      .addCase(calibrateVolSurface.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.payload;
      });
  }
});

export const {
  updateSpot,
  updatePriceType,
  updateReferenceDate,
  updateFittingMethod,
  updateCsvData,
  updateDiscountRates,
  updateRepoRates,
  updateDividends,
  resetCalibration
} = volSurfaceSlice.actions;

export default volSurfaceSlice.reducer;