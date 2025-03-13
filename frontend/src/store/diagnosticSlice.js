import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

// Use environment variables or default to API proxy
const API_URL = import.meta.env.VITE_API_URL || '/api';

// Async thunk for fetching option diagnostics
export const fetchOptionDiagnostics = createAsyncThunk(
  'diagnostics/fetchOptionDiagnostics',
  async (diagnosticData, { rejectWithValue }) => {
    try {
      const response = await axios.post(`${API_URL}/diagnostics`, diagnosticData);
      return response.data;
    } catch (error) {
      return rejectWithValue(
        error.response?.data?.detail || 'Failed to fetch diagnostics'
      );
    }
  }
);

const initialState = {
  diagnosticData: [],
  summary: {
    totalOptions: 0,
    includedOptions: 0,
    excludedOptions: 0,
    inclusionRate: 0
  },
  status: 'idle', // 'idle' | 'loading' | 'succeeded' | 'failed'
  error: null
};

const diagnosticSlice = createSlice({
  name: 'diagnostics',
  initialState,
  reducers: {
    clearDiagnostics(state) {
      state.diagnosticData = [];
      state.summary = {
        totalOptions: 0,
        includedOptions: 0,
        excludedOptions: 0,
        inclusionRate: 0
      };
      state.status = 'idle';
      state.error = null;
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchOptionDiagnostics.pending, (state) => {
        state.status = 'loading';
        state.error = null;
      })
      .addCase(fetchOptionDiagnostics.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.diagnosticData = action.payload.diagnostics;
        state.summary = action.payload.summary;
      })
      .addCase(fetchOptionDiagnostics.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.payload;
      });
  }
});

export const { clearDiagnostics } = diagnosticSlice.actions;

export default diagnosticSlice.reducer;