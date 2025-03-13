// store.js
import { configureStore } from '@reduxjs/toolkit';
import { combineReducers } from 'redux';
import volSurfaceReducer from './volSurfaceSlice';
import diagnosticReducer from './diagnosticSlice';

const rootReducer = combineReducers({
  volSurface: volSurfaceReducer,
  diagnostics: diagnosticReducer
});

const store = configureStore({
  reducer: rootReducer,
});

export default store;

