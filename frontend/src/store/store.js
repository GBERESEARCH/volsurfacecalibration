// store.js
import { configureStore } from '@reduxjs/toolkit';
import { combineReducers } from 'redux';
import volSurfaceReducer from './volSurfaceSlice';

const rootReducer = combineReducers({
  volSurface: volSurfaceReducer,
});

const store = configureStore({
  reducer: rootReducer,
});

export default store;

