import React, { useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { 
  calibrateVolSurface, 
  updateSpot, 
  updatePriceType, 
  updateReferenceDate,
  updateFittingMethod,
  updateCsvData,
  updateDiscountRates,
  updateRepoRates,
  updateDividends
} from '../store/volSurfaceSlice';
import VolatilitySurfaceVisualizer from './VolatilitySurfaceVisualizer';
import MarketDataForm from './MarketDataForm';
import OptionDataForm from './OptionDataForm';
import RatesAndDividendsForm from './RatesAndDividendsForm';
import '../styles/volSurface.css';

const VolSurfaceDashboard = () => {
  const dispatch = useDispatch();
  const { 
    surfaceData, 
    marketData, 
    priceType,
    referenceDate,
    fittingMethod,
    csvData, 
    status, 
    error 
  } = useSelector(state => state.volSurface);
  
  const [activeTab, setActiveTab] = useState('options');
  
  const handleCalibrate = () => {
    const calibrationData = {
      csv_data: csvData,
      price_type: priceType,
      reference_date: referenceDate,
      fitting_method: fittingMethod,
      spot: marketData.spot,
      discount_rates: marketData.discountRates,
      repo_rates: marketData.repoRates,
      dividends: marketData.dividends
    };
    
    dispatch(calibrateVolSurface(calibrationData));
  };
  
  return (
    <div className="container">
      <header className="header">
        <h1 className="header__title">Equity Derivatives Volatility Surface</h1>
        <p className="header__subtitle">Calibrate and visualize arbitrage-free volatility surfaces from market data</p>
      </header>
      
      <div className="dashboard">
        {/* Left Panel - Forms and Controls */}
        <div className="dashboard__controls">
          <div className="card">
            <div className="card__tabs">
              <button 
                className={`card__tab ${activeTab === 'options' ? 'card__tab--active' : ''}`}
                onClick={() => setActiveTab('options')}
              >
                Option Data
              </button>
              <button 
                className={`card__tab ${activeTab === 'rates' ? 'card__tab--active' : ''}`}
                onClick={() => setActiveTab('rates')}
              >
                Rates & Dividends
              </button>
              <button 
                className={`card__tab ${activeTab === 'config' ? 'card__tab--active' : ''}`}
                onClick={() => setActiveTab('config')}
              >
                Configuration
              </button>
            </div>
            
            <div className="card__content">
              {activeTab === 'options' && (
                <OptionDataForm 
                  csvData={csvData}
                  onCsvDataChange={(value) => dispatch(updateCsvData(value))}
                />
              )}
              
              {activeTab === 'rates' && (
                <RatesAndDividendsForm 
                  discountRates={marketData.discountRates}
                  repoRates={marketData.repoRates}
                  dividends={marketData.dividends}
                  onDiscountRatesChange={(rates) => dispatch(updateDiscountRates(rates))}
                  onRepoRatesChange={(rates) => dispatch(updateRepoRates(rates))}
                  onDividendsChange={(dividends) => dispatch(updateDividends(dividends))}
                />
              )}
              
              {activeTab === 'config' && (
                <MarketDataForm 
                  spot={marketData.spot}
                  priceType={priceType}
                  referenceDate={referenceDate}
                  fittingMethod={fittingMethod}
                  onSpotChange={(value) => dispatch(updateSpot(parseFloat(value)))}
                  onPriceTypeChange={(value) => dispatch(updatePriceType(value))}
                  onReferenceDateChange={(value) => dispatch(updateReferenceDate(value))}
                  onFittingMethodChange={(value) => dispatch(updateFittingMethod(value))}
                />
              )}
            </div>
          </div>
          
          <button
            onClick={handleCalibrate}
            disabled={status === 'loading'}
            className="btn btn--primary btn--full"
          >
            {status === 'loading' ? 'Calibrating...' : 'Calibrate Volatility Surface'}
          </button>
          
          {error && (
            <div className="alert alert--error">
              <p className="m-0"><strong>Error</strong></p>
              <p className="m-0">{error}</p>
            </div>
          )}
        </div>
        
        {/* Right Panel - Visualization */}
        <div className="dashboard__visualization">
          <h2 className="visualization-title">Volatility Surface</h2>
          
          {status === 'loading' ? (
            <div className="loader">
              <div className="loader__spinner"></div>
            </div>
          ) : surfaceData ? (
            <VolatilitySurfaceVisualizer surfaceData={surfaceData} />
          ) : (
            <div className="loader">
              <p className="text-center">
                Calibrate the volatility surface to view the 3D visualization
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VolSurfaceDashboard;