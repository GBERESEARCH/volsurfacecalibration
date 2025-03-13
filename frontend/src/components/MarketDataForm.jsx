// MarketDataForm.jsx
import React from 'react';
import '../styles/volSurface.css';

const MarketDataForm = ({ 
  spot, 
  priceType, 
  referenceDate,
  fittingMethod,
  onSpotChange, 
  onPriceTypeChange,
  onReferenceDateChange,
  onFittingMethodChange
}) => {
  // Get current date in YYYY-MM-DD format for default reference date
  const today = new Date().toISOString().split('T')[0];

  return (
    <div className="form">
      <div className="form-group">
        <label className="form-label">Spot Price</label>
        <input 
          type="number"
          value={spot}
          onChange={(e) => onSpotChange(e.target.value)}
          className="form-input"
        />
      </div>
      
      <div className="form-group">
        <label className="form-label">Price Type</label>
        <select
          value={priceType}
          onChange={(e) => onPriceTypeChange(e.target.value)}
          className="form-select"
        >
          <option value="bid">Bid</option>
          <option value="ask">Ask</option>
          <option value="mid">Mid</option>
          <option value="last">Last</option>
        </select>
        <p className="form-help">
          Select which price to use for implied volatility calculations
        </p>
      </div>
      
      <div className="form-group">
        <label className="form-label">Reference Date</label>
        <input 
          type="date"
          value={referenceDate || today}
          onChange={(e) => onReferenceDateChange(e.target.value)}
          className="form-input"
        />
        <p className="form-help">
          Only include options with 'Last Trade Date' on or before this date
        </p>
      </div>
            
      <div className="form-group">
        <label className="form-label">Surface Fitting Method</label>
        <select
          value={fittingMethod}
          onChange={(e) => onFittingMethodChange(e.target.value)}
          className="form-select"
        >
          <option value="rbf">RBF Interpolation</option>
          <option value="svi">SVI Parameterization</option>
        </select>
        {fittingMethod === 'svi' ? (
          <div className="form-help">
            <p>
              SVI (Stochastic Volatility Inspired) provides a theoretically sound parameterization with no-arbitrage constraints
            </p>
            <div className="svi-requirements">
              <strong>Requirements for SVI to work:</strong>
              <ul>
                <li>At least 2 different expiry dates in your data</li>
                <li>At least 3 options per expiry date</li>
                <li>Good spread of strikes for each expiry</li>
              </ul>
            </div>
          </div>
        ) : (
          <p className="form-help">
            RBF (Radial Basis Function) provides a flexible interpolation suitable for sparse or irregularly spaced data
          </p>
        )}
      </div>
    </div>
  );
};

export default MarketDataForm;