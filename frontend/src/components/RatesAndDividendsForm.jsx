import React, { useState, useEffect } from 'react';
import '../styles/volSurface.css';

const RatesAndDividendsForm = ({ 
  discountRates, 
  repoRates, 
  dividends, 
  onDiscountRatesChange, 
  onRepoRatesChange, 
  onDividendsChange 
}) => {
  const discountDefault = 4.5;
  const repoDefault = 1.0;
  const divDefault = 1.21;

  // State for detailed mode
  const [newDiscountDate, setNewDiscountDate] = useState('');
  const [newDiscountRate, setNewDiscountRate] = useState('');
  
  const [newRepoDate, setNewRepoDate] = useState('');
  const [newRepoRate, setNewRepoRate] = useState('');
  
  const [newDividendDate, setNewDividendDate] = useState('');
  const [newDividendAmount, setNewDividendAmount] = useState('');
  
  // State for simple mode - using localStorage to persist values
  const [useSimpleMode, setUseSimpleMode] = useState(true);
  
  // Initialize with localStorage values or defaults
  const [singleDiscountRate, setSingleDiscountRate] = useState(() => {
    const savedValue = localStorage.getItem('singleDiscountRate');
    return savedValue !== null ? parseFloat(savedValue) : discountDefault;
  });
  
  const [singleRepoRate, setSingleRepoRate] = useState(() => {
    const savedValue = localStorage.getItem('singleRepoRate');
    return savedValue !== null ? parseFloat(savedValue) : repoDefault;
  });
  
  const [annualDividendYield, setAnnualDividendYield] = useState(() => {
    const savedValue = localStorage.getItem('annualDividendYield');
    return savedValue !== null ? parseFloat(savedValue) : divDefault;
  });
  
  // Save values to localStorage when they change
  useEffect(() => {
    localStorage.setItem('singleDiscountRate', singleDiscountRate);
  }, [singleDiscountRate]);
  
  useEffect(() => {
    localStorage.setItem('singleRepoRate', singleRepoRate);
  }, [singleRepoRate]);
  
  useEffect(() => {
    localStorage.setItem('annualDividendYield', annualDividendYield);
  }, [annualDividendYield]);
  
  // Handler for switching between detailed and simple mode
  const handleModeChange = (mode) => {
    setUseSimpleMode(mode === 'simple');
  };
  
  // Apply single rates to all dates - completely separate from detailed mode
  const applySimpleRates = () => {
    console.log('Applying simple mode rates:');
    console.log(`Discount rate: ${singleDiscountRate}%`);
    console.log(`Repo rate: ${singleRepoRate}%`);
    console.log(`Dividend yield: ${annualDividendYield}%`);
    
    // Create some default future dates if none exist
    const datesToUse = [
      new Date(new Date().setMonth(new Date().getMonth() + 3)).toISOString().split('T')[0],
      new Date(new Date().setMonth(new Date().getMonth() + 6)).toISOString().split('T')[0],
      new Date(new Date().setMonth(new Date().getMonth() + 9)).toISOString().split('T')[0],
      new Date(new Date().setFullYear(new Date().getFullYear() + 1)).toISOString().split('T')[0]
    ];
    
    // Convert percentage values to decimal for internal calculations
    const discountRateDecimal = parseFloat(singleDiscountRate) / 100;
    const repoRateDecimal = parseFloat(singleRepoRate) / 100;
    const dividendYieldDecimal = parseFloat(annualDividendYield) / 100;
    
    // Create new rate objects with single rates for all dates
    const newDiscountRates = {};
    const newRepoRates = {};
    
    datesToUse.forEach(date => {
      newDiscountRates[date] = discountRateDecimal;
      newRepoRates[date] = repoRateDecimal;
    });
    
    // For dividends, calculate quarterly payments based on annual yield
    const newDividends = {};
    if (dividendYieldDecimal > 0) {
      // Calculate quarterly dividend amount based on yield
      const quarterlyAmount = dividendYieldDecimal / 4;
      
      // Add quarterly dividend dates for the next year
      for (let i = 1; i <= 4; i++) {
        const dividendDate = new Date();
        dividendDate.setMonth(dividendDate.getMonth() + (i * 3));
        const dateStr = dividendDate.toISOString().split('T')[0];
        newDividends[dateStr] = quarterlyAmount;
      }
    }
    
    // Update Redux state - ensure this happens!
    onDiscountRatesChange(newDiscountRates);
    onRepoRatesChange(newRepoRates);
    onDividendsChange(newDividends);
    
    // Show success message
    alert('Rates applied successfully!');
  };
  
  // Handlers for detailed mode
  const handleAddDiscountRate = () => {
    if (newDiscountDate && newDiscountRate) {
      const updatedRates = {
        ...discountRates,
        [newDiscountDate]: parseFloat(newDiscountRate) / 100 // Convert from percentage to decimal
      };
      onDiscountRatesChange(updatedRates);
      setNewDiscountDate('');
      setNewDiscountRate('');
    }
  };
  
  const handleAddRepoRate = () => {
    if (newRepoDate && newRepoRate) {
      const updatedRates = {
        ...repoRates,
        [newRepoDate]: parseFloat(newRepoRate) / 100 // Convert from percentage to decimal
      };
      onRepoRatesChange(updatedRates);
      setNewRepoDate('');
      setNewRepoRate('');
    }
  };
  
  const handleAddDividend = () => {
    if (newDividendDate && newDividendAmount) {
      const updatedDividends = {
        ...dividends,
        [newDividendDate]: parseFloat(newDividendAmount)
      };
      onDividendsChange(updatedDividends);
      setNewDividendDate('');
      setNewDividendAmount('');
    }
  };
  
  const handleRemoveDiscountRate = (date) => {
    const updatedRates = {...discountRates};
    delete updatedRates[date];
    onDiscountRatesChange(updatedRates);
  };
  
  const handleRemoveRepoRate = (date) => {
    const updatedRates = {...repoRates};
    delete updatedRates[date];
    onRepoRatesChange(updatedRates);
  };
  
  const handleRemoveDividend = (date) => {
    const updatedDividends = {...dividends};
    delete updatedDividends[date];
    onDividendsChange(updatedDividends);
  };
  
  // Helper to format rates for display in detailed mode
  const formatRateForDisplay = (rate) => {
    // Convert from decimal to percentage for display
    return (rate * 100).toFixed(2);
  };
  
  return (
    <div className="form">
      {/* Mode toggle */}
      <div className="form-group">
        <div className="mode-toggle">
          <button 
            className={`mode-toggle__btn ${!useSimpleMode ? 'mode-toggle__btn--active' : ''}`}
            onClick={() => handleModeChange('detailed')}
          >
            Detailed Mode
          </button>
          <button 
            className={`mode-toggle__btn ${useSimpleMode ? 'mode-toggle__btn--active' : ''}`}
            onClick={() => handleModeChange('simple')}
          >
            Simple Mode
          </button>
        </div>
      </div>
      
      {useSimpleMode ? (
        /* Simple Mode */
        <div className="form-group">
          <div className="simple-rates">
            <div className="form-group">
              <label className="form-label">Annual Discount Rate</label>
              <div className="input-with-unit">
                <input
                  type="number"
                  value={singleDiscountRate}
                  onChange={(e) => setSingleDiscountRate(e.target.value)}
                  className="form-input"
                  step="0.00001"
                  min="0"
                  max="100"
                />
                <span className="input-unit">%</span>
              </div>
              <p className="form-help">The annual risk-free rate applied across all dates (as percentage)</p>
            </div>
            
            <div className="form-group">
              <label className="form-label">Annual Repo Rate</label>
              <div className="input-with-unit">
                <input
                  type="number"
                  value={singleRepoRate}
                  onChange={(e) => setSingleRepoRate(e.target.value)}
                  className="form-input"
                  step="0.00001"
                  min="0"
                  max="100"
                />
                <span className="input-unit">%</span>
              </div>
              <p className="form-help">The annual cost of borrowing the underlying security (as percentage)</p>
            </div>
            
            <div className="form-group">
              <label className="form-label">Annual Dividend Yield</label>
              <div className="input-with-unit">
                <input
                  type="number"
                  value={annualDividendYield}
                  onChange={(e) => setAnnualDividendYield(e.target.value)}
                  className="form-input"
                  step="0.00001"
                  min="0"
                  max="100"
                />
                <span className="input-unit">%</span>
              </div>
              <p className="form-help">The annual dividend yield, will be distributed as quarterly payments (as percentage)</p>
            </div>
            
            <button 
              onClick={applySimpleRates}
              className="btn btn--primary"
            >
              Apply Rates
            </button>
          </div>
        </div>
      ) : (
        /* Detailed Mode */
        <>
          {/* Discount Rates */}
          <div className="form-group">
            <h3 className="form-label">Discount Rates</h3>
            
            <div className="form-group">
              <div className="flex space-x-2">
                <div className="flex-1">
                  <input
                    type="date"
                    value={newDiscountDate}
                    onChange={(e) => setNewDiscountDate(e.target.value)}
                    className="form-input"
                    placeholder="Date (YYYY-MM-DD)"
                  />
                </div>
                <div>
                  <div className="input-with-unit">
                    <input
                      type="number"
                      value={newDiscountRate}
                      onChange={(e) => setNewDiscountRate(e.target.value)}
                      className="form-input"
                      placeholder="Rate"
                      step="0.01"
                      style={{ width: '100px' }}
                    />
                    <span className="input-unit">%</span>
                  </div>
                </div>
                <button
                  onClick={handleAddDiscountRate}
                  className="btn btn--primary"
                >
                  Add
                </button>
              </div>
            </div>
            
            <div className="card">
              {Object.keys(discountRates).length > 0 ? (
                <table className="table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Rate</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(discountRates).map(([date, rate]) => (
                      <tr key={date}>
                        <td>{date}</td>
                        <td>{formatRateForDisplay(rate)}%</td>
                        <td>
                          <button
                            onClick={() => handleRemoveDiscountRate(date)}
                            className="text-error"
                          >
                            Remove
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="p-4 text-center">No discount rates added</p>
              )}
            </div>
          </div>
          
          {/* Repo Rates */}
          <div className="form-group">
            <h3 className="form-label">Repo Rates</h3>
            
            <div className="form-group">
              <div className="flex space-x-2">
                <div className="flex-1">
                  <input
                    type="date"
                    value={newRepoDate}
                    onChange={(e) => setNewRepoDate(e.target.value)}
                    className="form-input"
                    placeholder="Date (YYYY-MM-DD)"
                  />
                </div>
                <div>
                  <div className="input-with-unit">
                    <input
                      type="number"
                      value={newRepoRate}
                      onChange={(e) => setNewRepoRate(e.target.value)}
                      className="form-input"
                      placeholder="Rate"
                      step="0.01"
                      style={{ width: '100px' }}
                    />
                    <span className="input-unit">%</span>
                  </div>
                </div>
                <button
                  onClick={handleAddRepoRate}
                  className="btn btn--primary"
                >
                  Add
                </button>
              </div>
            </div>
            
            <div className="card">
              {Object.keys(repoRates).length > 0 ? (
                <table className="table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Rate</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(repoRates).map(([date, rate]) => (
                      <tr key={date}>
                        <td>{date}</td>
                        <td>{formatRateForDisplay(rate)}%</td>
                        <td>
                          <button
                            onClick={() => handleRemoveRepoRate(date)}
                            className="text-error"
                          >
                            Remove
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="p-4 text-center">No repo rates added</p>
              )}
            </div>
          </div>
          
          {/* Dividends */}
          <div className="form-group">
            <h3 className="form-label">Dividends</h3>
            
            <div className="form-group">
              <div className="flex space-x-2">
                <div className="flex-1">
                  <input
                    type="date"
                    value={newDividendDate}
                    onChange={(e) => setNewDividendDate(e.target.value)}
                    className="form-input"
                    placeholder="Date (YYYY-MM-DD)"
                  />
                </div>
                <div>
                  <input
                    type="number"
                    value={newDividendAmount}
                    onChange={(e) => setNewDividendAmount(e.target.value)}
                    className="form-input"
                    placeholder="Amount"
                    step="0.01"
                    style={{ width: '100px' }}
                  />
                </div>
                <button
                  onClick={handleAddDividend}
                  className="btn btn--primary"
                >
                  Add
                </button>
              </div>
            </div>
            
            <div className="card">
              {dividends && Object.keys(dividends).length > 0 ? (
                <table className="table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Amount</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(dividends).map(([date, amount]) => (
                      <tr key={date}>
                        <td>{date}</td>
                        <td>{amount}</td>
                        <td>
                          <button
                            onClick={() => handleRemoveDividend(date)}
                            className="text-error"
                          >
                            Remove
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="p-4 text-center">No dividends added</p>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default RatesAndDividendsForm;