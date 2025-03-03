import React, { useState } from 'react';
import '../styles/volSurface.css';

const RatesAndDividendsForm = ({ 
  discountRates, 
  repoRates, 
  dividends, 
  onDiscountRatesChange, 
  onRepoRatesChange, 
  onDividendsChange 
}) => {
  const discountDefault = 4.5
  const repoDefault = 1.0
  const divDefault = 1.21

  // State for detailed date-specific entries
  const [newDiscountDate, setNewDiscountDate] = useState('');
  const [newDiscountRate, setNewDiscountRate] = useState('');
  
  const [newRepoDate, setNewRepoDate] = useState('');
  const [newRepoRate, setNewRepoRate] = useState('');
  
  const [newDividendDate, setNewDividendDate] = useState('');
  const [newDividendAmount, setNewDividendAmount] = useState('');
  
  // State for simple mode with single rates
  const [useSimpleMode, setUseSimpleMode] = useState(true); // Set simple mode as default
  const [singleDiscountRate, setSingleDiscountRate] = useState(discountDefault);
  const [singleRepoRate, setSingleRepoRate] = useState(repoDefault);
  const [annualDividendYield, setAnnualDividendYield] = useState(divDefault);
  
  // Handler for switching between detailed and simple mode
  const handleModeChange = (mode) => {
    setUseSimpleMode(mode === 'simple');
    
    if (mode === 'simple') {
      // Set single rates based on the average of existing rates or defaults
      // Convert from decimal to percentage for display
      let avgDiscountRate = discountDefault; // Default 5%
      let avgRepoRate = repoDefault; // Default 1%
      let avgDividendYield = divDefault; // Default 2%
      
      if (Object.values(discountRates).length > 0) {
        avgDiscountRate = (Object.values(discountRates).reduce((sum, rate) => sum + rate, 0) / 
          Object.values(discountRates).length * 100).toFixed(5);
      }
      
      if (Object.values(repoRates).length > 0) {
        avgRepoRate = (Object.values(repoRates).reduce((sum, rate) => sum + rate, 0) / 
          Object.values(repoRates).length * 100).toFixed(5);
      }
      
      // For dividend yield, we would need more complex logic, so just use default
      // if there are existing dividends
      if (dividends && Object.keys(dividends).length > 0) {
        // This is a simplified approximation of annual yield
        const totalDividends = Object.values(dividends).reduce((sum, amount) => sum + amount, 0);
        avgDividendYield = (totalDividends * 4 * 100).toFixed(5); // Assuming quarterly
      }
      
      setSingleDiscountRate(avgDiscountRate);
      setSingleRepoRate(avgRepoRate);
      setAnnualDividendYield(avgDividendYield);
    }
  };
  
  // Apply single rates to all dates
  const applySimpleRates = () => {
    // Get all unique dates from existing entries
    const allDates = [...new Set([
      ...Object.keys(discountRates),
      ...Object.keys(repoRates)
    ])];
    
    // If no dates exist, add some default future dates
    const datesToUse = allDates.length > 0 ? allDates : [
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
    if (dividendYieldDecimal > 0) {
      const newDividends = {};
      // Calculate quarterly dividend amount based on yield
      // This is a simplification - in reality, would use spot price and other factors
      const quarterlyAmount = dividendYieldDecimal / 4;
      
      // Add quarterly dividend dates for the next year
      for (let i = 1; i <= 4; i++) {
        const dividendDate = new Date();
        dividendDate.setMonth(dividendDate.getMonth() + (i * 3));
        const dateStr = dividendDate.toISOString().split('T')[0];
        newDividends[dateStr] = quarterlyAmount;
      }
      
      onDividendsChange(newDividends);
    }
    
    onDiscountRatesChange(newDiscountRates);
    onRepoRatesChange(newRepoRates);
  };
  
  // Handlers for detailed mode
  const handleAddDiscountRate = () => {
    if (newDiscountDate && newDiscountRate) {
      const updatedRates = {
        ...discountRates,
        [newDiscountDate]: parseFloat(newDiscountRate)
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
        [newRepoDate]: parseFloat(newRepoRate)
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
        /* Detailed Mode - Original implementation */
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
                  <input
                    type="number"
                    value={newDiscountRate}
                    onChange={(e) => setNewDiscountRate(e.target.value)}
                    className="form-input"
                    placeholder="Rate"
                    step="0.001"
                    style={{ width: '100px' }}
                  />
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
                        <td>{rate}</td>
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
                  <input
                    type="number"
                    value={newRepoRate}
                    onChange={(e) => setNewRepoRate(e.target.value)}
                    className="form-input"
                    placeholder="Rate"
                    step="0.001"
                    style={{ width: '100px' }}
                  />
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
                        <td>{rate}</td>
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