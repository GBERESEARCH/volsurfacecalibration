import React, { useState } from 'react';
import '../styles/volSurface.css';

const OptionDiagnosticTable = ({ diagnosticData, summary }) => {
  const [filterStatus, setFilterStatus] = useState('all'); // 'all', 'included', 'excluded'
  const [sortBy, setSortBy] = useState('strike'); // Default sort by strike
  const [sortDirection, setSortDirection] = useState('asc'); // 'asc' or 'desc'
  
  if (!diagnosticData || diagnosticData.length === 0) {
    return (
      <div className="diagnostics-empty">
        No option diagnostic data available. Calibrate the surface to see option details.
      </div>
    );
  }

  // Filter data based on status
  const filteredData = filterStatus === 'all' 
    ? diagnosticData 
    : diagnosticData.filter(option => option.status.toLowerCase() === filterStatus);

  // Sort data
  const sortedData = [...filteredData].sort((a, b) => {
    // Special handling for implied_vol as it can be null
    if (sortBy === 'implied_vol') {
      if (a.implied_vol === null && b.implied_vol === null) return 0;
      if (a.implied_vol === null) return 1;
      if (b.implied_vol === null) return -1;
    }
    
    if (a[sortBy] < b[sortBy]) return sortDirection === 'asc' ? -1 : 1;
    if (a[sortBy] > b[sortBy]) return sortDirection === 'asc' ? 1 : -1;
    return 0;
  });

  // Handle sort column change
  const handleSortChange = (column) => {
    if (sortBy === column) {
      // Toggle direction if same column
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      // New column, default to ascending
      setSortBy(column);
      setSortDirection('asc');
    }
  };

  return (
    <div className="diagnostic-container">
      <div className="diagnostic-header">
        <h3 className="diagnostic-title">Option Diagnostics</h3>
        <div className="diagnostic-summary">
          <span className="diagnostic-stat">
            <strong>{summary.includedOptions}</strong> of <strong>{summary.totalOptions}</strong> options included
            ({summary.inclusionRate}%)
          </span>
          <div className="diagnostic-filters">
            <select 
              value={filterStatus} 
              onChange={(e) => setFilterStatus(e.target.value)}
              className="diagnostic-filter"
            >
              <option value="all">All Options</option>
              <option value="included">Included Options</option>
              <option value="excluded">Excluded Options</option>
            </select>
          </div>
        </div>
      </div>
      
      <div className="diagnostic-table-container">
        <table className="diagnostic-table">
          <thead>
            <tr>
              <th onClick={() => handleSortChange('strike')} className={sortBy === 'strike' ? `sorted-${sortDirection}` : ''}>
                Strike
              </th>
              <th onClick={() => handleSortChange('expiry')} className={sortBy === 'expiry' ? `sorted-${sortDirection}` : ''}>
                Expiry
              </th>
              <th onClick={() => handleSortChange('option_type')} className={sortBy === 'option_type' ? `sorted-${sortDirection}` : ''}>
                Type
              </th>
              <th onClick={() => handleSortChange('price')} className={sortBy === 'price' ? `sorted-${sortDirection}` : ''}>
                Price
              </th>
              <th onClick={() => handleSortChange('status')} className={sortBy === 'status' ? `sorted-${sortDirection}` : ''}>
                Status
              </th>
              <th onClick={() => handleSortChange('implied_vol')} className={sortBy === 'implied_vol' ? `sorted-${sortDirection}` : ''}>
                Impl. Vol
              </th>
              <th>Exclusion Reason</th>
            </tr>
          </thead>
          <tbody>
            {sortedData.map((option, index) => (
              <tr 
                key={index} 
                className={option.status.toLowerCase() === 'excluded' ? 'excluded-option' : 'included-option'}
              >
                <td>{option.strike.toFixed(2)}</td>
                <td>{option.expiry}</td>
                <td>{option.option_type.toUpperCase()}</td>
                <td>{option.price.toFixed(4)}</td>
                <td>
                  <span className={`status-badge status-${option.status.toLowerCase()}`}>
                    {option.status}
                  </span>
                </td>
                <td>
                  {option.implied_vol 
                    ? (option.implied_vol * 100).toFixed(2) + '%' 
                    : '-'}
                </td>
                <td className="exclusion-reason">
                  {option.reason || '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default OptionDiagnosticTable;