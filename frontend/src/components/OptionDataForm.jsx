// OptionDataForm.jsx
import React from 'react';
import FileUploader from './FileUploader';
import '../styles/volSurface.css';

const OptionDataForm = ({ csvData, onCsvDataChange }) => {
  const handleFileLoaded = (content) => {
    onCsvDataChange(content);
  };

  const handleDownloadSample = () => {
    // Create a link to the sample CSV in the public folder
    const link = document.createElement('a');
    link.href = '/sample_options.csv';
    link.download = 'sample_options.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="form">
      <div className="form-group">
        <div className="form-header">
          <label className="form-label">Option Data (CSV Format)</label>
          <div className="form-actions">
            <button 
              onClick={handleDownloadSample} 
              className="btn btn--link"
              type="button"
            >
              Download Sample
            </button>
            <FileUploader onFileLoaded={handleFileLoaded} />
          </div>
        </div>
        <textarea
          value={csvData}
          onChange={(e) => onCsvDataChange(e.target.value)}
          className="form-textarea"
        />
        <p className="form-help">
          Required columns: Expiry, Strike, Last Trade Date, Option Type<br/>
          At least one price column is required: Last Price, Bid, or Ask<br/>
          Option Type should be CALL or PUT<br/>
          Last Trade Date can be a date (YYYY-MM-DD) or timestamp (YYYY-MM-DD HH:MM:SS+00:00)<br/>
          If your selected price type isn't available for some rows, other price types will be used as fallbacks
        </p>
      </div>
    </div>
  );
};

export default OptionDataForm;