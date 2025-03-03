import React, { useRef } from 'react';
import 'src/styles/volSurface.css';

const FileUploader = ({ onFileLoaded }) => {
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Check if the file is a CSV
    if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
      alert('Please upload a CSV file');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      onFileLoaded(content);
    };
    reader.onerror = () => {
      alert('Error reading file');
    };
    reader.readAsText(file);
  };

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="file-uploader">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".csv"
        className="file-uploader__input"
      />
      <button 
        type="button" 
        onClick={handleButtonClick} 
        className="btn btn--secondary"
      >
        Load CSV File
      </button>
    </div>
  );
};

export default FileUploader;