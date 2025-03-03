import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js/dist/plotly';
import 'src/styles/volSurface.css';

const VolatilitySurfaceVisualizer = ({ surfaceData }) => {
  const plotRef = useRef(null);
  const hasRenderedRef = useRef(false);
  
  useEffect(() => {
    // Only proceed if we have surface data and a valid DOM element
    if (!surfaceData || !plotRef.current) return;
    
    try {
      // Parse string data if needed
      const surfaceJson = typeof surfaceData === 'string' 
        ? JSON.parse(surfaceData) 
        : JSON.parse(JSON.stringify(surfaceData)); // Deep copy
      
      if (!hasRenderedRef.current) {
        // First time rendering
        Plotly.newPlot(plotRef.current, surfaceJson.data, surfaceJson.layout);
        hasRenderedRef.current = true;
      } else {
        // Update existing plot
        Plotly.react(plotRef.current, surfaceJson.data, surfaceJson.layout);
      }
      
      // Handle resize
      const resizeHandler = () => {
        if (plotRef.current) {
          Plotly.Plots.resize(plotRef.current);
        }
      };
      
      window.addEventListener('resize', resizeHandler);
      
      return () => {
        window.removeEventListener('resize', resizeHandler);
        // Don't purge the plot on every effect cleanup
      };
    } catch (error) {
      console.error("Error rendering plot:", error);
    }
  }, [surfaceData]);
  
  // Only purge the plot when component unmounts
  useEffect(() => {
    return () => {
      if (plotRef.current) {
        Plotly.purge(plotRef.current);
      }
    };
  }, []);
  
  return (
    <div className="visualizer">
      <div ref={plotRef} className="visualizer__plot"></div>
      <div className="visualizer__legend">
        <p className="visualizer__legend-title">Surface Legend:</p>
        <ul className="visualizer__legend-list">
          <li className="visualizer__legend-item">• <span className="visualizer__legend-label">X-axis:</span> Time to Expiry (years)</li>
          <li className="visualizer__legend-item">• <span className="visualizer__legend-label">Y-axis:</span> Moneyness (K/F)</li>
          <li className="visualizer__legend-item">• <span className="visualizer__legend-label">Z-axis:</span> Implied Volatility</li>
          <li className="visualizer__legend-item">• <span className="visualizer__legend-label">Red Dots:</span> Market Data Points</li>
          <li className="visualizer__legend-item">• <span className="visualizer__legend-label">Surface:</span> Fitted Arbitrage-Free Volatility Surface</li>
        </ul>
      </div>
    </div>
  );
};

export default VolatilitySurfaceVisualizer;