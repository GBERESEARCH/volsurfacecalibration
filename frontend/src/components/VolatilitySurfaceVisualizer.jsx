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
      
      // Ensure the scene object exists and has proper axis titles
      if (surfaceJson.layout && surfaceJson.layout.scene) {
        // Make sure the axes have the correct titles
        surfaceJson.layout.scene.xaxis = {
          ...surfaceJson.layout.scene.xaxis,
          title: {
            text: 'Time to Expiry (days)',
            font: { size: 14 }
          },
          autorange: 'reversed' // Ensure x-axis is reversed
        };
        
        surfaceJson.layout.scene.yaxis = {
          ...surfaceJson.layout.scene.yaxis,
          title: {
            text: 'Strike Price',
            font: { size: 14 }
          }
        };
        
        surfaceJson.layout.scene.zaxis = {
          ...surfaceJson.layout.scene.zaxis,
          title: {
            text: 'Implied Volatility (%)',
            font: { size: 14 }
          }
        };
        
        // Set a good default camera angle
        surfaceJson.layout.scene.camera = {
          eye: { x: 1.5, y: 1.5, z: 1.2 }
        };
      }
      
      // Customize layout for larger graph area
      if (surfaceJson.layout) {
        // Increase the chart size
        surfaceJson.layout.height = 600;
        
        // Adjust margins to maximize graph area
        surfaceJson.layout.margin = {
          l: 50,
          r: 30,
          b: 50,
          t: 30,
          pad: 4
        };
        
        // Remove the legend to save space
        surfaceJson.layout.showlegend = false;
      }
      
      // Ensure colorbar has proper title
      if (surfaceJson.data && surfaceJson.data.length > 0) {
        const surfacePlot = surfaceJson.data.find(d => d.type === 'surface');
        if (surfacePlot) {
          surfacePlot.colorbar = {
            title: 'Implied Vol (%)',
            titleside: 'right'
          };
        }
      }
     
      if (!hasRenderedRef.current) {
        // First time rendering
        Plotly.newPlot(plotRef.current, surfaceJson.data, surfaceJson.layout, {
          responsive: true
        });
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
    </div>
  );
};

export default VolatilitySurfaceVisualizer;