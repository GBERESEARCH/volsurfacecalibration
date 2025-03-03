import React from 'react';
import { Provider } from 'react-redux';
import store from 'src/store/store';
import VolSurfaceDashboard from 'src/components/VolSurfaceDashboard';
import 'src/styles/volSurface.css';

const App = () => {
  return (
    <Provider store={store}>
      <div className="app">
        <VolSurfaceDashboard />
      </div>
    </Provider>
  );
};

export default App;
