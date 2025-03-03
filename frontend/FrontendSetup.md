# Setting Up the Vite Frontend

This guide will help you set up the Vite React frontend for the volatility surface calibration project.

## Initial Setup

1. Create a new Vite React project:
   ```bash
   npm create vite@latest frontend -- --template react
   cd frontend
   ```

2. Install necessary dependencies:
   ```bash
   npm install @reduxjs/toolkit react-redux axios plotly.js lodash
   ```

3. Install dev dependencies:
   ```bash
   npm install -D tailwindcss postcss autoprefixer
   npx tailwindcss init -p
   ```

## File Organization

Place the React components in the following structure:

1. Redux store files go in `src/store/`:
   - `store.js` - Redux store configuration
   - `volSurfaceSlice.js` - Volatility surface slice with actions and reducers

2. React components go in `src/components/`:
   - `MarketDataForm.jsx`
   - `OptionDataForm.jsx`
   - `RatesAndDividendsForm.jsx`
   - `VolatilitySurfaceVisualizer.jsx`
   - `VolSurfaceDashboard.jsx`

3. Main entry point at `src/App.jsx` and `src/main.jsx`

## Running in Development

You can run the frontend and backend separately:

### Running the Backend:
```bash
cd backend
python -m venv .venv
.venv\scripts\activate
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

### Running the Vite Frontend:
```bash
cd frontend
npm install
npm run dev
```

## Using Docker Compose

For development with Docker:
```bash
docker-compose -f docker-compose.dev.yml up
```

For production:
```bash
docker-compose up --build
```

## Tailwind CSS Setup

Configure Tailwind by updating the `tailwind.config.js` file:

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

And add the Tailwind directives to your `src/index.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

## API Configuration

The application is configured to proxy API requests to the backend in development mode.
In production, the Nginx configuration handles routing API requests to the backend service.