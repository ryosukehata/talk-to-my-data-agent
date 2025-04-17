# Data Analyst React Frontend

This application provides a modern React-based frontend for the **Talk to My Data** application. It allows users to interact with data, perform analyses, and chat with the system to gain insights from their datasets.

## Features

- Interactive chat interface for data analysis
- Data visualization with interactive plots
- Dataset management and cleansing
- Support for multiple data sources (CSV, AI Catalog, Snowflake, Google Cloud)
- Code execution and insights generation

## Tech Stack

- React 18 with TypeScript
- Vite for fast development and building
- Tailwind CSS for styling
- Jest for testing
- React Query for API state management

## Development

To start the development server (cd into the frontend_react directory):

```bash
npm i
npm run dev
```

To start backend:

in project root

```bash
uvicorn utils.rest_api:app --port 8080
```

## Building

To build the application for production:

```bash
npm run build
```

The build output will be placed in the `../deploy/dist` directory, which is then used by the Python backend to serve the application. When using the React frontend through the `FRONTEND_TYPE="react"` environment variable, the application will look for the built files in this location.

## Testing

To run the test suite:

```bash
npm run test
```

## Project Structure

- `src/api-state`: API client and hooks for data fetching
- `src/components/ui`: shadcn components
- `src/components/ui-custom`: shadcn based generic components
- `src/pages`: Main application pages
- `src/state`: Application state management
- `src/assets`: Static assets like images and icons
