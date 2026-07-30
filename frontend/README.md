# NewsIntel AI - Frontend Application

This is the React frontend for the **NewsIntel AI – Multi-Agent Enterprise Platform**. It provides the user interface for the dashboard, AI chat, news comparison, and analytics.

## Technologies Used
- **React 18** (with Vite for fast bundling)
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **React Router** for navigation
- **shadcn/ui** (planned for accessible components)
- **Axios** for API requests

## Folder Structure
- `src/layouts/`: Contains the `MainLayout.tsx` which provides the sidebar and header.
- `src/pages/`: Contains the main application views (`Dashboard.tsx`, `Chat.tsx`, `Compare.tsx`, `Analytics.tsx`).
- `src/components/`: Reusable UI components.
- `src/App.tsx`: Handles all routing logic.

## Running Locally

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Open your browser and navigate to the URL provided in the terminal (usually `http://localhost:5173`).

## Connecting to Backend
Ensure the FastAPI backend is running simultaneously on `http://localhost:8000` for features like the AI Chat and Analytics to fetch real data via LangGraph and Qdrant.
