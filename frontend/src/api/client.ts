import axios from "axios";

// Centralized API client — base URL reads from env var for easy prod/staging overrides.
// In dev: VITE_API_BASE_URL is unset, falls back to http://localhost:8000
// In prod: set VITE_API_BASE_URL=https://your-api-domain.com in .env.production
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
  timeout: 15000,
  headers: {
    "Content-Type": "application/json",
  },
});

export default apiClient;
