import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    // Dev-only proxy: the Spring Boot backend currently has no CORS
    // configuration, so browser calls to http://127.0.0.1:8080 would be
    // blocked. The app calls /api/... which is forwarded here. In production
    // builds, VITE_API_URL must point at a backend that allows the
    // frontend origin (or be served from the same origin).
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
      },
    },
  },
});
