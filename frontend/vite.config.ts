import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/validation-api": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
        // Frontend calls /validation-api/api/*; drop only the /validation-api
        // prefix so backend sees /api/* (not /api/api/*).
        rewrite: (path) => path.replace(/^\/validation-api/, ""),
      },
    },
  },
});
