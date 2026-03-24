import { defineConfig } from "vite";

const backendPort = process.env.API_PORT || process.env.PORT || 8000;
const backendTarget = `http://127.0.0.1:${backendPort}`;

export default defineConfig({
  server: {
    proxy: {
      "/api": {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
});
