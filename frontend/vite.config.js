import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  build: {
    rollupOptions: {
      input: {
        upload: resolve(import.meta.dirname, "upload.html"),
        processing: resolve(import.meta.dirname, "processing.html"),
        index: resolve(import.meta.dirname, "index.html"),
        dashboard: resolve(import.meta.dirname, "dashboard.html"),
        meetings: resolve(import.meta.dirname, "meetings.html"),
        meeting: resolve(import.meta.dirname, "meeting.html"),
      },
    },
  },
});
