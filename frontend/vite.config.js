import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backendUrl = env.VITE_API_URL ?? "http://localhost:5001";

  return {
    plugins: [react()],
    server: {
      port: 5173,
      strictPort: true,
      open: true,
      host: "0.0.0.0",
      proxy: {
        "/api": {
          target: backendUrl,
          changeOrigin: true,
          configure: (proxy) => {
            proxy.on("proxyReq", (proxyReq) => {
              proxyReq.setHeader("Accept", "text/event-stream");
            });
          },
        },
      },
    },
    preview: {
      port: 4173,
      strictPort: true,
    },
    build: {
      outDir: "dist",
      sourcemap: true,
      chunkSizeWarningLimit: 1500,
      rollupOptions: {
        output: {
          manualChunks: {
            "react-vendor": ["react", "react-dom"],
            "graph-vendor": ["react-force-graph-2d", "d3", "d3-force"],
          },
        },
      },
    },
    resolve: {
      alias: {
        "@": "/src",
        "@components": "/src/components",
        "@contexts": "/src/contexts",
      },
    },
    optimizeDeps: {
      include: ["react", "react-dom", "react-force-graph-2d", "d3"],
    },
  };
});