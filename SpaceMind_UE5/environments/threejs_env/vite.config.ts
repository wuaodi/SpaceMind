import { defineConfig } from "vite";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig({
  root: path.resolve(__dirname, "app"),
  server: {
    host: "127.0.0.1",
    port: 5173,
    fs: {
      allow: [__dirname],
    },
  },
  resolve: {
    alias: {
      "@shared": path.resolve(__dirname, "shared"),
      "@sim": path.resolve(__dirname, "sim_core"),
    },
  },
});
