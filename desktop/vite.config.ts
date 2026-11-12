import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";

// Multi-page, like Lorraine's: one HTML entry per window. Spike 0 has only the
// avatar; PR 2 adds control.html and PR 4 prompt.html here.
export default defineConfig({
  clearScreen: false,
  server: {
    // 1431, so this can run beside Lorraine (1430) and DivoraVoice (1420).
    port: 1431,
    strictPort: true,
    watch: { ignored: ["**/src-tauri/**"] },
  },
  build: {
    target: "esnext",
    // three.js and three-vrm in one chunk, embedded in the exe and read from
    // disk, never over a network: its size costs nothing here.
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      input: {
        avatar: fileURLToPath(new URL("./avatar.html", import.meta.url)),
      },
    },
  },
});
