import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import solid from "vite-plugin-solid";

// Multi-page, like Lorraine's: one HTML entry per window: the avatar, the
// control window (Solid); PR 4 adds prompt.html here.
export default defineConfig({
  plugins: [solid()],
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
        control: fileURLToPath(new URL("./control.html", import.meta.url)),
      },
    },
  },
});
