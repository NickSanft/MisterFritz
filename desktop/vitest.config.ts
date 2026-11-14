import { defineConfig } from "vitest/config";
import solid from "vite-plugin-solid";

// Two projects: "node" for the pure modules and the build's gates, as before,
// and "dom" for the Solid components (DivoraVoice's own setup: jsdom, the
// Solid plugin, jest-dom's matchers).
export default defineConfig({
  test: {
    globals: true,
    projects: [
      {
        extends: true,
        test: {
          name: "node",
          environment: "node",
          include: ["src/**/*.test.ts", "tests/**/*.test.ts"],
        },
      },
      {
        extends: true,
        // No hot reload in a test run: its runtime import (@solid-refresh) has no
        // file to load under vitest 5.
        plugins: [solid({ hot: false })],
        resolve: { conditions: ["development", "browser"] },
        test: {
          name: "dom",
          environment: "jsdom",
          include: ["src/**/*.test.tsx"],
          setupFiles: ["src/test-setup.ts"],
        },
      },
    ],
  },
});
