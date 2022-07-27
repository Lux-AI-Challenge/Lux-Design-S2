import { defineConfig } from 'vite'
import preact from '@preact/preset-vite'
import tsconfigPaths from 'vite-tsconfig-paths'

const preactAliases = (mode: string) => true // mode === 'production'
  ? {
    react: 'preact/compat',
    "react-dom": 'preact/compat',
    "react/jsx-runtime.js": 'preact/compat/jsx-runtime',
    "react-dom/test-utils": 'preact/test-utils',
  } : {}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [preact(), tsconfigPaths()],
  resolve: {
    alias: {
      ...preactAliases(mode),
    },
  },
}))
