import react from '@vitejs/plugin-react';
import { defineConfig, resolveBaseUrl } from 'vite';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // base: './',
  // build: {
  //   outDir: 'dist',
  // },
  // publicDir: 'public',
});
