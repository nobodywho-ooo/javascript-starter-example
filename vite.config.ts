import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  // NobodyWho's WASM glue uses top-level await, which is only supported in
  // ES module output — not the classic IIFE format Vite defaults to for workers.
  worker: {
    format: 'es',
  },
  server: {
    fs: {
      // The local `react-native-nobodywho` checkout lives in a sibling
      // directory; Vite must be allowed to serve its `uniffi.wasm` file.
      allow: ['..'],
    },
  },
})
