/// <reference types="vitest/config" />
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { createReadStream, existsSync } from 'node:fs'
import { join } from 'node:path'

// onnxruntime-web dynamically import()s its WASM glue .mjs files (from wherever
// ort.env.wasm.wasmPaths points -- here '/', i.e. public/) to set up threaded
// execution inside the worker. Vite's dev server refuses to serve public/ files
// for any request carrying a module-import marker, since public/ assets are meant
// to be served byte-for-byte, never transformed as source -- so that dynamic
// import 500s in dev even though the same files are served untouched in prod
// builds. This middleware serves the ORT wasm/mjs files directly, bypassing
// Vite's transform pipeline, matching production behavior.
function serveOrtWasmFiles(): Plugin {
  const ORT_FILE_RE = /^\/ort-wasm-simd-threaded(?:\.[a-z]+)?\.(mjs|wasm)$/
  return {
    name: 'serve-ort-wasm-files',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url?.split('?')[0] ?? ''
        const match = url.match(ORT_FILE_RE)
        if (!match) return next()
        const filePath = join(server.config.publicDir, url)
        if (!existsSync(filePath)) return next()
        res.setHeader('Content-Type', match[1] === 'wasm' ? 'application/wasm' : 'text/javascript')
        res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin')
        // The .mjs glue file is used as a nested worker's script (onnxruntime-web's
        // threaded WASM spawns a worker pool). Under COEP, worker/frame sub-resources
        // must declare their own COEP header, not just CORP, or Chromium blocks them
        // with "coep-frame-resource-needs-coep-header".
        res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
        createReadStream(filePath).pipe(res)
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), serveOrtWasmFiles()],
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
})
