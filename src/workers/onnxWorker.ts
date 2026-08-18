import * as ort from 'onnxruntime-web'
import { parseOnnxGraph } from '../lib/onnxParser'
import { isTfliteBuffer, parseTfliteGraph } from '../lib/tfliteParser'
import { estimateInt8Size, compressionRatio } from '../lib/quantize'
import { writeModifiedOnnx, type StructuralOp } from '../lib/onnxProtoWriter'
import type { OnnxGraph } from '../lib/onnxTypes'
import type { SideRunResult, ValidationRunResult } from '../lib/validationUtils'

ort.env.wasm.wasmPaths = '/'

export interface ProvidedValidationInput {
  data: Float32Array
  shape: number[]
}

type WorkerCommand =
  | { type: 'LOAD_MODEL'; payload: { buffer: ArrayBuffer; filename: string } }
  | { type: 'RUN_INFERENCE'; payload: { inputs: Record<string, Float32Array>; shapes: Record<string, number[]> } }
  | { type: 'BENCHMARK'; payload: { runs: number } }
  | { type: 'EXPORT' }
  | { type: 'EXPORT_MODIFIED'; payload: { overrides: Map<number, Record<string, string | number>>; structuralOps: StructuralOp[] } }
  | {
      type: 'VALIDATE'
      payload: {
        overrides: Map<number, Record<string, string | number>>
        structuralOps: StructuralOp[]
        providedInputs: Record<string, ProvidedValidationInput> | null
      }
    }

type WorkerResponse =
  | { type: 'MODEL_LOADED'; payload: OnnxGraph }
  | { type: 'INFERENCE_RESULT'; payload: { outputs: Record<string, Float32Array> } }
  | { type: 'BENCHMARK_RESULT'; payload: { avgMs: number; medianMs: number; minMs: number; maxMs: number; runs: number } }
  | { type: 'QUANTIZE_ESTIMATE'; payload: { int8SizeMB: number; originalSizeMB: number; ratio: number } }
  | { type: 'EXPORT_RESULT'; payload: ArrayBuffer }
  | { type: 'VERIFY_RESULT'; payload: { valid: boolean; message?: string } }
  | { type: 'VALIDATION_RESULT'; payload: ValidationRunResult }
  // scope distinguishes a failed LOAD_MODEL (nothing usable exists yet, the
  // dropzone/error screen is the right response) from a failed operation on an
  // already-loaded model (benchmark/inference/export) -- the loaded graph and
  // any pending edits are still perfectly good and shouldn't be torn down for
  // a benchmark hiccup. See useOnnxWorker.ts for how each scope is handled.
  | { type: 'ERROR'; payload: string; scope: 'load' | 'operation' }
  | { type: 'PROGRESS'; payload: { stage: string; percent: number } }

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<WorkerCommand>) => void) | null
  postMessage: (message: WorkerResponse, transfer?: Transferable[]) => void
}

let session: ort.InferenceSession | null = null
let benchmarkInputShapes: Record<string, number[]> = {}
let benchmarkInputTypes: Record<string, number> = {}
let exportBuffer: ArrayBuffer | null = null
let isTfliteLoaded = false

// Session creation (WASM download + compile) used to block MODEL_LOADED, so the
// graph -- already fully parsed in milliseconds -- couldn't render until a ~20MB
// runtime finished loading. Now it's created lazily on first actual use
// (Benchmark or an inference run), from the same bytes retained for EXPORT.
async function ensureSession(): Promise<ort.InferenceSession> {
  if (session) return session
  if (isTfliteLoaded) throw new Error('No TFLite runtime available -- TFLite models are view-only')
  if (!exportBuffer) throw new Error('No model loaded')
  session = await ort.InferenceSession.create(exportBuffer.slice(0))
  return session
}

// `fill` is called once per element (row-major); the benchmark's zeroed
// inputs and validation's generated/provided inputs both go through this,
// so every dtype is only handled here once.
function makeTensor(elemType: number, shape: number[], fill: (i: number) => number): ort.Tensor {
  const size = shape.reduce((a, b) => a * b, 1)
  switch (elemType) {
    case 7: { const a = new BigInt64Array(size); for (let i = 0; i < size; i++) a[i] = BigInt(Math.round(fill(i))); return new ort.Tensor('int64', a, shape) }
    case 6: { const a = new Int32Array(size); for (let i = 0; i < size; i++) a[i] = Math.round(fill(i)); return new ort.Tensor('int32', a, shape) }
    case 12: { const a = new Uint32Array(size); for (let i = 0; i < size; i++) a[i] = Math.round(fill(i)); return new ort.Tensor('uint32', a, shape) }
    case 13: { const a = new BigUint64Array(size); for (let i = 0; i < size; i++) a[i] = BigInt(Math.max(0, Math.round(fill(i)))); return new ort.Tensor('uint64', a, shape) }
    case 2: { const a = new Uint8Array(size); for (let i = 0; i < size; i++) a[i] = Math.round(fill(i)); return new ort.Tensor('uint8', a, shape) }
    case 3: { const a = new Int8Array(size); for (let i = 0; i < size; i++) a[i] = Math.round(fill(i)); return new ort.Tensor('int8', a, shape) }
    case 5: { const a = new Int16Array(size); for (let i = 0; i < size; i++) a[i] = Math.round(fill(i)); return new ort.Tensor('int16', a, shape) }
    case 9: { const a = new Uint8Array(size); for (let i = 0; i < size; i++) a[i] = fill(i) !== 0 ? 1 : 0; return new ort.Tensor('bool', a, shape) }
    case 11: { const a = new Float64Array(size); for (let i = 0; i < size; i++) a[i] = fill(i); return new ort.Tensor('float64', a, shape) }
    case 10: { const a = new Uint16Array(size); return new ort.Tensor('float16', a, shape) }
    default: { const a = new Float32Array(size); for (let i = 0; i < size; i++) a[i] = fill(i); return new ort.Tensor('float32', a, shape) }
  }
}

// Deterministic PRNG (mulberry32) so behavioral-validation smoke-test inputs
// are reproducible across runs -- not representative real data, just
// consistent enough that two runs of the same edit are comparable.
function mulberry32(seed: number): () => number {
  let state = seed
  return () => {
    state |= 0
    state = (state + 0x6d2b79f5) | 0
    let t = Math.imul(state ^ (state >>> 15), 1 | state)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function seedFromName(name: string): number {
  let h = 0
  for (let i = 0; i < name.length; i++) h = (Math.imul(h, 31) + name.charCodeAt(i)) | 0
  return h || 1
}

// Builds one feed tensor per graph input: the caller's own array/shape if
// provided (from a loaded NPY/NPZ file), otherwise deterministic pseudo-random
// values over the model's declared shape (symbolic dims resolved to 1, same
// as the benchmark's zeroed tensors) -- a smoke test, not real data.
function buildValidationFeeds(providedInputs: Record<string, ProvidedValidationInput> | null): Record<string, ort.Tensor> {
  const names = new Set([...Object.keys(benchmarkInputShapes), ...Object.keys(providedInputs ?? {})])
  const feeds: Record<string, ort.Tensor> = {}
  for (const name of names) {
    const elemType = benchmarkInputTypes[name] ?? 1
    const provided = providedInputs?.[name]
    if (provided) {
      feeds[name] = makeTensor(elemType, provided.shape, (i) => provided.data[i] ?? 0)
    } else {
      const shape = (benchmarkInputShapes[name] ?? [1]).map((d) => (d < 1 ? 1 : d))
      const rng = mulberry32(seedFromName(name))
      feeds[name] = makeTensor(elemType, shape, () => rng() * 2 - 1)
    }
  }
  return feeds
}

async function runValidationSide(bytes: ArrayBuffer, feeds: Record<string, ort.Tensor>): Promise<SideRunResult> {
  const result: SideRunResult = { loaded: false, inferenceOk: false, outputs: {} }
  let s: ort.InferenceSession | null = null
  try {
    s = await ort.InferenceSession.create(bytes)
    result.loaded = true
  } catch (loadErr) {
    result.error = (loadErr as Error).message
    return result
  }
  try {
    const sessionFeeds: Record<string, ort.Tensor> = {}
    for (const name of s.inputNames) if (feeds[name]) sessionFeeds[name] = feeds[name]
    const out = await s.run(sessionFeeds)
    result.inferenceOk = true
    for (const [name, tensor] of Object.entries(out)) {
      result.outputs[name] = { shape: (tensor.dims as number[]).slice(), data: (tensor.data as Float32Array).slice() }
    }
  } catch (inferErr) {
    result.error = (inferErr as Error).message
  } finally {
    await s.release()
  }
  return result
}

ctx.onmessage = async (event: MessageEvent<WorkerCommand>) => {
  const cmd = event.data
  try {
    if (cmd.type === 'LOAD_MODEL') {
      exportBuffer = null
      session = null
      ctx.postMessage({ type: 'PROGRESS', payload: { stage: 'Parsing graph', percent: 10 } })

      isTfliteLoaded = isTfliteBuffer(cmd.payload.buffer)

      // Slice a copy for parsing; a second copy is retained for EXPORT/benchmark below.
      const bufferForParsing = cmd.payload.buffer.slice(0)
      exportBuffer = cmd.payload.buffer.slice(0)

      // Parse graph topology from raw bytes (reliable, no WASM internals). This is
      // the only thing MODEL_LOADED waits on now -- no InferenceSession creation,
      // so the graph renders immediately instead of blocking on a ~20MB WASM
      // runtime download+compile that only Benchmark/inference actually need.
      const graph = isTfliteLoaded
        ? parseTfliteGraph(bufferForParsing, cmd.payload.filename)
        : parseOnnxGraph(bufferForParsing, cmd.payload.filename)

      // Store parsed input shapes and types for benchmark (symbolic dims -> 1)
      benchmarkInputShapes = {}
      benchmarkInputTypes = {}
      for (const vi of graph.graphInputs ?? []) {
        if (vi.name && vi.shape && vi.shape.length > 0) {
          benchmarkInputShapes[vi.name] = vi.shape.map(d => ('value' in d ? (d.value || 1) : 1))
        }
        if (vi.name) benchmarkInputTypes[vi.name] = vi.elemType ?? 1
      }

      ctx.postMessage({ type: 'PROGRESS', payload: { stage: 'Ready', percent: 100 } })
      ctx.postMessage({ type: 'MODEL_LOADED', payload: graph })

      const totalElemCount = graph.nodes.reduce((sum, n) => sum + n.paramCount, 0)
      const int8SizeMB = estimateInt8Size(totalElemCount)
      const ratio = compressionRatio(graph.totalSizeMB, totalElemCount)
      ctx.postMessage({ type: 'QUANTIZE_ESTIMATE', payload: { int8SizeMB, originalSizeMB: graph.totalSizeMB, ratio } })
    } else if (cmd.type === 'RUN_INFERENCE') {
      const s = await ensureSession()
      const feeds: Record<string, ort.Tensor> = {}
      for (const [name, data] of Object.entries(cmd.payload.inputs)) {
        const shape = cmd.payload.shapes[name]
        feeds[name] = new ort.Tensor('float32', data, shape)
      }
      const results = await s.run(feeds)
      const outputs: Record<string, Float32Array> = {}
      for (const [name, tensor] of Object.entries(results)) {
        outputs[name] = tensor.data as Float32Array
      }
      ctx.postMessage({ type: 'INFERENCE_RESULT', payload: { outputs } })
    } else if (cmd.type === 'BENCHMARK') {
      const s = await ensureSession()
      const runs = Math.max(1, Math.min(cmd.payload.runs, 50))

      const feeds: Record<string, ort.Tensor> = {}
      for (const name of s.inputNames) {
        const shape = (benchmarkInputShapes[name] ?? [1]).map(d => (d < 1 ? 1 : d))
        feeds[name] = makeTensor(benchmarkInputTypes[name] ?? 1, shape, () => 0)
      }

      // Two untimed warmup runs first -- the first real run after a model loads
      // pays for JIT warmup and allocator setup that later runs don't, which
      // otherwise skews avg/max toward a cold-start number nobody will see again.
      for (let i = 0; i < 2; i++) await s.run(feeds)

      const times: number[] = []
      for (let i = 0; i < runs; i++) {
        const t0 = performance.now()
        await s.run(feeds)
        times.push(performance.now() - t0)
      }

      const sorted = [...times].sort((a, b) => a - b)
      const mid = Math.floor(sorted.length / 2)
      const medianMs = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
      const avgMs = times.reduce((a, b) => a + b, 0) / times.length
      const minMs = Math.min(...times)
      const maxMs = Math.max(...times)
      ctx.postMessage({ type: 'BENCHMARK_RESULT', payload: { avgMs, medianMs, minMs, maxMs, runs } })
    } else if (cmd.type === 'EXPORT') {
      if (!exportBuffer) throw new Error('No model loaded')
      // Transfer the buffer back to the main thread. Slice it so the worker retains a copy.
      const toSend = exportBuffer.slice(0)
      ctx.postMessage({ type: 'EXPORT_RESULT', payload: toSend }, [toSend])
    } else if (cmd.type === 'EXPORT_MODIFIED') {
      if (!exportBuffer) throw new Error('No model loaded')
      const patched = writeModifiedOnnx(exportBuffer, cmd.payload.overrides, cmd.payload.structuralOps)
      // Send the bytes immediately -- the download shouldn't wait on the
      // verification below, which may need to pull the WASM runtime first.
      const toSend = patched.slice(0)
      ctx.postMessage({ type: 'EXPORT_RESULT', payload: toSend }, [toSend])
      // Verify-roundtrip: prove the exported bytes actually load in
      // onnxruntime rather than just claiming structural validity. Uses a
      // throwaway session so the active session for the loaded model (if
      // any) is untouched. A failure here is a warning, not an error --
      // the file already downloaded, this tells the user what a runtime
      // will say about it.
      try {
        const verifySession = await ort.InferenceSession.create(patched)
        await verifySession.release()
        ctx.postMessage({ type: 'VERIFY_RESULT', payload: { valid: true } })
      } catch (verifyErr) {
        ctx.postMessage({ type: 'VERIFY_RESULT', payload: { valid: false, message: (verifyErr as Error).message } })
      }
    } else if (cmd.type === 'VALIDATE') {
      if (!exportBuffer) throw new Error('No model loaded')
      if (isTfliteLoaded) throw new Error('Behavioral validation is only available for ONNX models')

      const patched = writeModifiedOnnx(exportBuffer, cmd.payload.overrides, cmd.payload.structuralOps)
      const feeds = buildValidationFeeds(cmd.payload.providedInputs)

      // Two throwaway sessions, one per side, so the active `session` used by
      // Benchmark/inference (if any) is untouched by either. Sequential, not
      // concurrent -- onnxruntime-web's WASM backend does not support two
      // sessions initializing at once from the same worker.
      const original = await runValidationSide(exportBuffer.slice(0), feeds)
      const modified = await runValidationSide(patched, feeds)
      const result: ValidationRunResult = {
        inputSource: cmd.payload.providedInputs ? 'provided' : 'generated',
        original,
        modified,
      }
      ctx.postMessage({ type: 'VALIDATION_RESULT', payload: result })
    }
  } catch (err) {
    ctx.postMessage({ type: 'ERROR', payload: (err as Error).message, scope: cmd.type === 'LOAD_MODEL' ? 'load' : 'operation' })
  }
}
