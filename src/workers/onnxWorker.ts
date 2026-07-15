import * as ort from 'onnxruntime-web'
import { parseOnnxGraph } from '../lib/onnxParser'
import { isTfliteBuffer, parseTfliteGraph } from '../lib/tfliteParser'
import { estimateInt8Size, compressionRatio } from '../lib/quantize'
import { writeModifiedOnnx, type StructuralOp } from '../lib/onnxProtoWriter'
import type { OnnxGraph } from '../lib/onnxTypes'

ort.env.wasm.wasmPaths = '/'

type WorkerCommand =
  | { type: 'LOAD_MODEL'; payload: { buffer: ArrayBuffer; filename: string } }
  | { type: 'RUN_INFERENCE'; payload: { inputs: Record<string, Float32Array>; shapes: Record<string, number[]> } }
  | { type: 'BENCHMARK'; payload: { runs: number } }
  | { type: 'EXPORT' }
  | { type: 'EXPORT_MODIFIED'; payload: { overrides: Map<number, Record<string, string | number>>; structuralOps: StructuralOp[] } }

type WorkerResponse =
  | { type: 'MODEL_LOADED'; payload: OnnxGraph }
  | { type: 'INFERENCE_RESULT'; payload: { outputs: Record<string, Float32Array> } }
  | { type: 'BENCHMARK_RESULT'; payload: { avgMs: number; medianMs: number; minMs: number; maxMs: number; runs: number } }
  | { type: 'QUANTIZE_ESTIMATE'; payload: { int8SizeMB: number; originalSizeMB: number; ratio: number } }
  | { type: 'EXPORT_RESULT'; payload: ArrayBuffer }
  | { type: 'ERROR'; payload: string }
  | { type: 'PROGRESS'; payload: { stage: string; percent: number } }

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<WorkerCommand>) => void) | null
  postMessage: (message: WorkerResponse, transfer?: Transferable[]) => void
}

let session: ort.InferenceSession | null = null
let benchmarkInputShapes: Record<string, number[]> = {}
let benchmarkInputTypes: Record<string, number> = {}
let exportBuffer: ArrayBuffer | null = null

function makeBenchmarkTensor(elemType: number, size: number, shape: number[]): ort.Tensor {
  switch (elemType) {
    case 7:  return new ort.Tensor('int64',   new BigInt64Array(size), shape)
    case 6:  return new ort.Tensor('int32',   new Int32Array(size),    shape)
    case 12: return new ort.Tensor('uint32',  new Uint32Array(size),   shape)
    case 13: return new ort.Tensor('uint64',  new BigUint64Array(size),shape)
    case 2:  return new ort.Tensor('uint8',   new Uint8Array(size),    shape)
    case 3:  return new ort.Tensor('int8',    new Int8Array(size),     shape)
    case 5:  return new ort.Tensor('int16',   new Int16Array(size),    shape)
    case 9:  return new ort.Tensor('bool',    new Uint8Array(size),    shape)
    case 11: return new ort.Tensor('float64', new Float64Array(size),  shape)
    case 10: return new ort.Tensor('float16', new Uint16Array(size),   shape)
    default: return new ort.Tensor('float32', new Float32Array(size),  shape)
  }
}

ctx.onmessage = async (event: MessageEvent<WorkerCommand>) => {
  const cmd = event.data
  try {
    if (cmd.type === 'LOAD_MODEL') {
      exportBuffer = null
      ctx.postMessage({ type: 'PROGRESS', payload: { stage: 'Parsing graph', percent: 10 } })

      const isTflite = isTfliteBuffer(cmd.payload.buffer)

      // Slice a copy for parsing; the original is transferred to InferenceSession (ONNX only)
      const bufferForParsing = cmd.payload.buffer.slice(0)

      // Keep a copy for the EXPORT command; the original is consumed by InferenceSession
      exportBuffer = cmd.payload.buffer.slice(0)

      // Parse graph topology from raw bytes (reliable, no WASM internals)
      const graph = isTflite
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

      // No TFLite runtime exists in this project (this is a read-only viewer for TFLite
      // models) -- session stays null, which the existing RUN_INFERENCE/BENCHMARK guards
      // below already handle safely; the UI simply doesn't offer those actions for it.
      if (!isTflite) {
        ctx.postMessage({ type: 'PROGRESS', payload: { stage: 'Loading WASM runtime', percent: 50 } })
        session = await ort.InferenceSession.create(cmd.payload.buffer)
      }

      ctx.postMessage({ type: 'PROGRESS', payload: { stage: 'Ready', percent: 100 } })
      ctx.postMessage({ type: 'MODEL_LOADED', payload: graph })

      const totalElemCount = graph.nodes.reduce((sum, n) => sum + n.paramCount, 0)
      const int8SizeMB = estimateInt8Size(totalElemCount)
      const ratio = compressionRatio(graph.totalSizeMB, totalElemCount)
      ctx.postMessage({ type: 'QUANTIZE_ESTIMATE', payload: { int8SizeMB, originalSizeMB: graph.totalSizeMB, ratio } })
    } else if (cmd.type === 'RUN_INFERENCE') {
      if (!session) throw new Error('No model loaded')
      const feeds: Record<string, ort.Tensor> = {}
      for (const [name, data] of Object.entries(cmd.payload.inputs)) {
        const shape = cmd.payload.shapes[name]
        feeds[name] = new ort.Tensor('float32', data, shape)
      }
      const results = await session.run(feeds)
      const outputs: Record<string, Float32Array> = {}
      for (const [name, tensor] of Object.entries(results)) {
        outputs[name] = tensor.data as Float32Array
      }
      ctx.postMessage({ type: 'INFERENCE_RESULT', payload: { outputs } })
    } else if (cmd.type === 'BENCHMARK') {
      if (!session) throw new Error('No model loaded')
      const runs = Math.max(1, Math.min(cmd.payload.runs, 50))

      const feeds: Record<string, ort.Tensor> = {}
      for (const name of session.inputNames) {
        const rawShape = benchmarkInputShapes[name] ?? [1]
        const shape = rawShape.map(d => (d < 1 ? 1 : d))
        const size = shape.reduce((a, b) => a * b, 1)
        feeds[name] = makeBenchmarkTensor(benchmarkInputTypes[name] ?? 1, size, shape)
      }

      // Two untimed warmup runs first -- the first real run after a model loads
      // pays for JIT warmup and allocator setup that later runs don't, which
      // otherwise skews avg/max toward a cold-start number nobody will see again.
      for (let i = 0; i < 2; i++) await session.run(feeds)

      const times: number[] = []
      for (let i = 0; i < runs; i++) {
        const t0 = performance.now()
        await session.run(feeds)
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
      ctx.postMessage({ type: 'EXPORT_RESULT', payload: patched }, [patched])
    }
  } catch (err) {
    ctx.postMessage({ type: 'ERROR', payload: (err as Error).message })
  }
}
