import * as ort from 'onnxruntime-web'
import { parseOnnxGraph } from '../lib/onnxParser'
import type { OnnxGraph } from '../lib/onnxTypes'

ort.env.wasm.wasmPaths = '/'

type WorkerCommand =
  | { type: 'LOAD_MODEL'; payload: { buffer: ArrayBuffer; filename: string } }
  | { type: 'RUN_INFERENCE'; payload: { inputs: Record<string, Float32Array>; shapes: Record<string, number[]> } }
  | { type: 'BENCHMARK'; payload: { runs: number } }

type WorkerResponse =
  | { type: 'MODEL_LOADED'; payload: OnnxGraph }
  | { type: 'INFERENCE_RESULT'; payload: { outputs: Record<string, Float32Array> } }
  | { type: 'BENCHMARK_RESULT'; payload: { avgMs: number; minMs: number; maxMs: number; runs: number } }
  | { type: 'ERROR'; payload: string }
  | { type: 'PROGRESS'; payload: { stage: string; percent: number } }

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<WorkerCommand>) => void) | null
  postMessage: (message: WorkerResponse) => void
}

let session: ort.InferenceSession | null = null

ctx.onmessage = async (event: MessageEvent<WorkerCommand>) => {
  const cmd = event.data
  try {
    if (cmd.type === 'LOAD_MODEL') {
      ctx.postMessage({ type: 'PROGRESS', payload: { stage: 'Parsing graph', percent: 10 } })

      // Slice a copy for parsing; the original is transferred to InferenceSession
      const bufferForParsing = cmd.payload.buffer.slice(0)

      // Parse graph topology from raw protobuf (reliable, no WASM internals)
      const graph = parseOnnxGraph(bufferForParsing, cmd.payload.filename)

      ctx.postMessage({ type: 'PROGRESS', payload: { stage: 'Loading WASM runtime', percent: 50 } })

      // Create inference session from the transferred buffer
      session = await ort.InferenceSession.create(cmd.payload.buffer)

      ctx.postMessage({ type: 'PROGRESS', payload: { stage: 'Ready', percent: 100 } })
      ctx.postMessage({ type: 'MODEL_LOADED', payload: graph })
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

      // Build dummy float32 inputs using the session's metadata
      const feeds: Record<string, ort.Tensor> = {}
      for (const name of session.inputNames) {
        // Default to a flat tensor of 1 element if shapes are unknown
        feeds[name] = new ort.Tensor('float32', new Float32Array([0]), [1])
      }

      const times: number[] = []
      for (let i = 0; i < runs; i++) {
        const t0 = performance.now()
        await session.run(feeds)
        times.push(performance.now() - t0)
      }

      const avgMs = times.reduce((a, b) => a + b, 0) / times.length
      const minMs = Math.min(...times)
      const maxMs = Math.max(...times)
      ctx.postMessage({ type: 'BENCHMARK_RESULT', payload: { avgMs, minMs, maxMs, runs } })
    }
  } catch (err) {
    ctx.postMessage({ type: 'ERROR', payload: (err as Error).message })
  }
}
