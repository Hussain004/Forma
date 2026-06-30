import * as ort from 'onnxruntime-web'
import { parseOnnxGraph } from '../lib/onnxParser'
import type { OnnxGraph } from '../lib/onnxTypes'

ort.env.wasm.wasmPaths = '/'

type WorkerCommand =
  | { type: 'LOAD_MODEL'; payload: { buffer: ArrayBuffer; filename: string } }
  | { type: 'RUN_INFERENCE'; payload: { inputs: Record<string, Float32Array>; shapes: Record<string, number[]> } }

type WorkerResponse =
  | { type: 'MODEL_LOADED'; payload: OnnxGraph }
  | { type: 'INFERENCE_RESULT'; payload: { outputs: Record<string, Float32Array> } }
  | { type: 'ERROR'; payload: string }
  | { type: 'PROGRESS'; payload: { stage: string; percent: number } }

// The app tsconfig uses the DOM lib (not WebWorker), where the global `postMessage`
// resolves to Window's 2-3 arg signature. Alias the worker scope to keep these calls
// typed against our message protocol.
const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<WorkerCommand>) => void) | null
  postMessage: (message: WorkerResponse) => void
}

let session: ort.InferenceSession | null = null

ctx.onmessage = async (event: MessageEvent<WorkerCommand>) => {
  const cmd = event.data
  try {
    if (cmd.type === 'LOAD_MODEL') {
      ctx.postMessage({ type: 'PROGRESS', payload: { stage: 'Loading model', percent: 10 } })
      session = await ort.InferenceSession.create(cmd.payload.buffer)
      ctx.postMessage({ type: 'PROGRESS', payload: { stage: 'Parsing graph', percent: 60 } })
      const graph = parseOnnxGraph(session, cmd.payload.filename)
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
    }
  } catch (err) {
    ctx.postMessage({ type: 'ERROR', payload: (err as Error).message })
  }
}
