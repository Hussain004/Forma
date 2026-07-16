import { useEffect, useRef, useState, useCallback } from 'react'
import type { OnnxGraph } from '../lib/onnxTypes'
import type { StructuralOp } from '../lib/onnxProtoWriter'

type Status = 'idle' | 'loading' | 'ready' | 'running' | 'benchmarking' | 'exporting' | 'error'

export interface BenchmarkResult {
  avgMs: number
  medianMs: number
  minMs: number
  maxMs: number
  runs: number
}

export interface QuantizeEstimate {
  int8SizeMB: number
  originalSizeMB: number
  ratio: number
}

type WorkerResponse =
  | { type: 'MODEL_LOADED'; payload: OnnxGraph }
  | { type: 'INFERENCE_RESULT'; payload: { outputs: Record<string, Float32Array> } }
  | { type: 'BENCHMARK_RESULT'; payload: BenchmarkResult }
  | { type: 'QUANTIZE_ESTIMATE'; payload: QuantizeEstimate }
  | { type: 'EXPORT_RESULT'; payload: ArrayBuffer }
  | { type: 'VERIFY_RESULT'; payload: { valid: boolean; message?: string } }
  | { type: 'ERROR'; payload: string; scope: 'load' | 'operation' }
  | { type: 'PROGRESS'; payload: { stage: string; percent: number } }

export function useOnnxWorker() {
  const workerRef = useRef<Worker | null>(null)
  const [graph, setGraph] = useState<OnnxGraph | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)
  // A failed benchmark/inference/export shouldn't tear down an already-loaded
  // graph the way a failed initial load should -- this is the surface for
  // those, kept distinct from `error` (the full-screen load-failure state).
  // `at` forces a distinct object on every event (not just every distinct
  // message) so retrying the exact same failure twice in a row still
  // re-triggers whatever effect is watching this value.
  const [operationError, setOperationError] = useState<{ message: string; at: number } | null>(null)
  // Result of the post-export verify-roundtrip (loading the exported bytes
  // through a throwaway onnxruntime session in the worker). Same `at`
  // convention as operationError.
  const [verifyResult, setVerifyResult] = useState<{ valid: boolean; message?: string; at: number } | null>(null)
  const [progress, setProgress] = useState<{ stage: string; percent: number } | null>(null)
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResult | null>(null)
  const [quantizeEstimate, setQuantizeEstimate] = useState<QuantizeEstimate | null>(null)

  const inferenceResolverRef = useRef<((outputs: Record<string, Float32Array>) => void) | null>(null)
  const inferenceRejecterRef = useRef<((err: Error) => void) | null>(null)
  const benchmarkResolverRef = useRef<((r: BenchmarkResult) => void) | null>(null)
  const benchmarkRejecterRef = useRef<((err: Error) => void) | null>(null)
  const exportResolve = useRef<((buf: ArrayBuffer) => void) | null>(null)
  const exportReject = useRef<((err: Error) => void) | null>(null)

  useEffect(() => {
    const worker = new Worker(new URL('../workers/onnxWorker.ts', import.meta.url), { type: 'module' })
    workerRef.current = worker

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const msg = event.data
      if (msg.type === 'MODEL_LOADED') {
        setGraph(msg.payload)
        setStatus('ready')
        setProgress(null)
        setBenchmarkResult(null)
      } else if (msg.type === 'QUANTIZE_ESTIMATE') {
        setQuantizeEstimate(msg.payload)
      } else if (msg.type === 'EXPORT_RESULT') {
        if (exportResolve.current) {
          exportResolve.current(msg.payload)
          exportResolve.current = null
          exportReject.current = null
        }
        setStatus('ready')
      } else if (msg.type === 'VERIFY_RESULT') {
        setVerifyResult({ ...msg.payload, at: Date.now() })
      } else if (msg.type === 'INFERENCE_RESULT') {
        inferenceResolverRef.current?.(msg.payload.outputs)
        inferenceResolverRef.current = null
        inferenceRejecterRef.current = null
        setStatus('ready')
      } else if (msg.type === 'BENCHMARK_RESULT') {
        setBenchmarkResult(msg.payload)
        benchmarkResolverRef.current?.(msg.payload)
        benchmarkResolverRef.current = null
        benchmarkRejecterRef.current = null
        setStatus('ready')
      } else if (msg.type === 'ERROR') {
        if (exportReject.current) {
          exportReject.current(new Error(msg.payload))
          exportResolve.current = null
          exportReject.current = null
        }
        if (msg.scope === 'load') {
          setError(msg.payload)
          setStatus('error')
        } else {
          setOperationError({ message: msg.payload, at: Date.now() })
          setStatus('ready')
        }
        inferenceRejecterRef.current?.(new Error(msg.payload))
        benchmarkRejecterRef.current?.(new Error(msg.payload))
        inferenceResolverRef.current = null
        inferenceRejecterRef.current = null
        benchmarkResolverRef.current = null
        benchmarkRejecterRef.current = null
      } else if (msg.type === 'PROGRESS') {
        setProgress(msg.payload)
      }
    }

    return () => { worker.terminate() }
  }, [])

  const loadModel = useCallback((buffer: ArrayBuffer, filename: string) => {
    if (exportReject.current) {
      exportReject.current(new Error('Model replaced'))
      exportResolve.current = null
      exportReject.current = null
    }
    setStatus('loading')
    setError(null)
    setOperationError(null)
    setVerifyResult(null)
    setGraph(null)
    setBenchmarkResult(null)
    setQuantizeEstimate(null)
    workerRef.current?.postMessage({ type: 'LOAD_MODEL', payload: { buffer, filename } }, [buffer])
  }, [])

  const runInference = useCallback(
    (inputs: Record<string, Float32Array>, shapes: Record<string, number[]>): Promise<Record<string, Float32Array>> => {
      return new Promise((resolve, reject) => {
        inferenceResolverRef.current = resolve
        inferenceRejecterRef.current = reject
        setStatus('running')
        workerRef.current?.postMessage({ type: 'RUN_INFERENCE', payload: { inputs, shapes } })
      })
    }, []
  )

  const runBenchmark = useCallback((runs = 10): Promise<BenchmarkResult> => {
    return new Promise((resolve, reject) => {
      benchmarkResolverRef.current = resolve
      benchmarkRejecterRef.current = reject
      setStatus('benchmarking')
      workerRef.current?.postMessage({ type: 'BENCHMARK', payload: { runs } })
    })
  }, [])

  const exportModel = useCallback((): Promise<ArrayBuffer> => {
    return new Promise((resolve, reject) => {
      if (!workerRef.current) { reject(new Error('No worker')); return }
      exportResolve.current = resolve
      exportReject.current = reject
      setStatus('exporting')
      workerRef.current.postMessage({ type: 'EXPORT' })
    })
  }, [])

  const exportModifiedModel = useCallback((overrides: Map<number, Record<string, string | number>>, structuralOps: StructuralOp[] = []): Promise<ArrayBuffer> => {
    return new Promise((resolve, reject) => {
      if (!workerRef.current) { reject(new Error('No worker')); return }
      exportResolve.current = resolve
      exportReject.current = reject
      setStatus('exporting')
      workerRef.current.postMessage({ type: 'EXPORT_MODIFIED', payload: { overrides, structuralOps } })
    })
  }, [])

  return { loadModel, runInference, runBenchmark, exportModel, exportModifiedModel, graph, status, error, operationError, verifyResult, progress, benchmarkResult, quantizeEstimate }
}
