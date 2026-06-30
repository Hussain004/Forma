import { useEffect, useRef, useState, useCallback } from 'react'
import type { OnnxGraph } from '../lib/onnxTypes'

type Status = 'idle' | 'loading' | 'ready' | 'running' | 'error'

type WorkerResponse =
  | { type: 'MODEL_LOADED'; payload: OnnxGraph }
  | { type: 'INFERENCE_RESULT'; payload: { outputs: Record<string, Float32Array> } }
  | { type: 'ERROR'; payload: string }
  | { type: 'PROGRESS'; payload: { stage: string; percent: number } }

export function useOnnxWorker() {
  const workerRef = useRef<Worker | null>(null)
  const [graph, setGraph] = useState<OnnxGraph | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<{ stage: string; percent: number } | null>(null)

  const inferenceResolverRef = useRef<((outputs: Record<string, Float32Array>) => void) | null>(null)
  const inferenceRejecterRef = useRef<((err: Error) => void) | null>(null)

  useEffect(() => {
    const worker = new Worker(new URL('../workers/onnxWorker.ts', import.meta.url), { type: 'module' })
    workerRef.current = worker

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const msg = event.data
      if (msg.type === 'MODEL_LOADED') {
        setGraph(msg.payload)
        setStatus('ready')
        setProgress(null)
      } else if (msg.type === 'INFERENCE_RESULT') {
        inferenceResolverRef.current?.(msg.payload.outputs)
        inferenceResolverRef.current = null
        inferenceRejecterRef.current = null
        setStatus('ready')
      } else if (msg.type === 'ERROR') {
        setError(msg.payload)
        setStatus('error')
        inferenceRejecterRef.current?.(new Error(msg.payload))
        inferenceResolverRef.current = null
        inferenceRejecterRef.current = null
      } else if (msg.type === 'PROGRESS') {
        setProgress(msg.payload)
      }
    }

    return () => {
      worker.terminate()
    }
  }, [])

  const loadModel = useCallback((buffer: ArrayBuffer, filename: string) => {
    setStatus('loading')
    setError(null)
    setGraph(null)
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
    },
    []
  )

  return { loadModel, runInference, graph, status, error, progress }
}
