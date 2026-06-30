import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'

// Mock onnxruntime-web before any imports that use it
vi.mock('onnxruntime-web', () => ({
  default: {},
  InferenceSession: {
    create: vi.fn().mockResolvedValue({
      inputNames: ['input_0'],
      outputNames: ['output_0'],
      handler: null,
    }),
  },
  Tensor: vi.fn().mockImplementation((type, data, shape) => ({ type, data, dims: shape })),
  env: { wasm: { wasmPaths: '' } },
}))

import { useOnnxWorker } from '../hooks/useOnnxWorker'
import { parseOnnxGraph } from '../lib/onnxParser'
import type { OnnxGraph, OnnxNode } from '../lib/onnxTypes'

const makeMockWorker = () => ({
  postMessage: vi.fn(),
  terminate: vi.fn(),
  onmessage: null as ((e: MessageEvent) => void) | null,
  onerror: null as ((e: unknown) => void) | null,
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
})

let mockWorker: ReturnType<typeof makeMockWorker>
let WorkerCtor: ReturnType<typeof vi.fn>

beforeEach(() => {
  mockWorker = makeMockWorker()
  // Must be a regular function (not an arrow function) so `new Worker()` works.
  WorkerCtor = vi.fn(function () { return mockWorker })
  vi.stubGlobal('Worker', WorkerCtor)
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

const emptyGraph: OnnxGraph = {
  nodes: [],
  edges: [],
  modelName: 'test',
  totalParams: 0,
  totalSizeMB: 0,
}

describe('useOnnxWorker lifecycle', () => {
  it('initializes with idle status, null graph, and null error', () => {
    const { result } = renderHook(() => useOnnxWorker())
    expect(result.current.status).toBe('idle')
    expect(result.current.graph).toBeNull()
    expect(result.current.error).toBeNull()
  })

  it('constructs a Worker on mount', () => {
    renderHook(() => useOnnxWorker())
    expect(WorkerCtor).toHaveBeenCalledTimes(1)
  })

  it('constructs the Worker with a URL referencing the onnx worker module', () => {
    renderHook(() => useOnnxWorker())
    const firstArg = WorkerCtor.mock.calls[0][0]
    expect(String(firstArg)).toContain('worker')
  })

  it('terminates the Worker on unmount', () => {
    const { unmount } = renderHook(() => useOnnxWorker())
    unmount()
    expect(mockWorker.terminate).toHaveBeenCalledTimes(1)
  })
})

describe('useOnnxWorker message handling', () => {
  it('sets status to loading after loadModel is called', () => {
    const { result } = renderHook(() => useOnnxWorker())
    act(() => {
      result.current.loadModel(new ArrayBuffer(8), 'model.onnx')
    })
    expect(result.current.status).toBe('loading')
  })

  it('sets status to ready and stores graph after MODEL_LOADED', () => {
    const { result } = renderHook(() => useOnnxWorker())
    act(() => {
      result.current.loadModel(new ArrayBuffer(8), 'model.onnx')
    })
    act(() => {
      mockWorker.onmessage?.({
        data: { type: 'MODEL_LOADED', payload: emptyGraph },
      } as MessageEvent)
    })
    expect(result.current.status).toBe('ready')
    expect(result.current.graph).toEqual(emptyGraph)
  })

  it('sets status to error and stores message after ERROR', () => {
    const { result } = renderHook(() => useOnnxWorker())
    act(() => {
      result.current.loadModel(new ArrayBuffer(8), 'model.onnx')
    })
    act(() => {
      mockWorker.onmessage?.({
        data: { type: 'ERROR', payload: 'failed to parse model' },
      } as MessageEvent)
    })
    expect(result.current.status).toBe('error')
    expect(result.current.error).toBe('failed to parse model')
  })
})

describe('WorkerCommand shape', () => {
  it('posts a LOAD_MODEL command with buffer and filename payload', () => {
    const { result } = renderHook(() => useOnnxWorker())
    act(() => {
      result.current.loadModel(new ArrayBuffer(8), 'model.onnx')
    })
    expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'LOAD_MODEL',
        payload: expect.objectContaining({
          buffer: expect.any(ArrayBuffer),
          filename: expect.any(String),
        }),
      }),
      // ArrayBuffers are typically transferred; allow either signature.
      expect.anything(),
    )
  })
})

describe('parseOnnxGraph contract', () => {
  const mockSession = {
    inputNames: ['input_0', 'input_1'],
    outputNames: ['output_0'],
    handler: null,
  }

  it('creates an Input node for each session input name', () => {
    const result = parseOnnxGraph(mockSession as never, 'model.onnx')
    const inputs = result.nodes.filter((n: OnnxNode) => n.opType === 'Input')
    expect(inputs).toHaveLength(mockSession.inputNames.length)
  })

  it('creates an Output node for each session output name', () => {
    const result = parseOnnxGraph(mockSession as never, 'model.onnx')
    const outputs = result.nodes.filter((n: OnnxNode) => n.opType === 'Output')
    expect(outputs).toHaveLength(mockSession.outputNames.length)
  })

  it('uses the provided filename as the model name', () => {
    const result = parseOnnxGraph(mockSession as never, 'model.onnx')
    expect(result.modelName).toBe('model.onnx')
  })

  it('never produces a node with a negative paramCount', () => {
    const result = parseOnnxGraph(mockSession as never, 'model.onnx')
    expect(result.nodes.every((n: OnnxNode) => n.paramCount >= 0)).toBe(true)
  })

  it('never produces a node with a negative estimatedSizeMB', () => {
    const result = parseOnnxGraph(mockSession as never, 'model.onnx')
    expect(result.nodes.every((n: OnnxNode) => n.estimatedSizeMB >= 0)).toBe(true)
  })

  it('reports a non-negative totalParams', () => {
    const result = parseOnnxGraph(mockSession as never, 'model.onnx')
    expect(result.totalParams).toBeGreaterThanOrEqual(0)
  })

  it('always returns a defined edges array', () => {
    const result = parseOnnxGraph(mockSession as never, 'model.onnx')
    expect(Array.isArray(result.edges)).toBe(true)
  })
})
