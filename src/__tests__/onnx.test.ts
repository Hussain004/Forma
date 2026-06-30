import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useOnnxWorker } from '../hooks/useOnnxWorker'
import { parseOnnxGraph } from '../lib/onnxParser'
import { parseOnnxProto, formatShape } from '../lib/onnxProtoParser'
import type { OnnxGraph } from '../lib/onnxTypes'

// ---- Minimal ONNX protobuf builder (for test fixtures only) ----

function encodeVarint(value: number): number[] {
  const bytes: number[] = []
  let v = value >>> 0
  do {
    let b = v & 0x7f
    v >>>= 7
    if (v !== 0) b |= 0x80
    bytes.push(b)
  } while (v !== 0)
  return bytes
}

function lenField(field: number, data: number[]): number[] {
  const tag = (field << 3) | 2
  return [...encodeVarint(tag), ...encodeVarint(data.length), ...data]
}

function strField(field: number, value: string): number[] {
  const bytes = Array.from(new TextEncoder().encode(value))
  return lenField(field, bytes)
}

function makeValueInfo(name: string): number[] { return strField(1, name) }

function makeNode(opType: string, inputs: string[], outputs: string[]): number[] {
  const bytes: number[] = []
  for (const s of inputs) bytes.push(...strField(1, s))
  for (const s of outputs) bytes.push(...strField(2, s))
  bytes.push(...strField(4, opType))
  return bytes
}

function makeInitializer(name: string, dims: number[], elemType = 1): number[] {
  const bytes: number[] = []
  // dims as packed varints (field 1, wire 2)
  const dimBytes: number[] = []
  for (const d of dims) dimBytes.push(...encodeVarint(d))
  bytes.push(...lenField(1, dimBytes))
  // elem_type (field 2, varint)
  bytes.push(0x10, ...encodeVarint(elemType))
  // name (field 8)
  bytes.push(...strField(8, name))
  return bytes
}

function makeGraph(
  nodes: number[][],
  inputs: string[],
  outputs: string[],
  initializers?: number[][],
): number[] {
  const bytes: number[] = []
  for (const n of nodes) bytes.push(...lenField(1, n))
  if (initializers) for (const i of initializers) bytes.push(...lenField(5, i))
  for (const s of inputs) bytes.push(...lenField(11, makeValueInfo(s)))
  for (const s of outputs) bytes.push(...lenField(12, makeValueInfo(s)))
  return bytes
}

function makeModel(graph: number[]): ArrayBuffer {
  // ir_version = 7 (field 1, varint)
  const irVersion = [0x08, 0x07]
  const bytes = [...irVersion, ...lenField(7, graph)]
  return new Uint8Array(bytes).buffer
}

// Convenience: a single-Relu model
function makeReluModel(): ArrayBuffer {
  const node = makeNode('Relu', ['x'], ['y'])
  const graph = makeGraph([node], ['x'], ['y'])
  return makeModel(graph)
}

// A model with two nodes and one weight initializer
function makeConvModel(): ArrayBuffer {
  const weight = makeInitializer('W', [64, 3, 3, 3]) // 64*3*3*3=1728 params
  const node = makeNode('Conv', ['x', 'W'], ['y'])
  const graph = makeGraph([node], ['x'], ['y'], [weight])
  return makeModel(graph)
}

// ---- parseOnnxProto unit tests ----

describe('parseOnnxProto', () => {
  it('returns empty graph for an empty buffer', () => {
    const result = parseOnnxProto(new ArrayBuffer(0))
    expect(result.nodes).toHaveLength(0)
    expect(result.initializers).toHaveLength(0)
  })

  it('parses a single Relu node correctly', () => {
    const result = parseOnnxProto(makeReluModel())
    expect(result.nodes).toHaveLength(1)
    expect(result.nodes[0].opType).toBe('Relu')
    expect(result.nodes[0].inputs).toContain('x')
    expect(result.nodes[0].outputs).toContain('y')
  })

  it('parses graph-level inputs and outputs', () => {
    const result = parseOnnxProto(makeReluModel())
    expect(result.inputs.some(vi => vi.name === 'x')).toBe(true)
    expect(result.outputs.some(vi => vi.name === 'y')).toBe(true)
  })

  it('parses initializer name, dims, and elem count', () => {
    const result = parseOnnxProto(makeConvModel())
    expect(result.initializers).toHaveLength(1)
    const init = result.initializers[0]
    expect(init.name).toBe('W')
    expect(init.dims).toEqual([64, 3, 3, 3])
    expect(init.elemCount).toBe(64 * 3 * 3 * 3)
  })

  it('computes sizeMB correctly for float32 (4 bytes per elem)', () => {
    const result = parseOnnxProto(makeConvModel())
    const init = result.initializers[0]
    const expectedMB = (init.elemCount * 4) / (1024 * 1024)
    expect(init.sizeMB).toBeCloseTo(expectedMB, 5)
  })

  it('handles multiple nodes without throwing', () => {
    const n1 = makeNode('Relu', ['x'], ['h'])
    const n2 = makeNode('Sigmoid', ['h'], ['y'])
    const buf = makeModel(makeGraph([n1, n2], ['x'], ['y']))
    const result = parseOnnxProto(buf)
    expect(result.nodes).toHaveLength(2)
    expect(result.nodes[0].opType).toBe('Relu')
    expect(result.nodes[1].opType).toBe('Sigmoid')
  })
})

describe('formatShape', () => {
  it('returns empty string for undefined shape', () => {
    expect(formatShape(undefined)).toBe('')
  })

  it('returns empty string for empty shape array', () => {
    expect(formatShape([])).toBe('')
  })

  it('formats numeric dims', () => {
    expect(formatShape([{ value: 1 }, { value: 3 }, { value: 224 }, { value: 224 }])).toBe('[1, 3, 224, 224]')
  })

  it('formats symbolic dims', () => {
    expect(formatShape([{ param: 'batch_size' }, { value: 512 }])).toBe('[batch_size, 512]')
  })
})

// ---- parseOnnxGraph integration tests ----

describe('parseOnnxGraph', () => {
  it('creates Input nodes for graph-level inputs', () => {
    const result = parseOnnxGraph(makeReluModel(), 'test.onnx')
    const inputs = result.nodes.filter(n => n.opType === 'Input')
    expect(inputs.length).toBeGreaterThanOrEqual(1)
  })

  it('creates Output nodes for graph-level outputs', () => {
    const result = parseOnnxGraph(makeReluModel(), 'test.onnx')
    const outputs = result.nodes.filter(n => n.opType === 'Output')
    expect(outputs.length).toBeGreaterThanOrEqual(1)
  })

  it('uses provided filename as the model name', () => {
    const result = parseOnnxGraph(makeReluModel(), 'my_model.onnx')
    expect(result.modelName).toBe('my_model.onnx')
  })

  it('counts parameters correctly from initializers', () => {
    const result = parseOnnxGraph(makeConvModel(), 'conv.onnx')
    const convNode = result.nodes.find(n => n.opType === 'Conv')
    expect(convNode).toBeDefined()
    expect(convNode!.paramCount).toBe(64 * 3 * 3 * 3)
  })

  it('accumulates totalParams from all compute nodes', () => {
    const result = parseOnnxGraph(makeConvModel(), 'conv.onnx')
    expect(result.totalParams).toBe(64 * 3 * 3 * 3)
  })

  it('never produces a node with a negative paramCount', () => {
    const result = parseOnnxGraph(makeConvModel(), 'conv.onnx')
    expect(result.nodes.every(n => n.paramCount >= 0)).toBe(true)
  })

  it('always returns a defined edges array', () => {
    const result = parseOnnxGraph(makeReluModel(), 'relu.onnx')
    expect(Array.isArray(result.edges)).toBe(true)
  })

  it('wires edges between input and compute nodes', () => {
    const result = parseOnnxGraph(makeReluModel(), 'relu.onnx')
    const inputNode = result.nodes.find(n => n.opType === 'Input')
    const reluNode = result.nodes.find(n => n.opType === 'Relu')
    expect(inputNode).toBeDefined()
    expect(reluNode).toBeDefined()
    const edge = result.edges.find(e => e.source === inputNode!.id && e.target === reluNode!.id)
    expect(edge).toBeDefined()
  })

  it('wires edges between compute and output nodes', () => {
    const result = parseOnnxGraph(makeReluModel(), 'relu.onnx')
    const reluNode = result.nodes.find(n => n.opType === 'Relu')
    const outputNode = result.nodes.find(n => n.opType === 'Output')
    expect(reluNode).toBeDefined()
    expect(outputNode).toBeDefined()
    const edge = result.edges.find(e => e.source === reluNode!.id && e.target === outputNode!.id)
    expect(edge).toBeDefined()
  })
})

// ---- useOnnxWorker lifecycle tests ----

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
  WorkerCtor = vi.fn(function () { return mockWorker })
  vi.stubGlobal('Worker', WorkerCtor)
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

const emptyGraph: OnnxGraph = { nodes: [], edges: [], modelName: 'test', totalParams: 0, totalSizeMB: 0 }

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
    act(() => { result.current.loadModel(new ArrayBuffer(8), 'model.onnx') })
    expect(result.current.status).toBe('loading')
  })

  it('sets status to ready and stores graph after MODEL_LOADED', () => {
    const { result } = renderHook(() => useOnnxWorker())
    act(() => { result.current.loadModel(new ArrayBuffer(8), 'model.onnx') })
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: emptyGraph } } as MessageEvent)
    })
    expect(result.current.status).toBe('ready')
    expect(result.current.graph).toEqual(emptyGraph)
  })

  it('sets status to error and stores message after ERROR', () => {
    const { result } = renderHook(() => useOnnxWorker())
    act(() => { result.current.loadModel(new ArrayBuffer(8), 'model.onnx') })
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'ERROR', payload: 'failed to parse' } } as MessageEvent)
    })
    expect(result.current.status).toBe('error')
    expect(result.current.error).toBe('failed to parse')
  })

  it('stores benchmarkResult after BENCHMARK_RESULT message', () => {
    const { result } = renderHook(() => useOnnxWorker())
    const bench = { avgMs: 5.0, minMs: 4.1, maxMs: 6.2, runs: 10 }
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'BENCHMARK_RESULT', payload: bench } } as MessageEvent)
    })
    expect(result.current.benchmarkResult).toEqual(bench)
    expect(result.current.status).toBe('ready')
  })
})

describe('WorkerCommand shape', () => {
  it('posts a LOAD_MODEL command with buffer and filename payload', () => {
    const { result } = renderHook(() => useOnnxWorker())
    act(() => { result.current.loadModel(new ArrayBuffer(8), 'model.onnx') })
    expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'LOAD_MODEL',
        payload: expect.objectContaining({ buffer: expect.any(ArrayBuffer), filename: expect.any(String) }),
      }),
      expect.anything(),
    )
  })

  it('posts a BENCHMARK command with runs payload', () => {
    const { result } = renderHook(() => useOnnxWorker())
    act(() => { result.current.runBenchmark(5) })
    expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'BENCHMARK', payload: { runs: 5 } }),
    )
  })
})
