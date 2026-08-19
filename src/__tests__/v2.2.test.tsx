import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import { parseOnnxProto } from '../lib/onnxProtoParser'
import { extractSubgraph, SubgraphExtractionError } from '../lib/subgraphExtractor'
import { isConnectedSubgraph } from '../lib/graphUtils'
import type { OnnxGraph } from '../lib/onnxTypes'

// ---- Minimal ONNX protobuf builder (for test fixtures only) -----------------
// Independent of onnxProtoWriter.ts/subgraphExtractor.ts -- hand-encodes real
// wire bytes so extraction is tested as a black box, not against its own
// encoder. Mirrors the fixture builder in v1.1.test.ts.

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

function varintField(field: number, value: number): number[] {
  const tag = (field << 3) | 0
  return [...encodeVarint(tag), ...encodeVarint(value)]
}

function strField(field: number, value: string): number[] {
  return lenField(field, Array.from(new TextEncoder().encode(value)))
}

function makeNode(opType: string, inputs: string[], outputs: string[], name: string): number[] {
  const bytes: number[] = []
  for (const s of inputs) bytes.push(...strField(1, s))
  for (const s of outputs) bytes.push(...strField(2, s))
  bytes.push(...strField(3, name))
  bytes.push(...strField(4, opType))
  return bytes
}

function makeInitializer(name: string, dims: number[], elemType = 1): number[] {
  const dimBytes = dims.flatMap((d) => encodeVarint(d))
  return [...lenField(1, dimBytes), ...varintField(2, elemType), ...strField(8, name)]
}

// dim: [4] value-dims, or omit for a typeless/unranked value info.
function makeValueInfo(name: string, elemType?: number, dims?: number[]): number[] {
  const bytes = [...strField(1, name)]
  if (elemType !== undefined) {
    const shapeBytes = (dims ?? []).flatMap((d) => lenField(1, varintField(1, d)))
    const tensorType = [...varintField(1, elemType), ...(dims ? lenField(2, shapeBytes) : [])]
    bytes.push(...lenField(2, lenField(1, tensorType)))
  }
  return bytes
}

function makeGraph(nodes: number[][], inputs: number[][], outputs: number[][], initializers: number[][]): number[] {
  const bytes: number[] = []
  for (const n of nodes) bytes.push(...lenField(1, n))
  for (const i of initializers) bytes.push(...lenField(5, i))
  for (const s of inputs) bytes.push(...lenField(11, s))
  for (const s of outputs) bytes.push(...lenField(12, s))
  return bytes
}

function makeModel(graph: number[], irVersion = 8, producerName = 'test-producer'): ArrayBuffer {
  const bytes = [...varintField(1, irVersion), ...strField(2, producerName), ...lenField(7, graph)]
  return new Uint8Array(bytes).buffer
}

// x -[Conv,W]-> y -[Relu]-> z -[MaxPool]-> p -[Reshape,shape_const]-> f -[Gemm,Wg,Bg]-> out
function makeChainFixture(): ArrayBuffer {
  const conv = makeNode('Conv', ['x', 'W'], ['y'], 'conv0')
  const relu = makeNode('Relu', ['y'], ['z'], 'relu0')
  const pool = makeNode('MaxPool', ['z'], ['p'], 'pool0')
  const reshape = makeNode('Reshape', ['p', 'shape_const'], ['f'], 'reshape0')
  const gemm = makeNode('Gemm', ['f', 'Wg', 'Bg'], ['out'], 'gemm0')
  const graph = makeGraph(
    [conv, relu, pool, reshape, gemm],
    [makeValueInfo('x', 1, [1, 3, 8, 8])],
    [makeValueInfo('out', 1, [1, 4])],
    [
      makeInitializer('W', [4, 3, 3, 3]),
      makeInitializer('shape_const', [2], 7),
      makeInitializer('Wg', [4, 4]),
      makeInitializer('Bg', [4]),
    ],
  )
  return makeModel(graph)
}

describe('v2.2 subgraphExtractor', () => {
  it('promotes an intermediate boundary tensor to an unranked graph input and output', () => {
    const extracted = extractSubgraph(makeChainFixture(), new Set([1, 2])) // Relu, MaxPool
    const parsed = parseOnnxProto(extracted)
    expect(parsed.nodes.map((n) => n.opType)).toEqual(['Relu', 'MaxPool'])
    expect(parsed.initializers).toEqual([])
    expect(parsed.inputs).toEqual([{ name: 'y', shape: undefined, elemType: 1 }])
    expect(parsed.outputs).toEqual([{ name: 'p', shape: undefined, elemType: 1 }])
  })

  it('reuses the original ValueInfoProto (with real shape) for a boundary tensor that was already a graph input', () => {
    const extracted = extractSubgraph(makeChainFixture(), new Set([0])) // Conv only
    const parsed = parseOnnxProto(extracted)
    expect(parsed.nodes.map((n) => n.opType)).toEqual(['Conv'])
    expect(parsed.initializers.map((i) => i.name)).toEqual(['W'])
    expect(parsed.inputs).toEqual([{ name: 'x', shape: [{ value: 1 }, { value: 3 }, { value: 8 }, { value: 8 }], elemType: 1 }])
    expect(parsed.outputs).toEqual([{ name: 'y', shape: undefined, elemType: 1 }]) // synthesized: y has no declared type
  })

  it('extracting the whole chain reproduces the original graph inputs/outputs and every initializer', () => {
    const extracted = extractSubgraph(makeChainFixture(), new Set([0, 1, 2, 3, 4]))
    const parsed = parseOnnxProto(extracted)
    expect(parsed.nodes).toHaveLength(5)
    expect(parsed.initializers.map((i) => i.name).sort()).toEqual(['Bg', 'W', 'Wg', 'shape_const'])
    expect(parsed.inputs).toEqual([{ name: 'x', shape: [{ value: 1 }, { value: 3 }, { value: 8 }, { value: 8 }], elemType: 1 }])
    expect(parsed.outputs).toEqual([{ name: 'out', shape: [{ value: 1 }, { value: 4 }], elemType: 1 }])
  })

  it('preserves model-level metadata untouched', () => {
    const extracted = extractSubgraph(makeChainFixture(), new Set([0]))
    const parsed = parseOnnxProto(extracted)
    expect(parsed.metadata?.irVersion).toBe(8)
    expect(parsed.metadata?.producerName).toBe('test-producer')
  })

  it('rejects a selection referencing a node index that does not exist', () => {
    expect(() => extractSubgraph(makeChainFixture(), new Set([0, 99]))).toThrow(SubgraphExtractionError)
  })

  it('rejects an empty selection', () => {
    expect(() => extractSubgraph(makeChainFixture(), new Set())).toThrow(SubgraphExtractionError)
  })
})

describe('v2.2 isConnectedSubgraph', () => {
  const graph = {
    edges: [
      { id: 'a-b', source: 'a', target: 'b' },
      { id: 'b-c', source: 'b', target: 'c' },
      { id: 'x-y', source: 'x', target: 'y' },
    ],
  }

  it('treats a single node and an empty selection as trivially connected', () => {
    expect(isConnectedSubgraph(graph, new Set())).toBe(true)
    expect(isConnectedSubgraph(graph, new Set(['a']))).toBe(true)
  })

  it('accepts a chain regardless of edge direction relative to selection order', () => {
    expect(isConnectedSubgraph(graph, new Set(['a', 'b', 'c']))).toBe(true)
  })

  it('rejects two disjoint components', () => {
    expect(isConnectedSubgraph(graph, new Set(['a', 'b', 'x', 'y']))).toBe(false)
  })
})

// ---- App wiring --------------------------------------------------------------

const appGraph: OnnxGraph = {
  modelName: 'chain.onnx',
  totalParams: 0,
  totalSizeMB: 0,
  nodes: [
    { id: 'input_0', opType: 'Input', inputs: [], outputs: ['x'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'node_0_Conv', opType: 'Conv', inputs: ['x'], outputs: ['y'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'node_1_Relu', opType: 'Relu', inputs: ['y'], outputs: ['z'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'node_2_Gemm', opType: 'Gemm', inputs: ['z'], outputs: ['out'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'output_0', opType: 'Output', inputs: ['out'], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
  ],
  edges: [
    { id: 'input_0->node_0_Conv@x', source: 'input_0', target: 'node_0_Conv', label: 'x' },
    { id: 'node_0_Conv->node_1_Relu@y', source: 'node_0_Conv', target: 'node_1_Relu', label: 'y' },
    { id: 'node_1_Relu->node_2_Gemm@z', source: 'node_1_Relu', target: 'node_2_Gemm', label: 'z' },
    { id: 'node_2_Gemm->output_0@out', source: 'node_2_Gemm', target: 'output_0', label: 'out' },
  ],
}

const graphCanvasState = vi.hoisted(() => ({
  props: null as null | {
    onNodeSelect?: (id: string) => void
    onNodeCtrlClick?: (id: string) => void
  },
}))

vi.mock('../components/GraphCanvas', () => ({
  GraphCanvas: (props: typeof graphCanvasState.props) => {
    graphCanvasState.props = props
    return null
  },
}))

const makeMockWorker = () => ({
  postMessage: vi.fn(),
  terminate: vi.fn(),
  onmessage: null as ((event: MessageEvent) => void) | null,
  onerror: null as ((event: unknown) => void) | null,
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
})

let mockWorker: ReturnType<typeof makeMockWorker>

beforeEach(() => {
  mockWorker = makeMockWorker()
  vi.stubGlobal('Worker', vi.fn(function () { return mockWorker }))
  graphCanvasState.props = null
  localStorage.clear()
  window.history.replaceState(null, '', '/')
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
  window.history.replaceState(null, '', '/')
})

function loadGraph(g: OnnxGraph) {
  act(() => {
    mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: g } } as MessageEvent)
  })
}

// Two separate act() calls, not one: onNodeCtrlClick's handler closes over
// selectedNodeIds from its own render, so calling it back-to-back with
// onNodeSelect inside a single act() would read the pre-selection Set (React
// batches the state update; the mock's props aren't refreshed until a render
// actually happens in between).
function selectTwo(first: string, second: string) {
  act(() => { graphCanvasState.props?.onNodeSelect?.(first) })
  act(() => { graphCanvasState.props?.onNodeCtrlClick?.(second) })
}

describe('v2.2 extract-repro UI', () => {
  it('sends the selected original-node indices and downloads the result on success', async () => {
    render(<App />)
    loadGraph(appGraph)

    selectTwo('node_0_Conv', 'node_1_Relu')

    fireEvent.click(screen.getByTestId('extract-repro-button'))
    await waitFor(() => expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'EXTRACT_SUBGRAPH', payload: { selectedIndices: expect.arrayContaining([0, 1]) } }),
    ))

    const bytes = new TextEncoder().encode('fake-onnx-bytes').buffer
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'EXPORT_RESULT', payload: bytes } } as MessageEvent)
    })
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'VERIFY_RESULT', payload: { valid: true } } } as MessageEvent)
    })
    expect(await screen.findByTestId('announcement')).toHaveTextContent(/loads cleanly in onnxruntime/i)
  })

  it('rejects a disconnected selection before contacting the worker', async () => {
    render(<App />)
    loadGraph(appGraph)

    selectTwo('node_0_Conv', 'node_2_Gemm')

    fireEvent.click(screen.getByTestId('extract-repro-button'))
    expect(await screen.findByTestId('announcement')).toHaveTextContent(/connected subgraph/i)
    expect(mockWorker.postMessage).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: 'EXTRACT_SUBGRAPH' }),
      expect.anything(),
    )
  })

  it('rejects a selection that includes a non-original node', async () => {
    render(<App />)
    loadGraph(appGraph)

    selectTwo('node_0_Conv', 'input_0')

    fireEvent.click(screen.getByTestId('extract-repro-button'))
    expect(await screen.findByTestId('announcement')).toHaveTextContent(/original model/i)
    expect(mockWorker.postMessage).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: 'EXTRACT_SUBGRAPH' }),
      expect.anything(),
    )
  })

  it('surfaces an extraction failure from the worker', async () => {
    render(<App />)
    loadGraph(appGraph)

    selectTwo('node_0_Conv', 'node_1_Relu')
    fireEvent.click(screen.getByTestId('extract-repro-button'))
    await waitFor(() => expect(mockWorker.postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'EXTRACT_SUBGRAPH' })))

    act(() => {
      mockWorker.onmessage?.({ data: { type: 'ERROR', payload: 'boom', scope: 'operation' } } as MessageEvent)
    })
    expect(await screen.findByTestId('announcement')).toHaveTextContent('boom')
  })

  it('hides the extract-repro action for read-only TFLite models', () => {
    render(<App />)
    loadGraph({ ...appGraph, format: 'tflite' })
    selectTwo('node_0_Conv', 'node_1_Relu')
    expect(screen.queryByTestId('extract-repro-button')).not.toBeInTheDocument()
  })
})
