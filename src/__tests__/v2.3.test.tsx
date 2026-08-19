import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import { parseOnnxProto } from '../lib/onnxProtoParser'
import { writeModifiedOnnx, type StructuralOp } from '../lib/onnxProtoWriter'
import {
  describeHistoryEntry,
  friendlyNodeLabel,
  graphIOIndex,
  isConnectedSubgraph,
  renameNode,
  renameTensor,
  replaceConstantValues,
  setGraphIOType,
  toSelectableGraph,
  type SelectableGraph,
} from '../lib/graphUtils'
import { createShareHash } from '../lib/shareLinks'
import type { OnnxGraph } from '../lib/onnxTypes'

// ---- Minimal ONNX protobuf builder (for test fixtures only) -----------------
// Independent of onnxProtoWriter.ts, mirroring v1.1/v2.2's fixture builders.

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
function float32Bytes(value: number): number[] {
  const buf = new ArrayBuffer(4)
  new DataView(buf).setFloat32(0, value, true)
  return Array.from(new Uint8Array(buf))
}
function makeNode(opType: string, inputs: string[], outputs: string[], name: string): number[] {
  const bytes: number[] = []
  for (const s of inputs) bytes.push(...strField(1, s))
  for (const s of outputs) bytes.push(...strField(2, s))
  bytes.push(...strField(3, name))
  bytes.push(...strField(4, opType))
  return bytes
}
// dtype 1 = float32 raw_data (field 9)
function makeInitializer(name: string, dims: number[], values: number[]): number[] {
  const dimBytes = dims.flatMap((d) => encodeVarint(d))
  const raw = values.flatMap(float32Bytes)
  return [...lenField(1, dimBytes), ...varintField(2, 1), ...lenField(9, raw), ...strField(8, name)]
}
function makeValueInfo(name: string, elemType?: number, dims?: (number | string)[]): number[] {
  const bytes = [...strField(1, name)]
  if (elemType !== undefined) {
    const dimBytes = (dims ?? []).flatMap((d) =>
      typeof d === 'number' ? lenField(1, varintField(1, d)) : lenField(1, strField(2, d)),
    )
    const tensorType = [...varintField(1, elemType), ...(dims ? lenField(2, dimBytes) : [])]
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
function makeModel(graph: number[]): ArrayBuffer {
  return new Uint8Array([...varintField(1, 8), ...lenField(7, graph)]).buffer
}

// x -[Conv,W,b]-> y -[Relu]-> z
function makeFixture(): ArrayBuffer {
  const conv = makeNode('Conv', ['x', 'W', 'b'], ['y'], 'conv0')
  const relu = makeNode('Relu', ['y'], ['z'], 'relu0')
  const graph = makeGraph(
    [conv, relu],
    [makeValueInfo('x', 1, [1, 3, 8, 8])],
    [makeValueInfo('z', 1, [1, 3, 8, 8])],
    [makeInitializer('W', [4, 3, 3, 3], new Array(108).fill(0.1)), makeInitializer('b', [4], [1, 2, 3, 4])],
  )
  return makeModel(graph)
}

function apply(op: StructuralOp): ReturnType<typeof parseOnnxProto> {
  const patched = writeModifiedOnnx(makeFixture(), new Map(), [op])
  return parseOnnxProto(patched)
}

// ---- Writer-level: each v2.3 op against real wire bytes ---------------------

describe('v2.3 writer ops', () => {
  it('renameNode sets NodeProto.name without touching wiring', () => {
    const parsed = apply({ type: 'renameNode', nodeIndex: 0, name: 'my_conv' })
    expect(parsed.nodes[0].name).toBe('my_conv')
    expect(parsed.nodes[0].inputs).toEqual(['x', 'W', 'b'])
  })

  it('renameTensor updates every node reference to an intermediate tensor', () => {
    const parsed = apply({ type: 'renameTensor', oldName: 'y', newName: 'y2' })
    expect(parsed.nodes[0].outputs).toEqual(['y2'])
    expect(parsed.nodes[1].inputs).toEqual(['y2'])
  })

  it('renameTensor updates a graph input name and every node that consumes it', () => {
    const parsed = apply({ type: 'renameTensor', oldName: 'x', newName: 'pixels' })
    expect(parsed.inputs[0].name).toBe('pixels')
    expect(parsed.nodes[0].inputs[0]).toBe('pixels')
  })

  it('renameTensor updates an initializer name and its referencing node', () => {
    const parsed = apply({ type: 'renameTensor', oldName: 'W', newName: 'conv_weight' })
    expect(parsed.initializers.map((i) => i.name)).toContain('conv_weight')
    expect(parsed.nodes[0].inputs[1]).toBe('conv_weight')
  })

  it('setGraphIO replaces a graph input\'s declared type and mixed concrete/symbolic shape', () => {
    const parsed = apply({
      type: 'setGraphIO', ioKind: 'input', ioIndex: 0, elemType: 11,
      dims: [{ value: 1 }, { param: 'C' }, { value: 8 }, { value: 8 }],
    })
    expect(parsed.inputs[0].elemType).toBe(11)
    expect(parsed.inputs[0].shape).toEqual([{ value: 1 }, { param: 'C' }, { value: 8 }, { value: 8 }])
  })

  it('setGraphIO with dims: null produces an unranked declaration, not an empty shape', () => {
    const parsed = apply({ type: 'setGraphIO', ioKind: 'output', ioIndex: 0, elemType: 1, dims: null })
    expect(parsed.outputs[0].shape).toBeUndefined()
    expect(parsed.outputs[0].elemType).toBe(1)
  })

  it('promoteOutput adds an intermediate tensor as an additional graph output, synthesized when undeclared', () => {
    const parsed = apply({ type: 'promoteOutput', tensorName: 'y' })
    expect(parsed.outputs.map((o) => o.name)).toEqual(['z', 'y'])
    expect(parsed.outputs[1].shape).toBeUndefined() // 'y' has no value_info in the fixture
  })

  it('promoteOutput is a no-op when the tensor is already a graph output', () => {
    const parsed = apply({ type: 'promoteOutput', tensorName: 'z' })
    expect(parsed.outputs.map((o) => o.name)).toEqual(['z'])
  })

  it('replaceConstant overwrites a small initializer\'s raw_data in place', () => {
    const parsed = apply({ type: 'replaceConstant', initializerName: 'b', values: [10, 20, 30, 40] })
    const b = parsed.initializers.find((i) => i.name === 'b')!
    expect(b.values).toEqual([10, 20, 30, 40])
    expect(b.dims).toEqual([4]) // shape untouched
  })

  it('replaceConstant is a no-op when the value count does not match', () => {
    const parsed = apply({ type: 'replaceConstant', initializerName: 'b', values: [1, 2, 3] })
    const b = parsed.initializers.find((i) => i.name === 'b')!
    expect(b.values).toEqual([1, 2, 3, 4]) // unchanged from the fixture
  })

  it('preserves model-level metadata across a deployment-surgery op', () => {
    const parsed = apply({ type: 'renameNode', nodeIndex: 0, name: 'x' })
    expect(parsed.metadata?.irVersion).toBe(8)
  })
})

// ---- graphUtils pure transforms and helpers ---------------------------------

const smallGraph: SelectableGraph = toSelectableGraph({
  modelName: 'small.onnx',
  totalParams: 0,
  totalSizeMB: 0,
  nodes: [
    { id: 'input_0', opType: 'Input', inputs: [], outputs: ['x'], attributes: {}, paramCount: 0, estimatedSizeMB: 0, outputMetadata: [{ elemType: 1, shape: [{ value: 1 }] }] },
    { id: 'node_0_Relu', opType: 'Relu', inputs: ['x'], outputs: ['y'], attributes: {}, paramCount: 0, estimatedSizeMB: 0, inputMetadata: [{ elemType: 1, values: [1, 2] }] },
    { id: 'output_0', opType: 'Output', inputs: ['y'], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
  ],
  edges: [
    { id: 'input_0->node_0_Relu@x', source: 'input_0', target: 'node_0_Relu', label: 'x' },
    { id: 'node_0_Relu->output_0@y', source: 'node_0_Relu', target: 'output_0', label: 'y' },
  ],
})

describe('v2.3 graphUtils', () => {
  it('renameNode updates only the targeted node\'s display name', () => {
    const next = renameNode(smallGraph, 'node_0_Relu', 'act1')
    expect(next.nodes.find((n) => n.id === 'node_0_Relu')?.name).toBe('act1')
  })

  it('renameTensor updates every node reference and the matching edge label', () => {
    const next = renameTensor(smallGraph, 'x', 'pixels')
    expect(next.nodes.find((n) => n.id === 'input_0')?.outputs).toEqual(['pixels'])
    expect(next.nodes.find((n) => n.id === 'node_0_Relu')?.inputs).toEqual(['pixels'])
    expect(next.edges[0].label).toBe('pixels')
  })

  it('setGraphIOType updates the Input pseudo-node\'s outputMetadata, not an unrelated node', () => {
    const next = setGraphIOType(smallGraph, 'input_0', 7, [{ value: 4 }])
    const inputNode = next.nodes.find((n) => n.id === 'input_0')!
    expect(inputNode.outputMetadata?.[0]).toEqual({ elemType: 7, shape: [{ value: 4 }] })
    expect(inputNode.outputShapes).toEqual([[{ value: 4 }]])
  })

  it('setGraphIOType with null dims clears outputShapes (unranked)', () => {
    const next = setGraphIOType(smallGraph, 'input_0', 1, null)
    expect(next.nodes.find((n) => n.id === 'input_0')?.outputShapes).toBeUndefined()
  })

  it('replaceConstantValues updates the inspectable values on the referencing node', () => {
    const graphWithConst: SelectableGraph = {
      ...smallGraph,
      nodes: smallGraph.nodes.map((n) => (n.id === 'node_0_Relu' ? { ...n, inputs: ['x', 'W'], inputMetadata: [{ elemType: 1 }, { elemType: 1, values: [1, 2] }] } : n)),
    }
    const result = replaceConstantValues(graphWithConst, 'W', [5, 6])
    expect(result.nodes.find((n) => n.id === 'node_0_Relu')?.inputMetadata?.[1].values).toEqual([5, 6])
  })

  it('graphIOIndex parses Input_N/Output_N pseudo-node ids and rejects everything else', () => {
    expect(graphIOIndex('input_3')).toEqual({ ioKind: 'input', ioIndex: 3 })
    expect(graphIOIndex('output_0')).toEqual({ ioKind: 'output', ioIndex: 0 })
    expect(graphIOIndex('node_0_Relu')).toBeNull()
    expect(graphIOIndex('custom_1')).toBeNull()
  })

  it('friendlyNodeLabel describes graph I/O pseudo-nodes', () => {
    expect(friendlyNodeLabel('input_2')).toBe('Graph input 2')
    expect(friendlyNodeLabel('output_0')).toBe('Graph output 0')
  })

  it('describeHistoryEntry covers every v2.3 entry type', () => {
    expect(describeHistoryEntry({ type: 'renameNode', nodeId: 'node_0_Relu', nodeIndex: 0, name: 'act1' })).toMatch(/Relu.*act1/)
    expect(describeHistoryEntry({ type: 'renameTensor', oldName: 'x', newName: 'pixels' })).toMatch(/x.*pixels/)
    expect(describeHistoryEntry({ type: 'setGraphIO', nodeId: 'input_0', ioKind: 'input', ioIndex: 0, elemType: 1, dims: null })).toMatch(/Graph input 0/)
    expect(describeHistoryEntry({ type: 'promoteOutput', tensorName: 'y' })).toMatch(/y/)
    expect(describeHistoryEntry({ type: 'replaceConstant', initializerName: 'W', values: [1, 2] })).toMatch(/W.*1, 2/)
  })

  it('isConnectedSubgraph still holds for the smallGraph fixture reused above', () => {
    expect(isConnectedSubgraph(smallGraph, new Set(['input_0', 'node_0_Relu']))).toBe(true)
  })
})

// ---- shareLinks: new entry types are rejected clearly, not silently dropped ---

describe('v2.3 share links', () => {
  it('refuses to encode a share hash containing a deployment-surgery edit', () => {
    expect(() => createShareHash(
      'ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0',
      'model.onnx',
      [{ type: 'renameNode', nodeId: 'node_0_Relu', nodeIndex: 0, name: 'act1' }],
    )).toThrow(/renameNode/)
  })
})

// ---- App wiring --------------------------------------------------------------

const appGraph: OnnxGraph = {
  modelName: 'io.onnx',
  totalParams: 0,
  totalSizeMB: 0,
  nodes: [
    { id: 'input_0', opType: 'Input', inputs: [], outputs: ['x'], attributes: {}, paramCount: 0, estimatedSizeMB: 0, outputShapes: [[{ value: 1 }, { value: 4 }]], outputMetadata: [{ elemType: 1, shape: [{ value: 1 }, { value: 4 }] }] },
    { id: 'node_0_Relu', opType: 'Relu', inputs: ['x', 'W'], outputs: ['y'], attributes: {}, paramCount: 4, estimatedSizeMB: 0, inputMetadata: [{ elemType: 1 }, { elemType: 1, values: [1, 2, 3, 4] }] },
    { id: 'node_1_Identity', opType: 'Identity', inputs: ['y'], outputs: ['w'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'output_0', opType: 'Output', inputs: ['w'], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
  ],
  edges: [
    { id: 'input_0->node_0_Relu@x', source: 'input_0', target: 'node_0_Relu', label: 'x' },
    { id: 'node_0_Relu->node_1_Identity@y', source: 'node_0_Relu', target: 'node_1_Identity', label: 'y' },
    { id: 'node_1_Identity->output_0@w', source: 'node_1_Identity', target: 'output_0', label: 'w' },
  ],
}

const graphCanvasState = vi.hoisted(() => ({
  props: null as null | { onNodeSelect?: (id: string) => void },
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

// Selects a node via the mocked GraphCanvas's callback directly -- works for
// Input/Output pseudo-nodes too, which the StatsBar search dropdown
// deliberately excludes (see App.tsx's dropdownResults filter).
function selectNode(id: string) {
  act(() => { graphCanvasState.props?.onNodeSelect?.(id) })
}

describe('v2.3 App wiring', () => {
  it('renames a node through the inspector and records it in history', async () => {
    render(<App />)
    loadGraph(appGraph)
    selectNode('node_0_Relu')

    const nameButton = await screen.findByTestId('nodename-value-nodename')
    fireEvent.click(nameButton)
    const input = screen.getByTestId('nodename-input-nodename')
    fireEvent.change(input, { target: { value: 'act1' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(screen.getByTestId('history-tab')).toHaveTextContent('History (1)')
    fireEvent.click(screen.getByTestId('history-tab'))
    expect(screen.getByText(/Renamed Relu to "act1"/)).toBeInTheDocument()
  })

  it('renames a tensor through an IORow and reflects it in the change log', async () => {
    render(<App />)
    loadGraph(appGraph)
    selectNode('node_0_Relu')

    const tensorButton = await screen.findByTestId('tensor-value-input:0')
    fireEvent.click(tensorButton)
    fireEvent.change(screen.getByTestId('tensor-input-input:0'), { target: { value: 'pixels' } })
    fireEvent.keyDown(screen.getByTestId('tensor-input-input:0'), { key: 'Enter' })

    fireEvent.click(screen.getByTestId('changes-tab'))
    expect(screen.getByText(/Renamed tensor x to pixels/)).toBeInTheDocument()
  })

  it('promotes an intermediate tensor to a graph output, and rejects promoting one twice', async () => {
    render(<App />)
    loadGraph(appGraph)
    selectNode('node_0_Relu')

    const promote = await screen.findByTestId('promote-output-y')
    fireEvent.click(promote)
    expect(await screen.findByTestId('announcement')).toHaveTextContent(/Promoted y/)

    fireEvent.click(promote)
    expect(await screen.findByTestId('announcement')).toHaveTextContent(/already a graph output/)
  })

  it('replaces small constant values and rejects a length mismatch', async () => {
    render(<App />)
    loadGraph(appGraph)
    selectNode('node_0_Relu')

    const constButton = await screen.findByTestId('const-value-const:W')
    expect(constButton).toHaveTextContent('1, 2, 3, 4')
    fireEvent.click(constButton)
    fireEvent.change(screen.getByTestId('const-input-const:W'), { target: { value: '5, 6, 7, 8' } })
    fireEvent.keyDown(screen.getByTestId('const-input-const:W'), { key: 'Enter' })
    expect(await screen.findByTestId('const-value-const:W')).toHaveTextContent('5, 6, 7, 8')

    fireEvent.click(screen.getByTestId('const-value-const:W'))
    fireEvent.change(screen.getByTestId('const-input-const:W'), { target: { value: '1, 2' } })
    fireEvent.keyDown(screen.getByTestId('const-input-const:W'), { key: 'Enter' })
    expect(await screen.findByTestId('announcement')).toHaveTextContent(/needs exactly 4 value/)
  })

  it('edits a graph input\'s declared elem type and shape from the Input pseudo-node', async () => {
    render(<App />)
    loadGraph(appGraph)
    selectNode('input_0')

    const typeButton = await screen.findByTestId('io-value-iotype')
    fireEvent.click(typeButton)
    fireEvent.change(screen.getByTestId('io-input-iotype'), { target: { value: '11' } })
    fireEvent.keyDown(screen.getByTestId('io-input-iotype'), { key: 'Enter' })

    const shapeButton = await screen.findByTestId('io-value-ioshape')
    fireEvent.click(shapeButton)
    fireEvent.change(screen.getByTestId('io-input-ioshape'), { target: { value: 'batch, 4' } })
    fireEvent.keyDown(screen.getByTestId('io-input-ioshape'), { key: 'Enter' })

    fireEvent.click(screen.getByTestId('changes-tab'))
    // Two separate setGraphIO entries (elem type, then shape) -- both show up.
    expect(screen.getByTestId('change-log-text').textContent).toMatch(/Graph input 0.*Graph input 0/s)
  })

  it('undoes a rename back to the original node name', async () => {
    render(<App />)
    loadGraph(appGraph)
    selectNode('node_0_Relu')

    const nameButton = await screen.findByTestId('nodename-value-nodename')
    fireEvent.click(nameButton)
    fireEvent.change(screen.getByTestId('nodename-input-nodename'), { target: { value: 'act1' } })
    fireEvent.keyDown(screen.getByTestId('nodename-input-nodename'), { key: 'Enter' })
    await waitFor(() => expect(screen.getByTestId('nodename-value-nodename')).toHaveTextContent('act1'))

    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    await waitFor(() => expect(screen.getByTestId('nodename-value-nodename')).not.toHaveTextContent('act1'))
  })
})
