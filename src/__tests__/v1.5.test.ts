import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { createElement } from 'react'
import { render, screen, fireEvent, act } from '@testing-library/react'
import { parseOnnxProto } from '../lib/onnxProtoParser'
import { writeModifiedOnnx, type StructuralOp } from '../lib/onnxProtoWriter'
import {
  structuralNodeIndex,
  isStructuralEditTarget,
  addCustomNode,
  getDeleteEligibility,
  validateRewire,
  toSelectableGraph,
  type SelectableGraph,
} from '../lib/graphUtils'
import App from '../App'
import type { OnnxGraph } from '../lib/onnxTypes'

// ---- Minimal ONNX protobuf builder (for test fixtures only) ----
// Same hand-encoding approach as v1.1-v1.4.test.ts -- independent of
// onnxProtoWriter.ts so the writer is tested as a black box against real wire bytes.

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

function makeNode(opType: string, inputs: string[], outputs: string[], name: string): number[] {
  const bytes: number[] = []
  for (const s of inputs) bytes.push(...strField(1, s))
  for (const s of outputs) bytes.push(...strField(2, s))
  bytes.push(...strField(3, name))
  bytes.push(...strField(4, opType))
  return bytes
}

function makeValueInfo(name: string): number[] { return strField(1, name) }

function makeGraph(nodes: number[][], inputs: string[], outputs: string[]): number[] {
  const bytes: number[] = []
  for (const n of nodes) bytes.push(...lenField(1, n))
  for (const s of inputs) bytes.push(...lenField(11, makeValueInfo(s)))
  for (const s of outputs) bytes.push(...lenField(12, makeValueInfo(s)))
  return bytes
}

function makeModel(graph: number[]): ArrayBuffer {
  const bytes = [...encodeVarint((1 << 3) | 0), ...encodeVarint(7), ...lenField(7, graph)]
  return new Uint8Array(bytes).buffer
}

// Relu(x)->y [node 0], graph output y
function makeSingleNodeFixture(): ArrayBuffer {
  const relu = makeNode('Relu', ['x'], ['y'], 'relu0')
  const graph = makeGraph([relu], ['x'], ['y'])
  return makeModel(graph)
}

describe('writeModifiedOnnx addNode (v1.5)', () => {
  it('adds a node with no attributes and unwired inputs left empty', () => {
    const original = makeSingleNodeFixture()
    const ops: StructuralOp[] = [{ type: 'addNode', newNodeIndex: 1, opType: 'Sigmoid', inputCount: 1 }]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes).toHaveLength(2)
    const added = result.nodes.find(n => n.opType === 'Sigmoid')
    expect(added).toBeDefined()
    expect(added?.inputs).toEqual([''])
    expect(added?.outputs).toEqual(['custom_1_out'])
    expect(Object.keys(added?.attributes ?? {})).toHaveLength(0)
  })

  it('accepts an arbitrary free-text op type with zero inputs', () => {
    const original = makeSingleNodeFixture()
    const ops: StructuralOp[] = [{ type: 'addNode', newNodeIndex: 1, opType: 'MyCustomOp', inputCount: 0 }]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    const added = result.nodes.find(n => n.opType === 'MyCustomOp')
    expect(added).toBeDefined()
    expect(added?.inputs).toEqual([])
  })

  it('reorders a custom node before an original node that rewires to consume it', () => {
    const original = makeSingleNodeFixture() // Relu(x)->y, index 0
    const ops: StructuralOp[] = [
      { type: 'addNode', newNodeIndex: 1, opType: 'Sigmoid', inputCount: 1 },
      { type: 'rewire', targetNodeIndex: 0, inputPosition: 0, sourceNodeIndex: -1 },
    ]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes.map(n => n.opType)).toEqual(['Sigmoid', 'Relu'])
    expect(result.nodes[1].inputs).toEqual(['custom_1_out'])
  })

  it('wires an original node output into a custom node input', () => {
    const original = makeSingleNodeFixture() // Relu(x)->y, index 0
    const ops: StructuralOp[] = [
      { type: 'addNode', newNodeIndex: 1, opType: 'Sigmoid', inputCount: 1 },
      { type: 'rewire', targetNodeIndex: -1, inputPosition: 0, sourceNodeIndex: 0 },
    ]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes.map(n => n.opType)).toEqual(['Relu', 'Sigmoid'])
    const custom = result.nodes.find(n => n.opType === 'Sigmoid')
    expect(custom?.inputs).toEqual(['y'])
  })

  it('deletes a custom node via the same delete op used for original nodes', () => {
    const original = makeSingleNodeFixture()
    const ops: StructuralOp[] = [
      { type: 'addNode', newNodeIndex: 1, opType: 'Sigmoid', inputCount: 1 },
      { type: 'delete', nodeIndex: -1, keepInputPosition: null },
    ]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes.map(n => n.opType)).toEqual(['Relu'])
  })
})

// ---- graphUtils structuralNodeIndex / addCustomNode (v1.5) ----

function makeChainOnnxGraph(): OnnxGraph {
  return {
    modelName: 'test',
    totalParams: 0,
    totalSizeMB: 0,
    nodes: [
      { id: 'node_0_Conv', opType: 'Conv', inputs: ['x', 'W'], outputs: ['y'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
      { id: 'node_1_Relu', opType: 'Relu', inputs: ['y'], outputs: ['z'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
      { id: 'node_2_Sigmoid', opType: 'Sigmoid', inputs: ['z'], outputs: ['w'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
      { id: 'output_0', opType: 'Output', inputs: ['w'], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    ],
    edges: [
      { id: 'node_0_Conv->node_1_Relu@y', source: 'node_0_Conv', target: 'node_1_Relu', label: 'y' },
      { id: 'node_1_Relu->node_2_Sigmoid@z', source: 'node_1_Relu', target: 'node_2_Sigmoid', label: 'z' },
      { id: 'node_2_Sigmoid->output_0@w', source: 'node_2_Sigmoid', target: 'output_0', label: 'w' },
    ],
  }
}

function chainSelectable(): SelectableGraph {
  return toSelectableGraph(makeChainOnnxGraph())
}

describe('structuralNodeIndex (v1.5)', () => {
  it('resolves an original node id to its non-negative position', () => {
    expect(structuralNodeIndex('node_3_Relu')).toBe(3)
  })

  it('resolves a custom node id to the negation of its counter', () => {
    expect(structuralNodeIndex('custom_2')).toBe(-2)
  })

  it('returns null for a generated passthrough id', () => {
    expect(structuralNodeIndex('passthrough_1')).toBeNull()
    expect(isStructuralEditTarget('passthrough_1')).toBe(false)
  })

  it('returns null for Input/Output pseudo-node ids', () => {
    expect(structuralNodeIndex('input_0')).toBeNull()
    expect(structuralNodeIndex('output_0')).toBeNull()
  })
})

describe('addCustomNode (v1.5)', () => {
  it('adds a floating node with placeholder inputs and a real output, no new edges', () => {
    const g = chainSelectable()
    const result = addCustomNode(g, 'custom_1', 'Sigmoid', 2)
    const added = result.nodes.find(n => n.id === 'custom_1')
    expect(added?.opType).toBe('Sigmoid')
    expect(added?.inputs).toEqual(['__unwired_custom_1_in0', '__unwired_custom_1_in1'])
    expect(added?.outputs).toEqual(['custom_1_out'])
    expect(result.edges).toEqual(g.edges)
  })
})

describe('getDeleteEligibility accepts custom nodes (v1.5)', () => {
  it('is eligible for a dead-end custom node', () => {
    const withCustom = addCustomNode(chainSelectable(), 'custom_1', 'Sigmoid', 1)
    const result = getDeleteEligibility(withCustom, 'custom_1')
    expect(result.eligible).toBe(true)
  })
})

describe('validateRewire accepts custom nodes (v1.5)', () => {
  it('allows rewiring an original node output into a custom node input', () => {
    const withCustom = addCustomNode(chainSelectable(), 'custom_1', 'Sigmoid', 1)
    const result = validateRewire(withCustom, 'node_0_Conv', 'custom_1', 0)
    expect(result.valid).toBe(true)
  })

  it('allows rewiring a custom node output into an original node input', () => {
    const withCustom = addCustomNode(chainSelectable(), 'custom_1', 'Sigmoid', 1)
    const result = validateRewire(withCustom, 'custom_1', 'node_2_Sigmoid', 0)
    expect(result.valid).toBe(true)
  })

  it('still rejects a generated passthrough node as either endpoint', () => {
    const g = chainSelectable()
    const withSynthetic: SelectableGraph = {
      ...g,
      nodes: [...g.nodes, { id: 'passthrough_1', opType: 'Identity', inputs: ['y'], outputs: ['y2'], attributes: {}, paramCount: 0, estimatedSizeMB: 0, selected: false }],
    }
    const result = validateRewire(withSynthetic, 'node_0_Conv', 'passthrough_1', 0)
    expect(result.valid).toBe(false)
  })
})

// ---- App-level Add Node picker UI (v1.5) ----

const makeMockWorker = () => ({
  postMessage: vi.fn(),
  terminate: vi.fn(),
  onmessage: null as ((e: MessageEvent) => void) | null,
  onerror: null as ((e: unknown) => void) | null,
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
})

let mockWorker: ReturnType<typeof makeMockWorker>

const testGraph: OnnxGraph = {
  nodes: [
    { id: 'input_0', opType: 'Input', inputs: [], outputs: ['x'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'node_0_Conv', opType: 'Conv', inputs: ['x', 'weight'], outputs: ['y'], attributes: { kernel_shape: '3x3' }, paramCount: 139264, estimatedSizeMB: 0.532 },
    { id: 'output_0', opType: 'Output', inputs: ['y'], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
  ],
  edges: [
    { id: 'e0', source: 'input_0', target: 'node_0_Conv' },
    { id: 'e1', source: 'node_0_Conv', target: 'output_0' },
  ],
  modelName: 'test.onnx',
  totalParams: 139264,
  totalSizeMB: 0.532,
}

describe('App -- Add Node picker (v1.5)', () => {
  beforeEach(() => {
    mockWorker = makeMockWorker()
    vi.stubGlobal('Worker', vi.fn(function () { return mockWorker }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('adding a curated node increases the node count in the stats bar', () => {
    render(createElement(App))
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: testGraph } } as MessageEvent)
    })

    const before = screen.getByText(/^\d+ NODES$/i).textContent
    fireEvent.click(screen.getByText('Add Node'))
    fireEvent.mouseDown(screen.getByTestId('add-node-option-Relu'))

    const after = screen.getByText(/^\d+ NODES$/i).textContent
    expect(after).not.toBe(before)
    expect(after).toBe('2 NODES')
  })

  it('free-text entry adds a node with that op type', () => {
    render(createElement(App))
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: testGraph } } as MessageEvent)
    })

    fireEvent.click(screen.getByText('Add Node'))
    fireEvent.change(screen.getByTestId('add-node-query'), { target: { value: 'MyCustomOp' } })
    fireEvent.keyDown(screen.getByTestId('add-node-query'), { key: 'Enter' })

    expect(screen.getByText(/^2 NODES$/i)).toBeInTheDocument()
  })
})
