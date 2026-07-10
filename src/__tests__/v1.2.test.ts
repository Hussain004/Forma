import { describe, it, expect, vi } from 'vitest'
import { createElement } from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { parseOnnxProto } from '../lib/onnxProtoParser'
import { writeModifiedOnnx, type StructuralOp } from '../lib/onnxProtoWriter'
import {
  getDeleteEligibility,
  deleteNodeWithReconnect,
  insertPassthroughNode,
  toSelectableGraph,
  type SelectableGraph,
  type DeleteEligibility,
} from '../lib/graphUtils'
import { LayerInspector } from '../components/LayerInspector'
import type { OnnxGraph, OnnxNode } from '../lib/onnxTypes'

// ---- Minimal ONNX protobuf builder (for test fixtures only) ----
// Same hand-encoding approach as v1.1.test.ts -- independent of onnxProtoWriter.ts
// so the writer is tested as a black box against real wire bytes.

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

function makeInitializer(name: string, dims: number[], elemType = 1): number[] {
  const bytes: number[] = []
  const dimBytes = dims.flatMap(d => encodeVarint(d))
  bytes.push(...lenField(1, dimBytes))
  bytes.push(...varintField(2, elemType))
  bytes.push(...strField(8, name))
  return bytes
}

function makeValueInfo(name: string): number[] { return strField(1, name) }

function makeGraph(nodes: number[][], inputs: string[], outputs: string[], initializers: number[][]): number[] {
  const bytes: number[] = []
  for (const n of nodes) bytes.push(...lenField(1, n))
  for (const i of initializers) bytes.push(...lenField(5, i))
  for (const s of inputs) bytes.push(...lenField(11, makeValueInfo(s)))
  for (const s of outputs) bytes.push(...lenField(12, makeValueInfo(s)))
  return bytes
}

function makeModel(graph: number[]): ArrayBuffer {
  const bytes = [...varintField(1, 7), ...lenField(7, graph)]
  return new Uint8Array(bytes).buffer
}

// Conv(x,W)->y [0] -> Relu(y)->z [1] -> Sigmoid(z)->w [2], graph output w
function makeChainFixture(): ArrayBuffer {
  const weight = makeInitializer('W', [4, 3, 3, 3])
  const conv = makeNode('Conv', ['x', 'W'], ['y'], 'conv0')
  const relu = makeNode('Relu', ['y'], ['z'], 'relu0')
  const sigmoid = makeNode('Sigmoid', ['z'], ['w'], 'sigmoid0')
  const graph = makeGraph([conv, relu, sigmoid], ['x'], ['w'], [weight])
  return makeModel(graph)
}

// Conv(x,W)->y [0] -> Relu(y)->z [1] (z is graph output), Dropout(y)->unused [2] (dead end)
function makeDeadEndFixture(): ArrayBuffer {
  const weight = makeInitializer('W', [4, 3, 3, 3])
  const conv = makeNode('Conv', ['x', 'W'], ['y'], 'conv0')
  const relu = makeNode('Relu', ['y'], ['z'], 'relu0')
  const dropout = makeNode('Dropout', ['y'], ['unused'], 'dropout0')
  const graph = makeGraph([conv, relu, dropout], ['x'], ['z'], [weight])
  return makeModel(graph)
}

// Relu(x)->y [0] -> Add(y,y)->z [1], graph output z
function makeDuplicateTensorFixture(): ArrayBuffer {
  const relu = makeNode('Relu', ['x'], ['y'], 'relu0')
  const add = makeNode('Add', ['y', 'y'], ['z'], 'add0')
  const graph = makeGraph([relu, add], ['x'], ['z'], [])
  return makeModel(graph)
}

describe('writeModifiedOnnx structural edits (v1.2)', () => {
  it('deletes a mid-chain node and reconnects its consumer', () => {
    const original = makeChainFixture()
    const ops: StructuralOp[] = [{ type: 'delete', nodeIndex: 1, keepInputPosition: 0 }]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes).toHaveLength(2)
    expect(result.nodes.map(n => n.opType)).toEqual(['Conv', 'Sigmoid'])
    expect(result.nodes[1].inputs).toEqual(['y'])
    expect(result.initializers).toHaveLength(1)
    expect(result.initializers[0].name).toBe('W')
  })

  it('deletes a dead-end node with no reconnection needed', () => {
    const original = makeDeadEndFixture()
    const ops: StructuralOp[] = [{ type: 'delete', nodeIndex: 2, keepInputPosition: null }]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes).toHaveLength(2)
    expect(result.nodes.map(n => n.opType)).toEqual(['Conv', 'Relu'])
    expect(result.nodes[1].inputs).toEqual(['y'])
  })

  it('rewires every occurrence of a duplicated tensor reference on delete', () => {
    const original = makeDuplicateTensorFixture()
    const ops: StructuralOp[] = [{ type: 'delete', nodeIndex: 0, keepInputPosition: 0 }]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes).toHaveLength(1)
    expect(result.nodes[0].opType).toBe('Add')
    expect(result.nodes[0].inputs).toEqual(['x', 'x'])
  })

  it('inserts a passthrough node before its consumer (topological order)', () => {
    const original = makeChainFixture()
    const ops: StructuralOp[] = [{ type: 'insertPassthrough', targetNodeIndex: 1, inputPosition: 0, newNodeName: 'id1' }]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes).toHaveLength(4)
    expect(result.nodes.map(n => n.opType)).toEqual(['Conv', 'Identity', 'Relu', 'Sigmoid'])
    const identity = result.nodes[1]
    expect(identity.inputs).toEqual(['y'])
    expect(result.nodes[2].inputs).toEqual(identity.outputs)
  })

  it('applies two sequential deletes where the second targets a tensor renamed by the first', () => {
    const original = makeChainFixture()
    const ops: StructuralOp[] = [
      { type: 'delete', nodeIndex: 1, keepInputPosition: 0 }, // Relu removed, Sigmoid now consumes y
      { type: 'delete', nodeIndex: 2, keepInputPosition: 0 }, // Sigmoid removed, its input[0] is now y
    ]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes).toHaveLength(1)
    expect(result.nodes[0].opType).toBe('Conv')
  })

  it('inserts a passthrough on an edge created by a prior delete-reconnection', () => {
    const original = makeChainFixture()
    const ops: StructuralOp[] = [
      { type: 'delete', nodeIndex: 1, keepInputPosition: 0 }, // Relu removed, Sigmoid now consumes y directly
      { type: 'insertPassthrough', targetNodeIndex: 2, inputPosition: 0, newNodeName: 'id1' },
    ]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes).toHaveLength(3)
    expect(result.nodes.map(n => n.opType)).toEqual(['Conv', 'Identity', 'Sigmoid'])
    expect(result.nodes[1].inputs).toEqual(['y'])
    expect(result.nodes[2].inputs).toEqual(result.nodes[1].outputs)
  })

  it('silently skips an op referencing a nonexistent node index', () => {
    const original = makeChainFixture()
    const ops: StructuralOp[] = [{ type: 'delete', nodeIndex: 99, keepInputPosition: 0 }]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes).toHaveLength(3)
  })

  it('applies attribute overrides and structural ops together in one export', () => {
    const original = makeChainFixture()
    const overrides = new Map([[0, { irrelevant: 1 }]]) // Conv has no attrs in this fixture; verifies no crash when combined
    const ops: StructuralOp[] = [{ type: 'delete', nodeIndex: 1, keepInputPosition: 0 }]
    const patched = writeModifiedOnnx(original, overrides, ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes).toHaveLength(2)
    expect(result.nodes.map(n => n.opType)).toEqual(['Conv', 'Sigmoid'])
  })
})

// ---- graphUtils structural helpers (v1.2) ----

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

describe('getDeleteEligibility (v1.2)', () => {
  it('is eligible with a single candidate input for a mid-chain node', () => {
    const g = chainSelectable()
    const result = getDeleteEligibility(g, 'node_1_Relu')
    expect(result.eligible).toBe(true)
    expect(result.candidateInputs).toEqual([{ tensorName: 'y', position: 0 }])
  })

  it('blocks a node that feeds a graph output', () => {
    const g = chainSelectable()
    const result = getDeleteEligibility(g, 'node_2_Sigmoid')
    expect(result.eligible).toBe(false)
    expect(result.reason).toMatch(/graph output/i)
  })

  it('allows dead-end deletion with no candidates', () => {
    const g = chainSelectable()
    // give node_2 a second, unconsumed successor to make it a non-output dead end for this case
    const withDeadEnd: SelectableGraph = {
      ...g,
      nodes: [...g.nodes, { id: 'node_3_Dropout', opType: 'Dropout', inputs: ['y'], outputs: ['unused'], attributes: {}, paramCount: 0, estimatedSizeMB: 0, selected: false }],
      edges: [...g.edges, { id: 'node_0_Conv->node_3_Dropout@y', source: 'node_0_Conv', target: 'node_3_Dropout', label: 'y' }],
    }
    const result = getDeleteEligibility(withDeadEnd, 'node_3_Dropout')
    expect(result.eligible).toBe(true)
    expect(result.candidateInputs).toEqual([])
  })

  it('blocks a node with multiple real outputs', () => {
    const g = chainSelectable()
    const withMultiOutput: SelectableGraph = {
      ...g,
      nodes: g.nodes.map(n => n.id === 'node_1_Relu' ? { ...n, outputs: ['z', 'z2'] } : n),
    }
    const result = getDeleteEligibility(withMultiOutput, 'node_1_Relu')
    expect(result.eligible).toBe(false)
    expect(result.reason).toMatch(/multiple outputs/i)
  })

  it('rejects generated (synthetic) node ids', () => {
    const g = chainSelectable()
    const withSynthetic: SelectableGraph = {
      ...g,
      nodes: [...g.nodes, { id: 'passthrough_1', opType: 'Identity', inputs: ['y'], outputs: ['y2'], attributes: {}, paramCount: 0, estimatedSizeMB: 0, selected: false }],
    }
    const result = getDeleteEligibility(withSynthetic, 'passthrough_1')
    expect(result.eligible).toBe(false)
  })
})

describe('deleteNodeWithReconnect (v1.2)', () => {
  it('removes the node and reconnects its consumer to its producer', () => {
    const g = chainSelectable()
    const result = deleteNodeWithReconnect(g, 'node_1_Relu', 0)
    expect(result.nodes.map(n => n.id)).toEqual(['node_0_Conv', 'node_2_Sigmoid', 'output_0'])
    const newEdge = result.edges.find(e => e.source === 'node_0_Conv' && e.target === 'node_2_Sigmoid')
    expect(newEdge).toBeDefined()
    expect(newEdge?.label).toBe('y')
  })
})

describe('insertPassthroughNode (v1.2)', () => {
  it('splits the target edge and inserts an Identity node', () => {
    const g = chainSelectable()
    const result = insertPassthroughNode(g, 'node_1_Relu', 0, 'passthrough_1')
    expect(result.nodes.some(n => n.id === 'passthrough_1' && n.opType === 'Identity')).toBe(true)
    const target = result.nodes.find(n => n.id === 'node_1_Relu')
    expect(target?.inputs[0]).toBe('y__identity_passthrough_1')
    const inEdge = result.edges.find(e => e.source === 'node_0_Conv' && e.target === 'passthrough_1')
    const outEdge = result.edges.find(e => e.source === 'passthrough_1' && e.target === 'node_1_Relu')
    expect(inEdge).toBeDefined()
    expect(outEdge).toBeDefined()
    expect(outEdge?.label).toBe('y__identity_passthrough_1')
  })
})

// ---- LayerInspector delete UI (v1.2) ----

const baseNode: OnnxNode = {
  id: 'node_1_Relu',
  opType: 'Relu',
  inputs: ['y'],
  outputs: ['z'],
  attributes: {},
  paramCount: 0,
  estimatedSizeMB: 0,
}

describe('LayerInspector delete node UI (v1.2)', () => {
  it('deletes immediately when there is a single reconnection candidate', () => {
    const onDeleteNode = vi.fn()
    const eligibility: DeleteEligibility = { eligible: true, candidateInputs: [{ tensorName: 'y', position: 0 }] }
    render(createElement(LayerInspector, { node: baseNode, onDeleteNode, deleteEligibility: eligibility }))
    fireEvent.click(screen.getByTestId('delete-node-button'))
    expect(onDeleteNode).toHaveBeenCalledWith('node_1_Relu', 0)
  })

  it('shows a picker when there are multiple reconnection candidates', () => {
    const onDeleteNode = vi.fn()
    const eligibility: DeleteEligibility = {
      eligible: true,
      candidateInputs: [{ tensorName: 'a', position: 0 }, { tensorName: 'b', position: 1 }],
    }
    render(createElement(LayerInspector, { node: baseNode, onDeleteNode, deleteEligibility: eligibility }))
    fireEvent.click(screen.getByTestId('delete-node-button'))
    expect(onDeleteNode).not.toHaveBeenCalled()
    expect(screen.getByTestId('delete-picker-option-1')).toBeInTheDocument()
    fireEvent.click(screen.getByTestId('delete-picker-option-1'))
    expect(onDeleteNode).toHaveBeenCalledWith('node_1_Relu', 1)
  })

  it('deletes with no reconnection for a dead-end node', () => {
    const onDeleteNode = vi.fn()
    const eligibility: DeleteEligibility = { eligible: true, candidateInputs: [] }
    render(createElement(LayerInspector, { node: baseNode, onDeleteNode, deleteEligibility: eligibility }))
    fireEvent.click(screen.getByTestId('delete-node-button'))
    expect(onDeleteNode).toHaveBeenCalledWith('node_1_Relu', null)
  })

  it('disables the button and shows the reason when blocked', () => {
    const onDeleteNode = vi.fn()
    const eligibility: DeleteEligibility = { eligible: false, reason: 'Node produces a graph output', candidateInputs: [] }
    render(createElement(LayerInspector, { node: baseNode, onDeleteNode, deleteEligibility: eligibility }))
    const button = screen.getByTestId('delete-node-button') as HTMLButtonElement
    expect(button.disabled).toBe(true)
    fireEvent.click(button)
    expect(onDeleteNode).not.toHaveBeenCalled()
    expect(screen.getByText('Node produces a graph output')).toBeInTheDocument()
  })
})
