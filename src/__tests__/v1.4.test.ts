import { describe, it, expect, vi } from 'vitest'
import { createElement } from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { parseOnnxProto } from '../lib/onnxProtoParser'
import { writeModifiedOnnx, type StructuralOp } from '../lib/onnxProtoWriter'
import { validateRewire, rewireEdge, toSelectableGraph, type SelectableGraph } from '../lib/graphUtils'
import { LayerInspector } from '../components/LayerInspector'
import type { OnnxGraph } from '../lib/onnxTypes'

// ---- Minimal ONNX protobuf builder (for test fixtures only) ----
// Same hand-encoding approach as v1.1.test.ts/v1.2.test.ts -- independent of
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

function makeValueInfo(name: string): number[] { return strField(1, name) }

function makeGraph(nodes: number[][], inputs: string[], outputs: string[]): number[] {
  const bytes: number[] = []
  for (const n of nodes) bytes.push(...lenField(1, n))
  for (const s of inputs) bytes.push(...lenField(11, makeValueInfo(s)))
  for (const s of outputs) bytes.push(...lenField(12, makeValueInfo(s)))
  return bytes
}

function makeModel(graph: number[]): ArrayBuffer {
  const bytes = [...varintField(1, 7), ...lenField(7, graph)]
  return new Uint8Array(bytes).buffer
}

// Relu(x)->y [0] -> Sigmoid(x)->z [1], both consuming graph input x, graph outputs y and z
function makeParallelFixture(): ArrayBuffer {
  const relu = makeNode('Relu', ['x'], ['y'], 'relu0')
  const sigmoid = makeNode('Sigmoid', ['x'], ['z'], 'sigmoid0')
  const graph = makeGraph([relu, sigmoid], ['x'], ['y', 'z'])
  return makeModel(graph)
}

describe('writeModifiedOnnx rewire (v1.4)', () => {
  it('rewires a target to consume a source that appears later in the node list, reordering for topological validity', () => {
    const original = makeParallelFixture()
    // Relu (index 0) is BEFORE Sigmoid (index 1) in the file, but we rewire Relu's
    // input to come from Sigmoid's output -- Sigmoid must move ahead of Relu.
    const ops: StructuralOp[] = [{ type: 'rewire', targetNodeIndex: 0, inputPosition: 0, sourceNodeIndex: 1 }]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes.map(n => n.opType)).toEqual(['Sigmoid', 'Relu'])
    expect(result.nodes[1].inputs).toEqual(['z'])
  })

  it('silently skips a rewire referencing a nonexistent source index', () => {
    const original = makeParallelFixture()
    const ops: StructuralOp[] = [{ type: 'rewire', targetNodeIndex: 0, inputPosition: 0, sourceNodeIndex: 99 }]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes.map(n => n.opType)).toEqual(['Relu', 'Sigmoid'])
    expect(result.nodes[0].inputs).toEqual(['x'])
  })

  it('does not reorder when structuralOps contain no rewire (delete/insertPassthrough paths unaffected)', () => {
    const original = makeParallelFixture()
    const ops: StructuralOp[] = [{ type: 'delete', nodeIndex: 1, keepInputPosition: null }]
    const patched = writeModifiedOnnx(original, new Map(), ops)
    const result = parseOnnxProto(patched)

    expect(result.nodes.map(n => n.opType)).toEqual(['Relu'])
  })
})

// ---- graphUtils rewire helpers (v1.4) ----

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

describe('validateRewire (v1.4)', () => {
  it('rejects a self-connection', () => {
    const result = validateRewire(chainSelectable(), 'node_1_Relu', 'node_1_Relu', 0)
    expect(result.valid).toBe(false)
    expect(result.reason).toMatch(/itself/i)
  })

  it('rejects connecting to a synthetic (generated) node', () => {
    const g = chainSelectable()
    const withSynthetic: SelectableGraph = {
      ...g,
      nodes: [...g.nodes, { id: 'passthrough_1', opType: 'Identity', inputs: ['y'], outputs: ['y2'], attributes: {}, paramCount: 0, estimatedSizeMB: 0, selected: false }],
    }
    const result = validateRewire(withSynthetic, 'node_0_Conv', 'passthrough_1', 0)
    expect(result.valid).toBe(false)
  })

  it('rejects a connection that would create a cycle', () => {
    // node_0_Conv already reaches node_2_Sigmoid via node_1_Relu -- feeding Conv
    // FROM Sigmoid would close the loop.
    const result = validateRewire(chainSelectable(), 'node_2_Sigmoid', 'node_0_Conv', 0)
    expect(result.valid).toBe(false)
    expect(result.reason).toMatch(/cycle/i)
  })

  it('accepts a valid non-cyclic rewire', () => {
    const result = validateRewire(chainSelectable(), 'node_0_Conv', 'node_2_Sigmoid', 0)
    expect(result.valid).toBe(true)
  })
})

describe('rewireEdge (v1.4)', () => {
  it('replaces the target input at the given position and rewires the edge', () => {
    const g = chainSelectable()
    const result = rewireEdge(g, 'node_0_Conv', 'node_2_Sigmoid', 0)
    const target = result.nodes.find(n => n.id === 'node_2_Sigmoid')
    expect(target?.inputs[0]).toBe('y')
    const newEdge = result.edges.find(e => e.source === 'node_0_Conv' && e.target === 'node_2_Sigmoid')
    expect(newEdge?.label).toBe('y')
    const oldEdge = result.edges.find(e => e.source === 'node_1_Relu' && e.target === 'node_2_Sigmoid')
    expect(oldEdge).toBeUndefined()
  })

  it('is a no-op when the source output already feeds that position', () => {
    const g = chainSelectable()
    const result = rewireEdge(g, 'node_1_Relu', 'node_2_Sigmoid', 0)
    expect(result).toBe(g)
  })
})

// ---- LayerInspector bulk delete UI (v1.4) ----

describe('LayerInspector bulk delete (v1.4)', () => {
  it('calls onBulkDelete when DELETE ALL is clicked', () => {
    const onBulkDelete = vi.fn()
    const multiSelection = {
      nodes: [
        { id: 'node_0_Conv', opType: 'Conv', inputs: [], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
        { id: 'node_1_Relu', opType: 'Relu', inputs: [], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
      ],
      totalParams: 0,
      totalSizeMB: 0,
    }
    render(createElement(LayerInspector, { node: null, multiSelection, onBulkDelete }))
    fireEvent.click(screen.getByTestId('bulk-delete-button'))
    expect(onBulkDelete).toHaveBeenCalled()
  })

  it('omits the DELETE ALL button when onBulkDelete is not provided (read-only)', () => {
    const multiSelection = {
      nodes: [
        { id: 'node_0_Conv', opType: 'Conv', inputs: [], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
        { id: 'node_1_Relu', opType: 'Relu', inputs: [], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
      ],
      totalParams: 0,
      totalSizeMB: 0,
    }
    render(createElement(LayerInspector, { node: null, multiSelection }))
    expect(screen.queryByTestId('bulk-delete-button')).not.toBeInTheDocument()
  })
})
