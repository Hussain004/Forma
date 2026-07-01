import { describe, it, expect } from 'vitest'
import { createElement } from 'react'
import { render, screen } from '@testing-library/react'
import { filterGraph, toSelectableGraph } from '../lib/graphUtils'
import { LayerInspector } from '../components/LayerInspector'
import type { OnnxGraph, OnnxNode, OnnxEdge } from '../lib/onnxTypes'

// ---- Shared fixtures ----

const makeGraph = (extras?: Partial<OnnxGraph>): OnnxGraph => ({
  nodes: [
    { id: 'input_0', opType: 'Input', inputs: [], outputs: ['x_tensor'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'node_0_Conv', opType: 'Conv', inputs: ['x_tensor', 'weight'], outputs: ['y_tensor'], attributes: { kernel_shape: '[3, 3]', group: 1 }, paramCount: 1000, estimatedSizeMB: 0.004 },
    { id: 'output_0', opType: 'Output', inputs: ['y_tensor'], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
  ],
  edges: [
    { id: 'e0', source: 'input_0', target: 'node_0_Conv', label: 'x_tensor' },
    { id: 'e1', source: 'node_0_Conv', target: 'output_0', label: 'y_tensor' },
  ],
  modelName: 'test.onnx',
  totalParams: 1000,
  totalSizeMB: 0.004,
  ...extras,
})

// ---- Group 1: filterGraph tensor name search ----

describe('filterGraph tensor name search (v0.9)', () => {
  it('matches a node by its output tensor name', () => {
    const sg = toSelectableGraph(makeGraph())
    const result = filterGraph(sg, 'x_tensor')
    expect(result.nodes.find(n => n.id === 'input_0')?.dimmed).toBe(false)
    expect(result.nodes.find(n => n.id === 'node_0_Conv')?.dimmed).toBe(false)
    expect(result.nodes.find(n => n.id === 'output_0')?.dimmed).toBe(true)
  })

  it('matches a node by its input tensor name', () => {
    const sg = toSelectableGraph(makeGraph())
    const result = filterGraph(sg, 'y_tensor')
    expect(result.nodes.find(n => n.id === 'node_0_Conv')?.dimmed).toBe(false)
    expect(result.nodes.find(n => n.id === 'output_0')?.dimmed).toBe(false)
    expect(result.nodes.find(n => n.id === 'input_0')?.dimmed).toBe(true)
  })

  it('still matches by op type', () => {
    const sg = toSelectableGraph(makeGraph())
    const result = filterGraph(sg, 'conv')
    expect(result.nodes.find(n => n.id === 'node_0_Conv')?.dimmed).toBe(false)
    expect(result.nodes.find(n => n.id === 'input_0')?.dimmed).toBe(true)
  })

  it('partial tensor name match works', () => {
    const sg = toSelectableGraph(makeGraph())
    const result = filterGraph(sg, 'tensor')
    expect(result.nodes.every(n => !n.dimmed)).toBe(true)
  })

  it('no match dims all nodes', () => {
    const sg = toSelectableGraph(makeGraph())
    const result = filterGraph(sg, 'zzz_no_match')
    expect(result.nodes.every(n => n.dimmed)).toBe(true)
  })
})

// ---- Group 2: OnnxEdge shape field ----

describe('OnnxEdge shape field (v0.9)', () => {
  it('accepts shape field on an edge', () => {
    const edge: OnnxEdge = {
      id: 'e0',
      source: 'a',
      target: 'b',
      shape: [{ value: 1 }, { value: 3 }, { value: 224 }, { value: 224 }],
    }
    expect(edge.shape).toHaveLength(4)
    expect((edge.shape![0] as { value: number }).value).toBe(1)
  })

  it('shape field is optional and defaults to undefined', () => {
    const edge: OnnxEdge = { id: 'e0', source: 'a', target: 'b' }
    expect(edge.shape).toBeUndefined()
  })
})

// ---- Group 3: LayerInspector attributes section ----

describe('LayerInspector attributes section (v0.9)', () => {
  it('renders Attributes section when node has non-empty attributes', () => {
    const node: OnnxNode = {
      id: 'conv_0',
      opType: 'Conv',
      inputs: ['x', 'w'],
      outputs: ['y'],
      attributes: { kernel_shape: '[3, 3]', group: 1, dilations: '[1, 1]' },
      paramCount: 1000,
      estimatedSizeMB: 0.004,
    }
    render(createElement(LayerInspector, { node }))
    expect(screen.getByText(/attributes/i)).toBeInTheDocument()
    expect(screen.getByText('kernel_shape')).toBeInTheDocument()
    expect(screen.getByText('[3, 3]')).toBeInTheDocument()
    expect(screen.getByText('group')).toBeInTheDocument()
    expect(screen.getByText('1')).toBeInTheDocument()
  })

  it('does not render Attributes section when attributes is empty', () => {
    const node: OnnxNode = {
      id: 'relu_0',
      opType: 'Relu',
      inputs: ['x'],
      outputs: ['y'],
      attributes: {},
      paramCount: 0,
      estimatedSizeMB: 0,
    }
    render(createElement(LayerInspector, { node }))
    expect(screen.queryByText(/^attributes$/i)).not.toBeInTheDocument()
  })

  it('renders multiple attributes in order', () => {
    const node: OnnxNode = {
      id: 'bn_0',
      opType: 'BatchNormalization',
      inputs: ['x'],
      outputs: ['y'],
      attributes: { epsilon: 0.00001, momentum: 0.9 },
      paramCount: 0,
      estimatedSizeMB: 0,
    }
    render(createElement(LayerInspector, { node }))
    expect(screen.getByText('epsilon')).toBeInTheDocument()
    expect(screen.getByText('momentum')).toBeInTheDocument()
  })

  it('renders string attribute values', () => {
    const node: OnnxNode = {
      id: 'pool_0',
      opType: 'MaxPool',
      inputs: ['x'],
      outputs: ['y'],
      attributes: { auto_pad: 'NOTSET', kernel_shape: '[2, 2]' },
      paramCount: 0,
      estimatedSizeMB: 0,
    }
    render(createElement(LayerInspector, { node }))
    expect(screen.getByText('auto_pad')).toBeInTheDocument()
    expect(screen.getByText('NOTSET')).toBeInTheDocument()
  })
})

// ---- Group 4: OnnxNode attributes field present in graph fixtures ----

describe('OnnxNode attributes field (v0.9)', () => {
  it('node attributes record is accessible on graph nodes', () => {
    const graph = makeGraph()
    const conv = graph.nodes.find(n => n.opType === 'Conv')
    expect(conv?.attributes).toBeDefined()
    expect(conv?.attributes['kernel_shape']).toBe('[3, 3]')
    expect(conv?.attributes['group']).toBe(1)
  })

  it('non-parameterised nodes have empty attributes', () => {
    const graph = makeGraph()
    const inp = graph.nodes.find(n => n.opType === 'Input')
    expect(Object.keys(inp?.attributes ?? {})).toHaveLength(0)
  })
})
