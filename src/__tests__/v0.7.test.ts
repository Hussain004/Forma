import { describe, it, expect } from 'vitest'
import type { SelectableGraph, SelectableNode } from '../lib/graphUtils'
import { setMultiSelection, bulkExclude, bulkInclude } from '../lib/graphUtils'
import type { OnnxNode } from '../lib/onnxTypes'
import { LayerInspector } from '../components/LayerInspector'
import { render, screen } from '@testing-library/react'
import { createElement } from 'react'

const makeNode = (id: string, overrides: Partial<SelectableNode> = {}): SelectableNode => ({
  id,
  opType: 'Relu',
  inputs: [],
  outputs: [],
  attributes: {},
  paramCount: 0,
  estimatedSizeMB: 0,
  selected: false,
  ...overrides,
})

const makeGraph = (nodes: SelectableNode[]): SelectableGraph => ({
  nodes,
  edges: [],
  modelName: 'test',
  totalParams: 0,
  totalSizeMB: 0,
})

// Feature 1: setMultiSelection

describe('setMultiSelection', () => {
  it('marks nodes in the ids Set as selected', () => {
    const graph = makeGraph([makeNode('a'), makeNode('b'), makeNode('c')])
    const result = setMultiSelection(graph, new Set(['a', 'c']))
    expect(result.nodes.find((n) => n.id === 'a')?.selected).toBe(true)
    expect(result.nodes.find((n) => n.id === 'c')?.selected).toBe(true)
  })

  it('marks nodes not in the ids Set as not selected', () => {
    const graph = makeGraph([
      makeNode('a', { selected: true }),
      makeNode('b', { selected: true }),
    ])
    const result = setMultiSelection(graph, new Set(['a']))
    expect(result.nodes.find((n) => n.id === 'b')?.selected).toBe(false)
  })

  it('leaves all nodes deselected for an empty Set', () => {
    const graph = makeGraph([
      makeNode('a', { selected: true }),
      makeNode('b', { selected: true }),
    ])
    const result = setMultiSelection(graph, new Set<string>())
    expect(result.nodes.every((n) => n.selected === false)).toBe(true)
  })

  it('returns a new graph without mutating the input', () => {
    const graph = makeGraph([makeNode('a'), makeNode('b')])
    const result = setMultiSelection(graph, new Set(['a']))
    expect(result).not.toBe(graph)
    expect(graph.nodes.find((n) => n.id === 'a')?.selected).toBe(false)
  })
})

// Feature 2: bulkExclude and bulkInclude

describe('bulkExclude and bulkInclude', () => {
  it('bulkExclude sets excluded=true for all matching nodes', () => {
    const graph = makeGraph([makeNode('a'), makeNode('b'), makeNode('c')])
    const result = bulkExclude(graph, new Set(['a', 'b']))
    expect(result.nodes.find((n) => n.id === 'a')?.excluded).toBe(true)
    expect(result.nodes.find((n) => n.id === 'b')?.excluded).toBe(true)
  })

  it('bulkExclude does not affect non-matching nodes', () => {
    const graph = makeGraph([makeNode('a'), makeNode('c')])
    const result = bulkExclude(graph, new Set(['a']))
    expect(result.nodes.find((n) => n.id === 'c')?.excluded).toBeFalsy()
  })

  it('bulkInclude sets excluded=false for all matching nodes', () => {
    const graph = makeGraph([
      makeNode('a', { excluded: true }),
      makeNode('b', { excluded: true }),
    ])
    const result = bulkInclude(graph, new Set(['a', 'b']))
    expect(result.nodes.find((n) => n.id === 'a')?.excluded).toBe(false)
    expect(result.nodes.find((n) => n.id === 'b')?.excluded).toBe(false)
  })

  it('bulkInclude does not affect non-matching nodes', () => {
    const graph = makeGraph([
      makeNode('a', { excluded: true }),
      makeNode('c', { excluded: true }),
    ])
    const result = bulkInclude(graph, new Set(['a']))
    expect(result.nodes.find((n) => n.id === 'c')?.excluded).toBe(true)
  })

  it('both return new graphs without mutating the input', () => {
    const graph = makeGraph([makeNode('a'), makeNode('b', { excluded: true })])
    const excluded = bulkExclude(graph, new Set(['a']))
    const included = bulkInclude(graph, new Set(['b']))
    expect(excluded).not.toBe(graph)
    expect(included).not.toBe(graph)
    expect(graph.nodes.find((n) => n.id === 'a')?.excluded).toBeFalsy()
    expect(graph.nodes.find((n) => n.id === 'b')?.excluded).toBe(true)
  })
})

// Feature 3: Aggregate LayerInspector

const aggregateNodes: OnnxNode[] = [
  { id: 'a', opType: 'Conv', inputs: [], outputs: [], attributes: {}, paramCount: 1000, estimatedSizeMB: 0.004 },
  { id: 'b', opType: 'Relu', inputs: [], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
]
const multiSelection = { nodes: aggregateNodes, totalParams: 1000, totalSizeMB: 0.004 }

describe('LayerInspector aggregate view', () => {
  it('renders an "N nodes selected" heading when node is null', () => {
    const { container } = render(
      createElement(LayerInspector, { node: null, multiSelection }),
    )
    expect(container.textContent).toMatch(/2\s*nodes?\s+selected/i)
  })

  it('renders the total param count somewhere in the output', () => {
    const { container } = render(
      createElement(LayerInspector, { node: null, multiSelection }),
    )
    expect(container.textContent).toMatch(/1,?000/)
  })

  it('renders an EXCLUDE ALL button', () => {
    render(createElement(LayerInspector, { node: null, multiSelection }))
    expect(screen.getByRole('button', { name: /exclude all/i })).toBeInTheDocument()
  })

  it('renders an INCLUDE ALL button', () => {
    render(createElement(LayerInspector, { node: null, multiSelection }))
    expect(screen.getByRole('button', { name: /include all/i })).toBeInTheDocument()
  })
})

// Feature 4: SelectableGraph field preservation

describe('setMultiSelection field preservation', () => {
  it('preserves modelName, totalParams, and edges unchanged', () => {
    const graph: SelectableGraph = {
      nodes: [makeNode('a')],
      edges: [{ id: 'e1', source: 'a', target: 'b' }],
      modelName: 'resnet',
      totalParams: 42,
      totalSizeMB: 1.5,
    }
    const result = setMultiSelection(graph, new Set(['a']))
    expect(result.modelName).toBe('resnet')
    expect(result.totalParams).toBe(42)
    expect(result.edges).toEqual(graph.edges)
  })
})
