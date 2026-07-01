import { describe, it, expect } from 'vitest'
import { opCategoryColor, getAncestors, getDescendants, computeGraphDepth } from '../lib/graphUtils'
import type { OnnxGraph, OnnxNode } from '../lib/onnxTypes'

const defaultNode: OnnxNode = {
  id: 'n0',
  opType: 'Relu',
  inputs: [],
  outputs: [],
  attributes: {},
  paramCount: 0,
  estimatedSizeMB: 0,
}

const makeNode = (overrides: Partial<OnnxNode> = {}): OnnxNode => ({ ...defaultNode, ...overrides })

const makeGraph = (nodes: OnnxNode[], edges: OnnxGraph['edges'] = []): OnnxGraph => ({
  nodes,
  edges,
  modelName: 'test',
  totalParams: 0,
  totalSizeMB: 0,
})

const edge = (source: string, target: string): OnnxGraph['edges'][number] => ({
  id: `${source}-${target}`,
  source,
  target,
})

// Linear chain A -> B -> C -> D
const chainGraph = (): OnnxGraph =>
  makeGraph(
    [
      makeNode({ id: 'A', opType: 'Conv' }),
      makeNode({ id: 'B', opType: 'Relu' }),
      makeNode({ id: 'C', opType: 'Conv' }),
      makeNode({ id: 'D', opType: 'Relu' }),
    ],
    [edge('A', 'B'), edge('B', 'C'), edge('C', 'D')],
  )

// Group 1: opCategoryColor

describe('opCategoryColor', () => {
  it('returns the convolution color for Conv', () => {
    expect(opCategoryColor('Conv')).toBe('#C0392B')
  })

  it('returns the activation color for Relu', () => {
    expect(opCategoryColor('Relu')).toBe('#52C57A')
  })

  it('returns the normalization color for BatchNormalization', () => {
    expect(opCategoryColor('BatchNormalization')).toBe('#3498DB')
  })

  it('returns the matrix color for Gemm and MatMul', () => {
    expect(opCategoryColor('Gemm')).toBe('#E67E22')
    expect(opCategoryColor('MatMul')).toBe('#E67E22')
  })

  it('returns the fallback color for an unknown op', () => {
    expect(opCategoryColor('UnknownOp')).toBe('rgba(255,255,255,0.15)')
  })
})

// Group 2: getAncestors and getDescendants

describe('getAncestors and getDescendants', () => {
  it('returns all ancestors of the last node in a chain', () => {
    const ancestors = getAncestors(chainGraph(), 'D')
    expect(ancestors.has('C')).toBe(true)
    expect(ancestors.has('B')).toBe(true)
    expect(ancestors.has('A')).toBe(true)
  })

  it('returns an empty set for a root node', () => {
    const ancestors = getAncestors(chainGraph(), 'A')
    expect(ancestors.size).toBe(0)
  })

  it('does not include the node itself in its ancestors', () => {
    const ancestors = getAncestors(chainGraph(), 'D')
    expect(ancestors.has('D')).toBe(false)
  })

  it('returns all descendants of the first node in a chain', () => {
    const descendants = getDescendants(chainGraph(), 'A')
    expect(descendants.has('B')).toBe(true)
    expect(descendants.has('C')).toBe(true)
    expect(descendants.has('D')).toBe(true)
  })

  it('returns an empty set for a leaf node', () => {
    const descendants = getDescendants(chainGraph(), 'D')
    expect(descendants.size).toBe(0)
  })

  it('does not include the node itself in its descendants', () => {
    const descendants = getDescendants(chainGraph(), 'A')
    expect(descendants.has('A')).toBe(false)
  })
})

// Group 3: computeGraphDepth

describe('computeGraphDepth', () => {
  it('returns 0 for an empty graph', () => {
    expect(computeGraphDepth(makeGraph([]))).toBe(0)
  })

  it('returns 0 for a single node with no edges', () => {
    expect(computeGraphDepth(makeGraph([makeNode({ id: 'A' })]))).toBe(0)
  })

  it('returns 2 for a linear chain of three nodes', () => {
    const graph = makeGraph(
      [makeNode({ id: 'A' }), makeNode({ id: 'B' }), makeNode({ id: 'C' })],
      [edge('A', 'B'), edge('B', 'C')],
    )
    expect(computeGraphDepth(graph)).toBe(2)
  })

  it('returns 2 for a diamond', () => {
    const graph = makeGraph(
      [makeNode({ id: 'A' }), makeNode({ id: 'B' }), makeNode({ id: 'C' }), makeNode({ id: 'D' })],
      [edge('A', 'B'), edge('A', 'C'), edge('B', 'D'), edge('C', 'D')],
    )
    expect(computeGraphDepth(graph)).toBe(2)
  })

  it('returns the longest path across two parallel paths', () => {
    const graph = makeGraph(
      [
        makeNode({ id: 'A' }),
        makeNode({ id: 'B' }),
        makeNode({ id: 'C' }),
        makeNode({ id: 'D' }),
        makeNode({ id: 'E' }),
        makeNode({ id: 'F' }),
        makeNode({ id: 'G' }),
        makeNode({ id: 'H' }),
        makeNode({ id: 'I' }),
      ],
      [
        edge('A', 'B'),
        edge('B', 'C'),
        edge('C', 'D'),
        edge('A', 'E'),
        edge('E', 'F'),
        edge('F', 'G'),
        edge('G', 'H'),
        edge('H', 'I'),
      ],
    )
    expect(computeGraphDepth(graph)).toBe(5)
  })
})
