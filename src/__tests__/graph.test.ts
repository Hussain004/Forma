import { describe, it, expect } from 'vitest'
import {
  toSelectableGraph,
  selectNode,
  deselectAll,
  getSelectedNodes,
  validateEdges,
} from '../lib/graphUtils'
import type { OnnxGraph } from '../lib/onnxTypes'

const makeGraph = (): OnnxGraph => ({
  nodes: [
    { id: 'input_0', opType: 'Input', inputs: [], outputs: ['x'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'conv_0', opType: 'Conv', inputs: ['x'], outputs: ['y'], attributes: {}, paramCount: 4096, estimatedSizeMB: 0.016 },
    { id: 'output_0', opType: 'Output', inputs: ['y'], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
  ],
  edges: [
    { id: 'e0', source: 'input_0', target: 'conv_0' },
    { id: 'e1', source: 'conv_0', target: 'output_0' },
  ],
  modelName: 'test_model',
  totalParams: 4096,
  totalSizeMB: 0.016,
})

describe('toSelectableGraph', () => {
  it('marks every node as unselected initially', () => {
    const sg = toSelectableGraph(makeGraph())
    expect(sg.nodes.every((n) => n.selected === false)).toBe(true)
  })

  it('preserves node count, edges, and metadata', () => {
    const g = makeGraph()
    const sg = toSelectableGraph(g)
    expect(sg.nodes).toHaveLength(g.nodes.length)
    expect(sg.edges).toEqual(g.edges)
    expect(sg.modelName).toBe(g.modelName)
    expect(sg.totalParams).toBe(g.totalParams)
    expect(sg.totalSizeMB).toBe(g.totalSizeMB)
  })

  it('does not mutate the input graph', () => {
    const g = makeGraph()
    const snapshot = JSON.stringify(g)
    toSelectableGraph(g)
    expect(JSON.stringify(g)).toBe(snapshot)
  })
})

describe('selectNode', () => {
  it('selecting a valid node marks only that node as selected', () => {
    const sg = selectNode(toSelectableGraph(makeGraph()), 'conv_0')
    expect(getSelectedNodes(sg).map((n) => n.id)).toEqual(['conv_0'])
    expect(sg.nodes.find((n) => n.id === 'input_0')?.selected).toBe(false)
    expect(sg.nodes.find((n) => n.id === 'output_0')?.selected).toBe(false)
  })

  it('selecting a non-existent nodeId leaves all nodes unselected', () => {
    const sg = selectNode(toSelectableGraph(makeGraph()), 'does_not_exist')
    expect(getSelectedNodes(sg)).toHaveLength(0)
  })

  it('calling selectNode twice on the same node keeps it selected (idempotent)', () => {
    const once = selectNode(toSelectableGraph(makeGraph()), 'conv_0')
    const twice = selectNode(once, 'conv_0')
    expect(getSelectedNodes(twice).map((n) => n.id)).toEqual(['conv_0'])
  })

  it('preserves earlier selections when selecting an additional node', () => {
    const sg = selectNode(selectNode(toSelectableGraph(makeGraph()), 'conv_0'), 'input_0')
    expect(getSelectedNodes(sg).map((n) => n.id).sort()).toEqual(['conv_0', 'input_0'])
  })

  it('does not mutate the input graph (pure function)', () => {
    const sg = toSelectableGraph(makeGraph())
    const snapshot = JSON.stringify(sg)
    selectNode(sg, 'conv_0')
    expect(JSON.stringify(sg)).toBe(snapshot)
  })
})

describe('deselectAll', () => {
  it('after selecting a node, deselectAll clears all selections', () => {
    const selected = selectNode(toSelectableGraph(makeGraph()), 'conv_0')
    const cleared = deselectAll(selected)
    expect(getSelectedNodes(cleared)).toHaveLength(0)
  })

  it('calling deselectAll on a graph with no selections is a no-op', () => {
    const sg = toSelectableGraph(makeGraph())
    const cleared = deselectAll(sg)
    expect(getSelectedNodes(cleared)).toHaveLength(0)
    expect(cleared.nodes).toHaveLength(sg.nodes.length)
  })

  it('does not mutate the input graph', () => {
    const selected = selectNode(toSelectableGraph(makeGraph()), 'conv_0')
    const snapshot = JSON.stringify(selected)
    deselectAll(selected)
    expect(JSON.stringify(selected)).toBe(snapshot)
  })
})

describe('getSelectedNodes', () => {
  it('returns empty array when nothing is selected', () => {
    expect(getSelectedNodes(toSelectableGraph(makeGraph()))).toEqual([])
  })

  it('returns only the selected nodes after selectNode is called', () => {
    const sg = selectNode(toSelectableGraph(makeGraph()), 'conv_0')
    const selected = getSelectedNodes(sg)
    expect(selected).toHaveLength(1)
    expect(selected[0].id).toBe('conv_0')
  })

  it('returns correct count when multiple nodes are selected', () => {
    const sg = selectNode(selectNode(toSelectableGraph(makeGraph()), 'conv_0'), 'output_0')
    expect(getSelectedNodes(sg)).toHaveLength(2)
  })
})

describe('validateEdges', () => {
  it('returns empty array for a valid graph (all edges reference existing nodes)', () => {
    expect(validateEdges(makeGraph())).toEqual([])
  })

  it('detects an edge with a non-existent source node', () => {
    const g = makeGraph()
    g.edges.push({ id: 'bad', source: 'ghost', target: 'conv_0' })
    const invalid = validateEdges(g)
    expect(invalid).toHaveLength(1)
    expect(invalid[0].id).toBe('bad')
  })

  it('detects an edge with a non-existent target node', () => {
    const g = makeGraph()
    g.edges.push({ id: 'bad', source: 'conv_0', target: 'ghost' })
    const invalid = validateEdges(g)
    expect(invalid).toHaveLength(1)
    expect(invalid[0].id).toBe('bad')
  })

  it('empty graph has no invalid edges', () => {
    const g: OnnxGraph = { nodes: [], edges: [], modelName: 'empty', totalParams: 0, totalSizeMB: 0 }
    expect(validateEdges(g)).toEqual([])
  })
})

describe('totalParams consistency', () => {
  it('totalParams in OnnxGraph equals the sum of all node paramCounts', () => {
    const g = makeGraph()
    const sum = g.nodes.reduce((acc, n) => acc + n.paramCount, 0)
    expect(g.totalParams).toBe(sum)
  })

  it('totalSizeMB equals totalParams * 4 / (1024 * 1024)', () => {
    const g = makeGraph()
    const expected = (g.totalParams * 4) / (1024 * 1024)
    expect(g.totalSizeMB).toBeCloseTo(expected, 2)
  })

  it('Input and Output nodes have paramCount of 0', () => {
    const g = makeGraph()
    const ioNodes = g.nodes.filter((n) => n.opType === 'Input' || n.opType === 'Output')
    expect(ioNodes.length).toBeGreaterThan(0)
    expect(ioNodes.every((n) => n.paramCount === 0)).toBe(true)
  })
})

describe('node structure', () => {
  it('every node has a non-empty id string', () => {
    const g = makeGraph()
    for (const n of g.nodes) {
      expect(typeof n.id).toBe('string')
      expect(n.id.length).toBeGreaterThan(0)
    }
  })

  it('every node has a defined opType', () => {
    const g = makeGraph()
    for (const n of g.nodes) {
      expect(n.opType).toBeDefined()
      expect(typeof n.opType).toBe('string')
    }
  })
})
