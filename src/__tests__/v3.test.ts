import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useOnnxWorker } from '../hooks/useOnnxWorker'
import type { SelectableGraph, SelectableNode } from '../lib/graphUtils'
import type { OnnxGraph } from '../lib/onnxTypes'

import { filterGraph, excludeNode, includeNode } from '../lib/graphUtils'
import { estimateInt8Size } from '../lib/quantize'

// Test node factory. SelectableNode does not yet carry dimmed/excluded,
// so the helper accepts them as optional extras for forward compatibility.
type V3Node = SelectableNode & { dimmed?: boolean; excluded?: boolean }

const defaultNode: SelectableNode = {
  id: 'n0',
  opType: 'Relu',
  inputs: [],
  outputs: [],
  attributes: {},
  paramCount: 0,
  estimatedSizeMB: 0,
  selected: false,
}

const makeNode = (overrides: Partial<V3Node> = {}): V3Node => ({ ...defaultNode, ...overrides })

const makeGraph = (nodes: V3Node[], edges: OnnxGraph['edges'] = []): SelectableGraph => ({
  nodes,
  edges,
  modelName: 'test',
  totalParams: 0,
  totalSizeMB: 0,
})

const sampleGraph = (): SelectableGraph =>
  makeGraph([
    makeNode({ id: 'conv1', opType: 'Conv' }),
    makeNode({ id: 'relu1', opType: 'Relu' }),
    makeNode({ id: 'add1', opType: 'Add' }),
  ])

const findNode = (graph: SelectableGraph, id: string): V3Node =>
  graph.nodes.find((n) => n.id === id) as V3Node

// Group 1: filterGraph

describe('filterGraph', () => {
  it('marks every node as not dimmed for an empty query', () => {
    const result = filterGraph(sampleGraph(), '')
    expect(result.nodes.every((n: V3Node) => n.dimmed === false)).toBe(true)
  })

  it('keeps an exact opType match undimmed and dims the rest', () => {
    const result = filterGraph(sampleGraph(), 'Conv')
    expect(findNode(result, 'conv1').dimmed).toBe(false)
    expect(findNode(result, 'relu1').dimmed).toBe(true)
    expect(findNode(result, 'add1').dimmed).toBe(true)
  })

  it('matches a partial opType substring', () => {
    const result = filterGraph(sampleGraph(), 'Con')
    expect(findNode(result, 'conv1').dimmed).toBe(false)
  })

  it('matches opType case-insensitively', () => {
    const result = filterGraph(sampleGraph(), 'conv')
    expect(findNode(result, 'conv1').dimmed).toBe(false)
  })

  it('dims all nodes when nothing matches', () => {
    const result = filterGraph(sampleGraph(), 'Softmax')
    expect(result.nodes.every((n: V3Node) => n.dimmed === true)).toBe(true)
  })

  it('matches against the node id as well as the opType', () => {
    const result = filterGraph(sampleGraph(), 'relu1')
    expect(findNode(result, 'relu1').dimmed).toBe(false)
    expect(findNode(result, 'conv1').dimmed).toBe(true)
  })

  it('returns a new graph without mutating the input', () => {
    const input = sampleGraph()
    const result = filterGraph(input, 'Conv')
    expect(result).not.toBe(input)
    expect(result.nodes).not.toBe(input.nodes)
    expect(input.nodes.every((n: V3Node) => n.dimmed === undefined)).toBe(true)
  })

  it('passes edges through unchanged', () => {
    const edges = [{ id: 'e1', source: 'conv1', target: 'relu1' }]
    const input = makeGraph(
      [makeNode({ id: 'conv1', opType: 'Conv' }), makeNode({ id: 'relu1', opType: 'Relu' })],
      edges,
    )
    const result = filterGraph(input, 'Conv')
    expect(result.edges).toEqual(edges)
  })
})

// Group 2: excludeNode and includeNode

describe('excludeNode and includeNode', () => {
  it('sets excluded to true on the matching node', () => {
    const result = excludeNode(sampleGraph(), 'relu1')
    expect(findNode(result, 'relu1').excluded).toBe(true)
  })

  it('does not affect other nodes when excluding', () => {
    const result = excludeNode(sampleGraph(), 'relu1')
    expect(findNode(result, 'conv1').excluded).not.toBe(true)
    expect(findNode(result, 'add1').excluded).not.toBe(true)
  })

  it('is idempotent when excluding the same node twice', () => {
    const once = excludeNode(sampleGraph(), 'relu1')
    const twice = excludeNode(once, 'relu1')
    expect(findNode(twice, 'relu1').excluded).toBe(true)
  })

  it('returns the graph unchanged for a non-existent node id', () => {
    const input = sampleGraph()
    const result = excludeNode(input, 'does-not-exist')
    expect(result.nodes.map((n: V3Node) => n.id)).toEqual(input.nodes.map((n) => n.id))
    expect(result.nodes.every((n: V3Node) => n.excluded !== true)).toBe(true)
  })

  it('sets excluded back to false via includeNode', () => {
    const excluded = excludeNode(sampleGraph(), 'relu1')
    const included = includeNode(excluded, 'relu1')
    expect(findNode(included, 'relu1').excluded).toBe(false)
  })

  it('returns a new graph without mutating the input when excluding', () => {
    const input = sampleGraph()
    const result = excludeNode(input, 'relu1')
    expect(result).not.toBe(input)
    expect(findNode(input, 'relu1').excluded).not.toBe(true)
  })
})

// Group 3: estimateInt8Size

describe('estimateInt8Size', () => {
  it('returns 0 for an element count of 0', () => {
    expect(estimateInt8Size(0)).toBe(0)
  })

  it('computes a small element count in MB', () => {
    expect(estimateInt8Size(524288)).toBeCloseTo(524288 / (1024 * 1024), 10)
  })

  it('returns exactly 1.0 MB for 1048576 elements', () => {
    expect(estimateInt8Size(1048576)).toBe(1.0)
  })

  it('returns a finite number for a large element count', () => {
    const result = estimateInt8Size(1073741824)
    expect(Number.isFinite(result)).toBe(true)
    expect(result).toBeCloseTo(1024, 10)
  })
})

// Group 4: exportModel worker command

const makeMockWorker = () => ({
  postMessage: vi.fn(),
  terminate: vi.fn(),
  onmessage: null as ((e: MessageEvent) => void) | null,
  onerror: null as ((e: unknown) => void) | null,
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
})

const emptyGraph: OnnxGraph = { nodes: [], edges: [], modelName: 'test', totalParams: 0, totalSizeMB: 0 }

describe('useOnnxWorker exportModel command', () => {
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

  it('exposes exportModel as a function', () => {
    const { result } = renderHook(() => useOnnxWorker())
    expect(typeof (result.current as Record<string, unknown>).exportModel).toBe('function')
  })

  it('posts an EXPORT command to the worker when exportModel is called', () => {
    const { result } = renderHook(() => useOnnxWorker())
    act(() => {
      (result.current as { exportModel: () => void }).exportModel()
    })
    expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'EXPORT' }),
    )
  })

  it('posts an EXPORT command after a model has been loaded', () => {
    const { result } = renderHook(() => useOnnxWorker())
    act(() => { result.current.loadModel(new ArrayBuffer(8), 'model.onnx') })
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: emptyGraph } } as MessageEvent)
    })
    act(() => {
      (result.current as { exportModel: () => void }).exportModel()
    })
    expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'EXPORT' }),
    )
  })
})
