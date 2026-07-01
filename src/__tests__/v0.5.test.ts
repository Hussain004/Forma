import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act, fireEvent } from '@testing-library/react'
import { createElement } from 'react'
import App from '../App'
import { LayerInspector } from '../components/LayerInspector'
import type { OnnxGraph, OnnxNode } from '../lib/onnxTypes'
import { computeOpCounts } from '../lib/graphUtils'

// ---- Shared helpers ----

const makeNode = (id: string, opType: string): OnnxNode => ({
  id,
  opType,
  inputs: [],
  outputs: [],
  attributes: {},
  paramCount: 0,
  estimatedSizeMB: 0,
})

const makeMockWorker = () => ({
  postMessage: vi.fn(),
  terminate: vi.fn(),
  onmessage: null as ((e: MessageEvent) => void) | null,
  onerror: null as ((e: unknown) => void) | null,
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
})

const testGraph: OnnxGraph = {
  nodes: [
    { id: 'input_0', opType: 'Input', inputs: [], outputs: ['x'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'conv_0', opType: 'Conv', inputs: ['x', 'weight'], outputs: ['y'], attributes: { kernel_shape: '3x3' }, paramCount: 139264, estimatedSizeMB: 0.532 },
    { id: 'relu_0', opType: 'Relu', inputs: ['y'], outputs: ['z'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'output_0', opType: 'Output', inputs: ['z'], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
  ],
  edges: [
    { id: 'e0', source: 'input_0', target: 'conv_0' },
    { id: 'e1', source: 'conv_0', target: 'relu_0' },
    { id: 'e2', source: 'relu_0', target: 'output_0' },
  ],
  modelName: 'test.onnx',
  totalParams: 139264,
  totalSizeMB: 0.532,
}

// ---- Feature 1: computeOpCounts ----

describe('computeOpCounts', () => {
  it('returns an empty object for an empty node array', () => {
    expect(computeOpCounts([])).toEqual({})
  })

  it('returns a count of 1 for a single compute node', () => {
    expect(computeOpCounts([makeNode('conv_0', 'Conv')])).toEqual({ Conv: 1 })
  })

  it('counts multiple nodes of the same op type', () => {
    const nodes = [makeNode('c0', 'Conv'), makeNode('c1', 'Conv'), makeNode('c2', 'Conv')]
    expect(computeOpCounts(nodes).Conv).toBeGreaterThan(1)
    expect(computeOpCounts(nodes)).toEqual({ Conv: 3 })
  })

  it('excludes Input and Output op types', () => {
    const nodes = [
      makeNode('input_0', 'Input'),
      makeNode('conv_0', 'Conv'),
      makeNode('output_0', 'Output'),
    ]
    const result = computeOpCounts(nodes)
    expect(result).toEqual({ Conv: 1 })
    expect(result.Input).toBeUndefined()
    expect(result.Output).toBeUndefined()
  })

  it('gives each mixed op type its own count', () => {
    const nodes = [
      makeNode('c0', 'Conv'),
      makeNode('c1', 'Conv'),
      makeNode('r0', 'Relu'),
      makeNode('a0', 'Add'),
    ]
    expect(computeOpCounts(nodes)).toEqual({ Conv: 2, Relu: 1, Add: 1 })
  })
})

// ---- Feature 2: computeOpCounts sorted entries ----

describe('computeOpCounts sorted entries', () => {
  it('Object.entries sorted by count descending yields the correct order', () => {
    const nodes = [
      makeNode('c0', 'Conv'),
      makeNode('c1', 'Conv'),
      makeNode('c2', 'Conv'),
      makeNode('r0', 'Relu'),
      makeNode('r1', 'Relu'),
      makeNode('a0', 'Add'),
    ]
    const result = computeOpCounts(nodes)
    const sorted = Object.entries(result).sort((a, b) => b[1] - a[1])
    expect(sorted.map(([op]) => op)).toEqual(['Conv', 'Relu', 'Add'])
    expect(sorted.map(([, count]) => count)).toEqual([3, 2, 1])
  })
})

// ---- Feature 3: Keyboard shortcuts in App ----

describe('App keyboard shortcuts', () => {
  let mockWorker: ReturnType<typeof makeMockWorker>

  beforeEach(() => {
    mockWorker = makeMockWorker()
    vi.stubGlobal('Worker', vi.fn(function () { return mockWorker }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  const loadModel = () => {
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: testGraph } } as MessageEvent)
    })
  }

  it('pressing Escape clears the filter input', () => {
    render(createElement(App))
    loadModel()

    const filter = screen.getByPlaceholderText(/filter nodes/i)
    fireEvent.change(filter, { target: { value: 'conv' } })
    expect(filter).toHaveValue('conv')

    fireEvent.keyDown(document.body, { key: 'Escape' })
    expect(filter).toHaveValue('')
  })

  it('pressing / focuses the filter input when the graph is visible', () => {
    render(createElement(App))
    loadModel()

    const filter = screen.getByPlaceholderText(/filter nodes/i)
    expect(filter).not.toHaveFocus()

    fireEvent.keyDown(document.body, { key: '/' })
    expect(filter).toHaveFocus()
  })

  it('pressing Escape deselects the selected node', () => {
    render(createElement(App))
    loadModel()

    // Selecting a node happens through a React Flow node click.
    const rfNode = screen.getByTestId('rf__node-conv_0')
    fireEvent.click(rfNode)
    // The OP TYPE row appears only when a node is selected.
    expect(screen.getByText(/^op type$/i)).toBeInTheDocument()

    fireEvent.keyDown(document.body, { key: 'Escape' })
    // After deselect the inspector returns to its null state, so OP TYPE is gone.
    expect(screen.queryByText(/^op type$/i)).not.toBeInTheDocument()
  })
})

// ---- Feature 4: Op type histogram in LayerInspector ----

describe('LayerInspector op type histogram', () => {
  const modelStats = { opCounts: { Conv: 3, Relu: 2, MaxPool: 1 }, totalNodes: 6 }

  it('renders op type counts when no node is selected and modelStats is provided', () => {
    render(createElement(LayerInspector, { node: null, modelStats }))

    // The op type summary lists each op type alongside its count.
    expect(screen.getByText('Conv')).toBeInTheDocument()
    expect(screen.getByText('Relu')).toBeInTheDocument()
    expect(screen.getByText('MaxPool')).toBeInTheDocument()
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument()
    expect(screen.getByText('1')).toBeInTheDocument()
  })

  it('does not show the histogram when a node is selected', () => {
    const convNode: OnnxNode = {
      id: 'conv_0',
      opType: 'Conv',
      inputs: ['x'],
      outputs: ['y'],
      attributes: {},
      paramCount: 4096,
      estimatedSizeMB: 0.016,
    }

    const { rerender } = render(createElement(LayerInspector, { node: null, modelStats }))
    // The histogram summary is present while no node is selected.
    expect(screen.getByText(/model summary/i)).toBeInTheDocument()

    rerender(createElement(LayerInspector, { node: convNode, modelStats }))
    // Once a node is selected the histogram summary must not render.
    expect(screen.queryByText(/model summary/i)).not.toBeInTheDocument()
    // The node detail view (OP TYPE row) is shown instead.
    expect(screen.getByText(/^op type$/i)).toBeInTheDocument()
  })
})
