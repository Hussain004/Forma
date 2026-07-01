import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act, fireEvent } from '@testing-library/react'
import { createElement } from 'react'
import App from '../App'
import { LayerInspector } from '../components/LayerInspector'
import type { OnnxGraph, OnnxNode } from '../lib/onnxTypes'

const makeMockWorker = () => ({
  postMessage: vi.fn(),
  terminate: vi.fn(),
  onmessage: null as ((e: MessageEvent) => void) | null,
  onerror: null as ((e: unknown) => void) | null,
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
})

let mockWorker: ReturnType<typeof makeMockWorker>

beforeEach(() => {
  mockWorker = makeMockWorker()
  vi.stubGlobal('Worker', vi.fn(function () { return mockWorker }))
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

const node = (id: string, opType: string, paramCount = 0): OnnxNode => ({
  id,
  opType,
  inputs: [],
  outputs: [],
  attributes: {},
  paramCount,
  estimatedSizeMB: 0,
})

// 10 Conv nodes plus a few other op types, so the "conv" filter has more than
// 8 matches (to exercise the dropdown cap) and other filters still match.
const buildGraph = (): OnnxGraph => ({
  nodes: [
    node('input_0', 'Input'),
    ...Array.from({ length: 10 }, (_, i) => node(`conv_${i}`, 'Conv', 1000)),
    node('relu_0', 'Relu'),
    node('bn_0', 'BatchNormalization', 500),
    node('pool_0', 'MaxPool'),
    node('output_0', 'Output'),
  ],
  edges: [],
  modelName: 'test.onnx',
  totalParams: 10500,
  totalSizeMB: 0.042,
})

const loadModel = (graph: OnnxGraph) => {
  act(() => {
    mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: graph } } as MessageEvent)
  })
}

// Feature 1: Layout direction toggle

describe('layout direction toggle', () => {
  beforeEach(() => {
    render(createElement(App))
    loadModel(buildGraph())
  })

  it('shows a TB button in the stats bar when a model is loaded', () => {
    expect(screen.getByRole('button', { name: 'TB' })).toBeInTheDocument()
  })

  it('changes the label to LR when clicked', () => {
    fireEvent.click(screen.getByRole('button', { name: 'TB' }))
    expect(screen.getByRole('button', { name: 'LR' })).toBeInTheDocument()
  })

  it('changes back to TB when clicked again', () => {
    fireEvent.click(screen.getByRole('button', { name: 'TB' }))
    fireEvent.click(screen.getByRole('button', { name: 'LR' }))
    expect(screen.getByRole('button', { name: 'TB' })).toBeInTheDocument()
  })
})

// Feature 2: Search results dropdown

describe('search dropdown', () => {
  beforeEach(() => {
    render(createElement(App))
    loadModel(buildGraph())
  })

  const filterInput = () => screen.getByPlaceholderText(/filter nodes/i)

  it('shows a dropdown when the filter matches nodes in the graph', () => {
    fireEvent.change(filterInput(), { target: { value: 'Conv' } })
    expect(screen.getByTestId('search-dropdown')).toBeInTheDocument()
  })

  it('hides the dropdown when the filter is cleared', () => {
    fireEvent.change(filterInput(), { target: { value: 'Conv' } })
    expect(screen.getByTestId('search-dropdown')).toBeInTheDocument()
    fireEvent.change(filterInput(), { target: { value: '' } })
    expect(screen.queryByTestId('search-dropdown')).not.toBeInTheDocument()
  })

  it('shows at most 8 results', () => {
    // 10 Conv nodes match, so the dropdown must cap the list at 8.
    fireEvent.change(filterInput(), { target: { value: 'Conv' } })
    expect(screen.getAllByTestId('search-result')).toHaveLength(8)
  })

  it('highlights the next item when ArrowDown is pressed', () => {
    fireEvent.change(filterInput(), { target: { value: 'Conv' } })
    const selectedCount = () =>
      screen.getAllByRole('option').filter((o) => o.getAttribute('aria-selected') === 'true').length
    expect(selectedCount()).toBe(0)
    fireEvent.keyDown(filterInput(), { key: 'ArrowDown' })
    expect(selectedCount()).toBe(1)
  })
})

// Feature 3: Copy to clipboard

describe('copy to clipboard', () => {
  let writeText: ReturnType<typeof vi.fn>

  beforeEach(() => {
    writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    })
  })

  const convNode: OnnxNode = {
    id: 'n1',
    opType: 'Conv',
    inputs: ['x'],
    outputs: ['y'],
    attributes: {},
    paramCount: 1000,
    estimatedSizeMB: 0.004,
  }

  it('shows a COPY button when a node is selected', () => {
    render(createElement(LayerInspector, { node: convNode, quantizeEstimate: null }))
    expect(screen.getByRole('button', { name: 'COPY' })).toBeInTheDocument()
  })

  it('calls navigator.clipboard.writeText when COPY is clicked', () => {
    render(createElement(LayerInspector, { node: convNode, quantizeEstimate: null }))
    fireEvent.click(screen.getByRole('button', { name: 'COPY' }))
    expect(writeText).toHaveBeenCalled()
  })

  it('copies text that contains the node opType', () => {
    render(createElement(LayerInspector, { node: convNode, quantizeEstimate: null }))
    fireEvent.click(screen.getByRole('button', { name: 'COPY' }))
    expect(writeText.mock.calls[0][0]).toContain('Conv')
  })
})
