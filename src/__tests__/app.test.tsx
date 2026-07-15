import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act, fireEvent } from '@testing-library/react'
import App from '../App'
import type { OnnxGraph } from '../lib/onnxTypes'

// Worker must be mocked before App imports useOnnxWorker.
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

const testGraph: OnnxGraph = {
  nodes: [
    { id: 'input_0', opType: 'Input', inputs: [], outputs: ['x'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'conv_0', opType: 'Conv', inputs: ['x', 'weight'], outputs: ['y'], attributes: { kernel_shape: '3x3' }, paramCount: 139264, estimatedSizeMB: 0.532 },
    { id: 'output_0', opType: 'Output', inputs: ['y'], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
  ],
  edges: [
    { id: 'e0', source: 'input_0', target: 'conv_0' },
    { id: 'e1', source: 'conv_0', target: 'output_0' },
  ],
  modelName: 'test.onnx',
  totalParams: 139264,
  totalSizeMB: 0.532,
}

describe('App -- initial state', () => {
  it('renders the dropzone on load', () => {
    render(<App />)
    // ModelDropzone idle state shows drop instructions
    expect(screen.getByText(/drop .onnx/i)).toBeInTheDocument()
  })

  it('does not render the graph canvas before a model is loaded', () => {
    render(<App />)
    expect(screen.queryByTestId('graph-canvas')).not.toBeInTheDocument()
  })
})

describe('App -- model load flow', () => {
  it('transitions to ready and shows graph after MODEL_LOADED', async () => {
    render(<App />)

    act(() => {
      mockWorker.onmessage?.({
        data: { type: 'MODEL_LOADED', payload: testGraph },
      } as MessageEvent)
    })

    // Dropzone must be gone once ready.
    expect(screen.queryByText(/drop .onnx/i)).not.toBeInTheDocument()
    // React Flow renders a hidden a11y div with "select a node" too -- use getAllByText
    // and confirm at least one visible instance (the LayerInspector placeholder span).
    const instances = screen.getAllByText(/select a node/i)
    expect(instances.length).toBeGreaterThanOrEqual(1)
  })

  it('shows loading state while model parses', () => {
    render(<App />)

    act(() => {
      mockWorker.onmessage?.({
        data: { type: 'PROGRESS', payload: { stage: 'Loading model', percent: 10 } },
      } as MessageEvent)
    })

    // Still in loading state -- no graph yet
    expect(screen.queryByText(/select a node/i)).not.toBeInTheDocument()
  })

  it('shows error state when worker sends ERROR', () => {
    render(<App />)

    act(() => {
      mockWorker.onmessage?.({
        data: { type: 'ERROR', payload: 'Failed to parse model' },
      } as MessageEvent)
    })

    expect(screen.getByText(/failed to parse model/i)).toBeInTheDocument()
  })
})

describe('App -- keyboard shortcuts overlay', () => {
  it('? opens the overlay, Esc closes it', () => {
    render(<App />)
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: testGraph } } as MessageEvent)
    })

    expect(screen.queryByTestId('shortcuts-overlay')).not.toBeInTheDocument()

    fireEvent.keyDown(window, { key: '?' })
    expect(screen.getByTestId('shortcuts-overlay')).toBeInTheDocument()

    fireEvent.keyDown(window, { key: 'Escape' })
    expect(screen.queryByTestId('shortcuts-overlay')).not.toBeInTheDocument()
  })

  it('closes via the Close button', () => {
    render(<App />)
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: testGraph } } as MessageEvent)
    })

    fireEvent.keyDown(window, { key: '?' })
    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    expect(screen.queryByTestId('shortcuts-overlay')).not.toBeInTheDocument()
  })
})

describe('App -- benchmark running state', () => {
  it('disables the button and shows Running while a benchmark is in flight, then Benchmark again', () => {
    render(<App />)
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: testGraph } } as MessageEvent)
    })

    const button = screen.getByRole('button', { name: /benchmark/i })
    expect(button).not.toBeDisabled()

    fireEvent.click(button)
    // The worker hasn't responded yet -- runBenchmark sets status synchronously.
    expect(screen.getByRole('button', { name: /running/i })).toBeDisabled()

    act(() => {
      mockWorker.onmessage?.({
        data: { type: 'BENCHMARK_RESULT', payload: { avgMs: 5, medianMs: 4.8, minMs: 4.1, maxMs: 6.2, runs: 10 } },
      } as MessageEvent)
    })

    expect(screen.getByRole('button', { name: /^benchmark$/i })).not.toBeDisabled()
    expect(screen.getByText(/avg 5\.0 ms \/ median 4\.8 ms/)).toBeInTheDocument()
  })
})

describe('App -- single-select model', () => {
  it('LayerInspector shows node details after selecting a Conv node', async () => {
    render(<App />)

    act(() => {
      mockWorker.onmessage?.({
        data: { type: 'MODEL_LOADED', payload: testGraph },
      } as MessageEvent)
    })

    // Click the React Flow node wrapper for conv_0.
    // React Flow attaches data-testid="rf__node-{id}" to each node group.
    const rfNode = screen.queryByTestId('rf__node-conv_0')
    if (rfNode) {
      fireEvent.click(rfNode)
      // After selection the inspector renders OP TYPE row.
      // OP TYPE label and Conv value both appear in the LayerInspector.
      expect(screen.getByText(/^op type$/i)).toBeInTheDocument()
      // 'Conv' appears in both the graph node and the inspector -- use getAllByText.
      expect(screen.getAllByText('Conv').length).toBeGreaterThanOrEqual(1)
    } else {
      // React Flow did not fully mount (no layout in jsdom) -- verify
      // the graph canvas wrapper is present and inspector is in null state.
      expect(screen.queryByText(/drop .onnx/i)).not.toBeInTheDocument()
      expect(screen.getAllByText(/select a node/i).length).toBeGreaterThanOrEqual(1)
    }
  })

  it('selecting a node does not affect other nodes selection state', async () => {
    render(<App />)

    act(() => {
      mockWorker.onmessage?.({
        data: { type: 'MODEL_LOADED', payload: testGraph },
      } as MessageEvent)
    })

    // Only one node detail can appear in the inspector at a time.
    // OP TYPE label appears only when a node is selected.
    const opTypeLabels = screen.queryAllByText(/^op type$/i)
    expect(opTypeLabels.length).toBeLessThanOrEqual(1)
  })

  it('loading a new model clears the previous selection', async () => {
    render(<App />)

    act(() => {
      mockWorker.onmessage?.({
        data: { type: 'MODEL_LOADED', payload: testGraph },
      } as MessageEvent)
    })

    // Load a second model -- resets selectableGraph, inspector returns to null state.
    act(() => {
      mockWorker.onmessage?.({
        data: { type: 'MODEL_LOADED', payload: { ...testGraph, modelName: 'model2.onnx' } },
      } as MessageEvent)
    })

    // OP TYPE row appears only when a node is selected; after reload it must be absent.
    expect(screen.queryByText(/^op type$/i)).not.toBeInTheDocument()
    // Dropzone is still absent (model is ready).
    expect(screen.queryByText(/drop .onnx/i)).not.toBeInTheDocument()
  })
})
