import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createElement } from 'react'
import App from '../App'
import { HistoryPanel } from '../components/HistoryPanel'
import { describeHistoryEntry, friendlyNodeLabel, type HistoryEntry } from '../lib/graphUtils'
import type { OnnxGraph } from '../lib/onnxTypes'

const makeMockWorker = () => ({
  postMessage: vi.fn(),
  terminate: vi.fn(),
  onmessage: null as ((event: MessageEvent) => void) | null,
  onerror: null as ((event: unknown) => void) | null,
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
})

let mockWorker: ReturnType<typeof makeMockWorker>

const graph: OnnxGraph = {
  modelName: 'history.onnx',
  totalParams: 0,
  totalSizeMB: 0,
  nodes: [
    { id: 'input_0', opType: 'Input', inputs: [], outputs: ['x'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'node_0_Relu', opType: 'Relu', inputs: ['x'], outputs: ['y'], attributes: { alpha: 1 }, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'output_0', opType: 'Output', inputs: ['y'], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
  ],
  edges: [
    { id: 'input_0->node_0_Relu@x', source: 'input_0', target: 'node_0_Relu', label: 'x' },
    { id: 'node_0_Relu->output_0@y', source: 'node_0_Relu', target: 'output_0', label: 'y' },
  ],
}

const entries: HistoryEntry[] = [
  { type: 'attr', nodeId: 'node_0_Relu', attrName: 'alpha', value: 2 },
  { type: 'delete', nodeId: 'node_0_Relu', nodeIndex: 0, keepInputPosition: null },
  {
    type: 'bulkDelete',
    deletions: [
      { type: 'delete', nodeId: 'node_0_Relu', nodeIndex: 0, keepInputPosition: null },
      { type: 'delete', nodeId: 'custom_1', nodeIndex: -1, keepInputPosition: null },
    ],
  },
  { type: 'insertPassthrough', targetNodeId: 'node_0_Relu', targetNodeIndex: 0, inputPosition: 0, newNodeId: 'passthrough_1' },
  { type: 'rewire', sourceNodeId: 'custom_1', sourceNodeIndex: -1, targetNodeId: 'node_0_Relu', targetNodeIndex: 0, inputPosition: 0 },
  { type: 'addNode', newNodeId: 'custom_1', newNodeIndex: 1, opType: 'Sigmoid', inputCount: 1, position: { x: 0, y: 0 } },
]

beforeEach(() => {
  mockWorker = makeMockWorker()
  vi.stubGlobal('Worker', vi.fn(function () { return mockWorker }))
  localStorage.clear()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

function loadApp() {
  render(createElement(App))
  act(() => {
    mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: graph } } as MessageEvent)
  })
}

async function waitForCanvas() {
  await act(() => new Promise((resolve) => setTimeout(resolve, 0)))
}

function placeCuratedNode(opType: string) {
  fireEvent.click(screen.getByText('Add Node'))
  fireEvent.mouseDown(screen.getByTestId(`add-node-option-${opType}`))
  const pane = document.querySelector('.react-flow__pane')
  expect(pane).not.toBeNull()
  fireEvent.click(pane as Element)
}

describe('v1.6 history labels', () => {
  it('formats friendly labels for original and generated ids', () => {
    expect(friendlyNodeLabel('node_3_MatMul')).toBe('MatMul')
    expect(friendlyNodeLabel('custom_2')).toBe('Custom node 2')
    expect(friendlyNodeLabel('passthrough_4')).toBe('Passthrough 4')
    expect(friendlyNodeLabel('unknown')).toBe('Node')
  })

  it('describes every history entry variant', () => {
    expect(entries.map(describeHistoryEntry)).toEqual([
      'Changed Relu alpha to 2',
      'Deleted Relu',
      'Deleted 2 nodes',
      'Inserted passthrough before Relu',
      'Rewired Relu input 1 to Custom node 1',
      'Added Sigmoid node',
    ])
  })
})

describe('HistoryPanel', () => {
  it('marks the current point, dims the future, and jumps on row click', () => {
    const onJump = vi.fn()
    render(createElement(HistoryPanel, { entries: entries.slice(0, 2), index: 1, onJump }))

    expect(screen.getByTestId('history-row-0')).toHaveAttribute('data-history-state', 'applied')
    expect(screen.getByTestId('history-row-1')).toHaveAttribute('data-history-state', 'current')
    expect(screen.getByTestId('history-row-2')).toHaveAttribute('data-history-state', 'future')

    fireEvent.click(screen.getByTestId('history-row-2'))
    expect(onJump).toHaveBeenCalledWith(2)
  })
})

describe('App history controls', () => {
  it('undoes and redoes an added node through both redo shortcuts', async () => {
    loadApp()
    await waitForCanvas()
    placeCuratedNode('Relu')

    expect(screen.getByText(/^2 NODES$/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /export modified \(1\)/i })).toBeInTheDocument()

    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    expect(screen.getByText(/^1 NODE$/i)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /export modified/i })).not.toBeInTheDocument()

    fireEvent.keyDown(window, { key: 'Z', ctrlKey: true, shiftKey: true })
    expect(screen.getByText(/^2 NODES$/i)).toBeInTheDocument()

    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    fireEvent.keyDown(window, { key: 'y', ctrlKey: true })
    expect(screen.getByText(/^2 NODES$/i)).toBeInTheDocument()
  })

  it('jumps to the original model and preserves redoable history in the timeline', async () => {
    loadApp()
    await waitForCanvas()
    placeCuratedNode('Relu')

    fireEvent.click(screen.getByTestId('history-tab'))
    expect(screen.getByTestId('history-row-1')).toHaveTextContent('Added Relu node')

    fireEvent.click(screen.getByTestId('history-row-0'))
    expect(screen.getByText(/^1 NODE$/i)).toBeInTheDocument()
    expect(screen.getByTestId('history-row-1')).toHaveAttribute('data-history-state', 'future')

    fireEvent.click(screen.getByTestId('history-row-1'))
    expect(screen.getByText(/^2 NODES$/i)).toBeInTheDocument()
  })

  it('truncates redo history when a new edit follows undo', async () => {
    loadApp()
    await waitForCanvas()
    placeCuratedNode('Relu')
    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    placeCuratedNode('Sigmoid')

    fireEvent.click(screen.getByTestId('history-tab'))
    expect(screen.getByTestId('history-row-1')).toHaveTextContent('Added Sigmoid node')
    expect(screen.queryByText('Added Relu node')).not.toBeInTheDocument()
  })

  it('reverts active edits without discarding the history timeline', async () => {
    loadApp()
    await waitForCanvas()
    placeCuratedNode('Relu')

    fireEvent.click(screen.getByTestId('revert-edits'))
    expect(screen.getByText(/^1 NODE$/i)).toBeInTheDocument()
    expect(screen.getByTestId('revert-edits')).toBeDisabled()

    fireEvent.click(screen.getByTestId('history-tab'))
    expect(screen.getByTestId('history-row-1')).toHaveAttribute('data-history-state', 'future')
  })

  it('does not undo while focus is in an input', async () => {
    loadApp()
    await waitForCanvas()
    placeCuratedNode('Relu')

    const filter = screen.getByRole('combobox')
    filter.focus()
    fireEvent.keyDown(filter, { key: 'z', ctrlKey: true })

    expect(screen.getByText(/^2 NODES$/i)).toBeInTheDocument()
  })
})
