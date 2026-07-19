import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createElement } from 'react'
import App from '../App'
import { ChangeLogPanel } from '../components/ChangeLogPanel'
import { GraphCanvas } from '../components/GraphCanvas'
import { buildGraphDiff, toSelectableGraph, type HistoryEntry } from '../lib/graphUtils'
import type { OnnxGraph } from '../lib/onnxTypes'

const graph: OnnxGraph = {
  modelName: 'diff.onnx',
  totalParams: 0,
  totalSizeMB: 0,
  nodes: [
    { id: 'input_0', opType: 'Input', inputs: [], outputs: ['x'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'node_0_Relu', opType: 'Relu', inputs: ['x'], outputs: ['y'], attributes: { alpha: 1 }, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'node_1_Sigmoid', opType: 'Sigmoid', inputs: ['y'], outputs: ['z'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'output_0', opType: 'Output', inputs: ['z'], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
  ],
  edges: [
    { id: 'input_0->node_0_Relu@x', source: 'input_0', target: 'node_0_Relu', label: 'x' },
    { id: 'node_0_Relu->node_1_Sigmoid@y', source: 'node_0_Relu', target: 'node_1_Sigmoid', label: 'y' },
    { id: 'node_1_Sigmoid->output_0@z', source: 'node_1_Sigmoid', target: 'output_0', label: 'z' },
  ],
}

const entries: HistoryEntry[] = [
  { type: 'attr', nodeId: 'node_0_Relu', attrName: 'alpha', value: 2 },
  { type: 'addNode', newNodeId: 'custom_1', newNodeIndex: 1, opType: 'Tanh', inputCount: 1, position: { x: 8, y: 8 } },
]

const makeMockWorker = () => ({
  postMessage: vi.fn(),
  terminate: vi.fn(),
  onmessage: null as ((event: MessageEvent) => void) | null,
  onerror: null as ((event: unknown) => void) | null,
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
})

let mockWorker: ReturnType<typeof makeMockWorker>

beforeEach(() => {
  mockWorker = makeMockWorker()
  vi.stubGlobal('Worker', vi.fn(function () { return mockWorker }))
  localStorage.clear()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

describe('v1.7 graph comparison', () => {
  it('restores deleted nodes and marks changed and removed connections', () => {
    const original = toSelectableGraph(graph)
    const current = {
      ...original,
      nodes: original.nodes
        .filter((node) => node.id !== 'node_0_Relu')
        .map((node) => node.id === 'node_1_Sigmoid' ? { ...node, inputs: ['x'] } : node),
      edges: [
        { id: 'input_0->node_1_Sigmoid@x', source: 'input_0', target: 'node_1_Sigmoid', label: 'x' },
        graph.edges[2],
      ],
    }

    const diff = buildGraphDiff(original, current)
    expect(diff.nodes.find((node) => node.id === 'node_0_Relu')?.diffStatus).toBe('deleted')
    expect(diff.edges.find((edge) => edge.id === 'input_0->node_1_Sigmoid@x')?.diffStatus).toBe('changed')
    expect(diff.edges.find((edge) => edge.id === graph.edges[2].id)?.diffStatus).toBeUndefined()
    expect(diff.edges.filter((edge) => edge.diffStatus === 'removed')).toHaveLength(2)
  })

  it('renders deleted nodes as ghosted comparison nodes', () => {
    render(createElement(GraphCanvas, {
      onnxNodes: [{ ...graph.nodes[1], diffStatus: 'deleted' as const }],
      onnxEdges: [],
      selectedNodeId: null,
      onNodeSelect: vi.fn(),
      diffActive: true,
    }))

    expect(screen.getByTestId('deleted-node-badge')).toHaveTextContent('DEL')
    expect(screen.getByTestId('diff-legend')).toBeInTheDocument()
  })
})

describe('v1.7 plain-text change log', () => {
  it('lists active edits and copies the same plain text', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    const onCopy = vi.fn()

    render(createElement(ChangeLogPanel, { entries, onCopy }))
    expect(screen.getByTestId('change-log-text')).toHaveTextContent('01 Changed Relu alpha to 2')
    expect(screen.getByTestId('change-log-text')).toHaveTextContent('02 Added Tanh node')

    fireEvent.click(screen.getByRole('button', { name: 'Copy' }))
    await act(() => Promise.resolve())
    expect(writeText).toHaveBeenCalledWith('01  Changed Relu alpha to 2\n02  Added Tanh node')
    expect(onCopy).toHaveBeenCalled()
  })

  it('tracks the active history prefix in the application', async () => {
    render(createElement(App))
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: graph } } as MessageEvent)
    })
    await act(() => new Promise((resolve) => setTimeout(resolve, 0)))

    const toggle = screen.getByTestId('diff-toggle')
    expect(toggle).toBeDisabled()

    fireEvent.click(screen.getByText('Add Node'))
    fireEvent.mouseDown(screen.getByTestId('add-node-option-Relu'))
    const pane = document.querySelector('.react-flow__pane')
    expect(pane).not.toBeNull()
    fireEvent.click(pane as Element)

    expect(toggle).toBeEnabled()
    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByTestId('diff-legend')).toBeInTheDocument()

    fireEvent.click(screen.getByTestId('changes-tab'))
    expect(screen.getByTestId('change-log-text')).toHaveTextContent('Added Relu node')

    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    expect(screen.getByTestId('change-log-text')).toHaveTextContent('No active changes.')
    expect(screen.queryByTestId('diff-legend')).not.toBeInTheDocument()
  })
})
