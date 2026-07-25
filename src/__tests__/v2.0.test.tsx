import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import {
  ShareLinkError,
  createShareHash,
  hashModelBuffer,
  parseShareHash,
  restoreSharedHistory,
} from '../lib/shareLinks'
import type { HistoryEntry } from '../lib/graphUtils'
import type { OnnxGraph } from '../lib/onnxTypes'

const HASH = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'

const graph: OnnxGraph = {
  modelName: 'shared.onnx',
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
  { type: 'addNode', newNodeId: 'custom_1', newNodeIndex: 1, opType: 'Tanh', inputCount: 1, position: { x: 24, y: 48 } },
  { type: 'rewire', sourceNodeId: 'node_0_Relu', sourceNodeIndex: 0, targetNodeId: 'custom_1', targetNodeIndex: -1, inputPosition: 0 },
  { type: 'insertPassthrough', targetNodeId: 'node_1_Sigmoid', targetNodeIndex: 1, inputPosition: 0, newNodeId: 'passthrough_3' },
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
  window.history.replaceState(null, '', '/')
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
  window.history.replaceState(null, '', '/')
})

describe('v2.0 share-link codec', () => {
  it('round-trips every shared edit and restores live node ids and counters', () => {
    const hash = createShareHash(HASH, graph.modelName, entries)
    const payload = parseShareHash(hash)

    expect(hash).toMatch(/^#s=[A-Za-z0-9_-]+$/)
    expect(payload?.modelHash).toBe(HASH)
    expect(payload?.modelName).toBe('shared.onnx')

    const restored = restoreSharedHistory(payload!, graph)
    expect(restored.entries).toEqual(entries)
    expect(restored.customNodeCounter).toBe(1)
    expect(restored.passthroughCounter).toBe(3)
  })

  it('preserves grouped bulk deletion as one history entry', () => {
    const bulk: HistoryEntry[] = [{
      type: 'bulkDelete',
      deletions: [
        { type: 'delete', nodeId: 'node_1_Sigmoid', nodeIndex: 1, keepInputPosition: null },
        { type: 'delete', nodeId: 'node_0_Relu', nodeIndex: 0, keepInputPosition: null },
      ],
    }]
    const payload = parseShareHash(createShareHash(HASH, graph.modelName, bulk))
    const restored = restoreSharedHistory(payload!, graph)
    expect(restored.entries).toEqual(bulk)
  })

  it('rejects malformed, oversized, and unsupported links', () => {
    expect(() => parseShareHash('#s=%%%')).toThrow(ShareLinkError)
    expect(() => parseShareHash('#s=' + 'A'.repeat(100_001))).toThrow(/too large/i)

    const unsupported = btoa(JSON.stringify({ v: 2, h: HASH, e: [['d', 0, null]] }))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/g, '')
    expect(() => parseShareHash('#s=' + unsupported)).toThrow(/unsupported version/i)
  })

  it('computes a stable base64url SHA-256 fingerprint', async () => {
    const bytes = new TextEncoder().encode('abc')
    await expect(hashModelBuffer(bytes.buffer)).resolves.toBe('ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0')
  })
})

describe('v2.0 shared-link App flow', () => {
  it('prompts for the original model named in the link', () => {
    window.history.replaceState(null, '', createShareHash(HASH, graph.modelName, entries))
    render(<App />)

    expect(screen.getByText('SHARED EDIT SEQUENCE / ORIGINAL FILE')).toBeInTheDocument()
    expect(screen.getByText('Load shared.onnx')).toBeInTheDocument()
    expect(screen.getByText(/Expected SHA-256 AAAAAAAAAAAA/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /load sample model/i })).not.toBeInTheDocument()
  })

  it('rejects a model with a different fingerprint before worker loading', async () => {
    window.history.replaceState(null, '', createShareHash(HASH, graph.modelName, entries))
    render(<App />)

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File([new Uint8Array([1, 2, 3])], 'wrong.onnx')
    fireEvent.change(input, { target: { files: [file] } })

    expect(await screen.findByText(/Model fingerprint mismatch/)).toBeInTheDocument()
    expect(mockWorker.postMessage).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: 'LOAD_MODEL' }),
      expect.anything(),
    )
  })

  it('replays verified edits into the active history timeline', async () => {
    const abcHash = 'ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0'
    const sharedEntries: HistoryEntry[] = [
      { type: 'attr', nodeId: 'node_0_Relu', attrName: 'alpha', value: 2 },
    ]
    window.history.replaceState(null, '', createShareHash(abcHash, graph.modelName, sharedEntries))
    render(<App />)

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(input, {
      target: { files: [new File([new TextEncoder().encode('abc')], 'shared.onnx')] },
    })
    await waitFor(() => expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'LOAD_MODEL' }),
      expect.anything(),
    ))

    act(() => {
      mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: graph } } as MessageEvent)
    })

    expect(await screen.findByText('Changed Relu alpha to 2')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /export modified \(1\)/i })).toBeInTheDocument()
    expect(screen.getByTestId('announcement')).toHaveTextContent('Verified and replayed 1 shared edit')
  })

  it('copies a hash-bound link for active edits', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { ...navigator, clipboard: { writeText } })
    render(<App />)

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File([new TextEncoder().encode('abc')], 'shared.onnx')
    fireEvent.change(input, { target: { files: [file] } })
    await waitFor(() => expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'LOAD_MODEL' }),
      expect.anything(),
    ))

    act(() => {
      mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: graph } } as MessageEvent)
    })
    await act(() => new Promise((resolve) => setTimeout(resolve, 0)))

    fireEvent.click(screen.getByText('Add Node'))
    fireEvent.mouseDown(screen.getByTestId('add-node-option-Tanh'))
    const pane = document.querySelector('.react-flow__pane')
    expect(pane).not.toBeNull()
    fireEvent.click(pane as Element)

    fireEvent.click(screen.getByRole('button', { name: 'Share Edits' }))
    await waitFor(() => expect(writeText).toHaveBeenCalledOnce())
    const copied = new URL(writeText.mock.calls[0][0])
    expect(parseShareHash(copied.hash)?.modelHash).toBe('ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0')
  })
})
