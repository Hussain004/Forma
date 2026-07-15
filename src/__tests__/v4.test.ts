import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, render, screen, act, fireEvent, waitFor } from '@testing-library/react'
import { createElement } from 'react'
import { useOnnxWorker } from '../hooks/useOnnxWorker'
import App from '../App'
import type { OnnxGraph } from '../lib/onnxTypes'

import { formatQuantizeEstimate } from '../lib/quantize'

// ---- Shared mock worker (mirrors onnx.test.ts) ----

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

// ---- Group 1: exportModel rejects on error ----

describe('useOnnxWorker exportModel rejection (v4)', () => {
  let mockWorker: ReturnType<typeof makeMockWorker>

  beforeEach(() => {
    mockWorker = makeMockWorker()
    vi.stubGlobal('Worker', vi.fn(function () { return mockWorker }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('rejects the exportModel promise with the worker error message on ERROR', async () => {
    const { result } = renderHook(() => useOnnxWorker())
    let exportPromise!: Promise<ArrayBuffer>
    act(() => {
      exportPromise = result.current.exportModel()
    })
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'ERROR', payload: 'no model loaded' } } as MessageEvent)
    })
    await expect(exportPromise).rejects.toThrow('no model loaded')
  }, 2000)

  it('rejects the exportModel promise with an Error instance on ERROR', async () => {
    const { result } = renderHook(() => useOnnxWorker())
    let exportPromise!: Promise<ArrayBuffer>
    act(() => {
      exportPromise = result.current.exportModel()
    })
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'ERROR', payload: 'export failed' } } as MessageEvent)
    })
    await expect(exportPromise).rejects.toBeInstanceOf(Error)
  }, 2000)

  it('does not leave the export promise unsettled after an ERROR', async () => {
    const { result } = renderHook(() => useOnnxWorker())
    let exportPromise!: Promise<ArrayBuffer>
    act(() => {
      exportPromise = result.current.exportModel()
    })
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'ERROR', payload: 'boom' } } as MessageEvent)
    })
    const outcome = await Promise.race([
      exportPromise.then(() => 'resolved', () => 'rejected'),
      new Promise<string>((resolve) => setTimeout(() => resolve('pending'), 50)),
    ])
    expect(outcome).toBe('rejected')
  })
})

// ---- Group 2: formatQuantizeEstimate display helper ----

describe('formatQuantizeEstimate (v4)', () => {
  it('returns an empty string for null', () => {
    expect(formatQuantizeEstimate(null)).toBe('')
  })

  it('formats a typical estimate as INT8 size and ratio', () => {
    expect(formatQuantizeEstimate({ int8SizeMB: 1.5, ratio: 2.67 })).toBe('INT8: 1.5 MB (2.7x)')
  })

  it('rounds the ratio to one decimal place', () => {
    expect(formatQuantizeEstimate({ int8SizeMB: 2, ratio: 4.567 })).toBe('INT8: 2.0 MB (4.6x)')
  })

  it('shows int8SizeMB to one decimal place for whole numbers', () => {
    expect(formatQuantizeEstimate({ int8SizeMB: 3, ratio: 2 })).toBe('INT8: 3.0 MB (2.0x)')
  })

  it('returns an empty string for a zero estimate', () => {
    expect(formatQuantizeEstimate({ int8SizeMB: 0, ratio: 0 })).toBe('')
  })
})

// ---- Group 3: Download button triggers a browser download ----

describe('App download button (v4)', () => {
  let mockWorker: ReturnType<typeof makeMockWorker>
  let createObjectURL: ReturnType<typeof vi.fn>
  let revokeObjectURL: ReturnType<typeof vi.fn>
  let origCreateObjectURL: PropertyDescriptor | undefined
  let origRevokeObjectURL: PropertyDescriptor | undefined

  beforeEach(() => {
    mockWorker = makeMockWorker()
    vi.stubGlobal('Worker', vi.fn(function () { return mockWorker }))

    createObjectURL = vi.fn(() => 'blob:mock-url')
    revokeObjectURL = vi.fn()
    origCreateObjectURL = Object.getOwnPropertyDescriptor(URL, 'createObjectURL')
    origRevokeObjectURL = Object.getOwnPropertyDescriptor(URL, 'revokeObjectURL')
    Object.defineProperty(URL, 'createObjectURL', { value: createObjectURL, configurable: true, writable: true })
    Object.defineProperty(URL, 'revokeObjectURL', { value: revokeObjectURL, configurable: true, writable: true })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    if (origCreateObjectURL) Object.defineProperty(URL, 'createObjectURL', origCreateObjectURL)
    else Reflect.deleteProperty(URL, 'createObjectURL')
    if (origRevokeObjectURL) Object.defineProperty(URL, 'revokeObjectURL', origRevokeObjectURL)
    else Reflect.deleteProperty(URL, 'revokeObjectURL')
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  const loadModel = () => {
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: testGraph } } as MessageEvent)
    })
  }

  it('renders a Download button when a model is loaded', () => {
    render(createElement(App))
    loadModel()
    expect(screen.getByRole('button', { name: /download/i })).toBeInTheDocument()
  })

  it('posts an EXPORT command to the worker when Download is clicked', () => {
    render(createElement(App))
    loadModel()
    fireEvent.click(screen.getByRole('button', { name: /download/i }))
    expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'EXPORT' }),
    )
  })

  it('triggers a browser download via a Blob after EXPORT_RESULT', async () => {
    render(createElement(App))
    loadModel()
    fireEvent.click(screen.getByRole('button', { name: /download/i }))
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'EXPORT_RESULT', payload: new ArrayBuffer(16) } } as MessageEvent)
    })
    await waitFor(() => expect(createObjectURL).toHaveBeenCalledWith(expect.any(Blob)))
  })

  it('names the downloaded file after the model', async () => {
    render(createElement(App))
    loadModel()

    const realCreateElement = document.createElement.bind(document)
    const anchor = realCreateElement('a') as HTMLAnchorElement
    const clickSpy = vi.spyOn(anchor, 'click').mockImplementation(() => {})
    vi.spyOn(document, 'createElement').mockImplementation(
      ((tagName: string, options?: ElementCreationOptions) =>
        tagName === 'a' ? anchor : realCreateElement(tagName, options)) as typeof document.createElement,
    )

    fireEvent.click(screen.getByRole('button', { name: /download/i }))
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'EXPORT_RESULT', payload: new ArrayBuffer(16) } } as MessageEvent)
    })

    await waitFor(() => expect(clickSpy).toHaveBeenCalled())
    expect(anchor.download).toMatch(/^test_original\.onnx$/)
  })
})
