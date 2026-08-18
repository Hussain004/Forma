import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import { isNpyBuffer, isNpzBuffer, parseNpy, parseNpz } from '../lib/npyParser'
import {
  cosineSimilarity,
  compareOutputs,
  maxAbsError,
  maxRelError,
  topKAgreement,
  type ValidationRunResult,
} from '../lib/validationUtils'
import type { OnnxGraph } from '../lib/onnxTypes'

// ---- Fixture builders -----------------------------------------------------

function buildNpy(values: number[], shape: number[]): ArrayBuffer {
  const shapeStr = shape.length === 1 ? `${shape[0]},` : shape.join(', ')
  const dict = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shapeStr}), }`
  const preLen = 10 // magic(6) + version(2) + headerLen(2)
  const unpadded = preLen + dict.length + 1
  const pad = (64 - (unpadded % 64)) % 64
  const header = dict + ' '.repeat(pad) + '\n'
  const headerBytes = new TextEncoder().encode(header)
  const data = new Float32Array(values)

  const buf = new ArrayBuffer(preLen + headerBytes.length + data.byteLength)
  const view = new DataView(buf)
  const bytes = new Uint8Array(buf)
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59], 0)
  bytes[6] = 1
  bytes[7] = 0
  view.setUint16(8, headerBytes.length, true)
  bytes.set(headerBytes, preLen)
  bytes.set(new Uint8Array(data.buffer), preLen + headerBytes.length)
  return buf
}

// Hand-encodes a minimal STORED-only (uncompressed) zip, matching what
// `numpy.savez` produces by default. CRC-32 is left as 0 -- parseNpz never
// checks it, it only needs the size/offset bookkeeping to be correct.
function buildStoredNpz(entries: Record<string, ArrayBuffer>): ArrayBuffer {
  const names = Object.entries(entries)
  const localParts: Uint8Array[] = []
  const localOffsets: number[] = []
  let offset = 0
  for (const [name, data] of names) {
    const filename = `${name}.npy`
    const filenameBytes = new TextEncoder().encode(filename)
    const dataBytes = new Uint8Array(data)
    const local = new Uint8Array(30 + filenameBytes.length + dataBytes.length)
    const view = new DataView(local.buffer)
    view.setUint32(0, 0x04034b50, true)
    view.setUint16(4, 20, true)
    view.setUint16(8, 0, true) // method: stored
    view.setUint32(18, dataBytes.length, true)
    view.setUint32(22, dataBytes.length, true)
    view.setUint16(26, filenameBytes.length, true)
    local.set(filenameBytes, 30)
    local.set(dataBytes, 30 + filenameBytes.length)
    localOffsets.push(offset)
    localParts.push(local)
    offset += local.length
  }

  const centralParts: Uint8Array[] = []
  let centralSize = 0
  names.forEach(([name, data], i) => {
    const filename = `${name}.npy`
    const filenameBytes = new TextEncoder().encode(filename)
    const dataBytes = new Uint8Array(data)
    const central = new Uint8Array(46 + filenameBytes.length)
    const view = new DataView(central.buffer)
    view.setUint32(0, 0x02014b50, true)
    view.setUint16(10, 0, true)
    view.setUint32(20, dataBytes.length, true)
    view.setUint32(24, dataBytes.length, true)
    view.setUint16(28, filenameBytes.length, true)
    view.setUint32(42, localOffsets[i], true)
    central.set(filenameBytes, 46)
    centralParts.push(central)
    centralSize += central.length
  })

  const centralOffset = offset
  const eocd = new Uint8Array(22)
  const eocdView = new DataView(eocd.buffer)
  eocdView.setUint32(0, 0x06054b50, true)
  eocdView.setUint16(8, names.length, true)
  eocdView.setUint16(10, names.length, true)
  eocdView.setUint32(12, centralSize, true)
  eocdView.setUint32(16, centralOffset, true)

  const total = new Uint8Array(offset + centralSize + eocd.length)
  let pos = 0
  for (const part of localParts) { total.set(part, pos); pos += part.length }
  for (const part of centralParts) { total.set(part, pos); pos += part.length }
  total.set(eocd, pos)
  return total.buffer
}

// ---- npyParser -------------------------------------------------------------

describe('v2.1 npyParser', () => {
  it('detects .npy and .npz magic bytes', () => {
    expect(isNpyBuffer(new Uint8Array(buildNpy([1], [1])))).toBe(true)
    expect(isNpyBuffer(new Uint8Array([0, 1, 2, 3]))).toBe(false)
    expect(isNpzBuffer(new Uint8Array(buildStoredNpz({ x: buildNpy([1], [1]) })))).toBe(true)
    expect(isNpzBuffer(new Uint8Array(buildNpy([1], [1])))).toBe(false)
  })

  it('parses a multi-dimensional float32 .npy array', () => {
    const values = [1, 2, 3, 4, 5, 6]
    const buf = buildNpy(values, [2, 3])
    const parsed = parseNpy(buf)
    expect(parsed.shape).toEqual([2, 3])
    expect(Array.from(parsed.data)).toEqual(values)
  })

  it('parses a 1-D array with a trailing-comma shape tuple', () => {
    const parsed = parseNpy(buildNpy([1, 2, 3], [3]))
    expect(parsed.shape).toEqual([3])
    expect(Array.from(parsed.data)).toEqual([1, 2, 3])
  })

  it('rejects a buffer with bad magic bytes', () => {
    expect(() => parseNpy(new ArrayBuffer(16))).toThrow(/bad magic/i)
  })

  it('parses every named array out of a STORED .npz', async () => {
    const npz = buildStoredNpz({
      x: buildNpy([1, 2, 3, 4], [2, 2]),
      y: buildNpy([9], [1]),
    })
    const result = await parseNpz(npz)
    expect(Object.keys(result).sort()).toEqual(['x', 'y'])
    expect(result.x.shape).toEqual([2, 2])
    expect(Array.from(result.x.data)).toEqual([1, 2, 3, 4])
    expect(Array.from(result.y.data)).toEqual([9])
  })

  it('rejects a buffer with no end-of-central-directory record', async () => {
    await expect(parseNpz(new ArrayBuffer(10))).rejects.toThrow(/end-of-central-directory/i)
  })
})

// ---- validationUtils --------------------------------------------------------

describe('v2.1 validationUtils', () => {
  it('computes max absolute and relative error', () => {
    const a = new Float32Array([1, 2, 4])
    const b = new Float32Array([1, 2.5, 3])
    expect(maxAbsError(a, b)).toBeCloseTo(1, 5)
    // max(|1-1|/1, |2-2.5|/2, |4-3|/4) = max(0, 0.25, 0.25) = 0.25
    expect(maxRelError(a, b)).toBeCloseTo(0.25, 3)
  })

  it('gives identical vectors a cosine similarity of 1 and orthogonal vectors 0', () => {
    expect(cosineSimilarity(new Float32Array([1, 2, 3]), new Float32Array([1, 2, 3]))).toBeCloseTo(1, 5)
    expect(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([0, 1]))).toBeCloseTo(0, 5)
  })

  it('is not applicable for scalar outputs', () => {
    expect(topKAgreement(new Float32Array([1]), new Float32Array([1]), 5)).toBeNull()
  })

  it('reports full overlap when the top-k ranking is unchanged', () => {
    const a = new Float32Array([0.1, 0.9, 0.5, 0.05])
    const b = new Float32Array([0.15, 0.8, 0.4, 0.02])
    expect(topKAgreement(a, b, 2)).toEqual({ k: 2, overlap: 2 })
  })

  it('reports partial overlap when the top-k ranking changes', () => {
    const a = new Float32Array([0.9, 0.1, 0.1, 0.1])
    const b = new Float32Array([0.1, 0.1, 0.1, 0.9])
    expect(topKAgreement(a, b, 1)).toEqual({ k: 1, overlap: 0 })
  })

  it('flags outputs missing from one side and shape mismatches', () => {
    const comparisons = compareOutputs(
      { a: { shape: [2], data: new Float32Array([1, 2]) }, b: { shape: [1], data: new Float32Array([1]) } },
      { a: { shape: [3], data: new Float32Array([1, 2, 3]) } },
    )
    const a = comparisons.find((c) => c.name === 'a')!
    const b = comparisons.find((c) => c.name === 'b')!
    expect(a.shapeMatch).toBe(false)
    expect(b.presentInModified).toBe(false)
  })

  it('computes full metrics when shapes match', () => {
    const [result] = compareOutputs(
      { y: { shape: [3], data: new Float32Array([1, 0, 0]) } },
      { y: { shape: [3], data: new Float32Array([1, 0, 0]) } },
    )
    expect(result.shapeMatch).toBe(true)
    expect(result.maxAbsErr).toBe(0)
    expect(result.cosineSim).toBeCloseTo(1, 5)
  })
})

// ---- App wiring --------------------------------------------------------------

const graph: OnnxGraph = {
  modelName: 'validate.onnx',
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
  graphInputs: [{ name: 'x', shape: [{ value: 1 }, { value: 4 }], elemType: 1 }],
}

const tfliteGraph: OnnxGraph = { ...graph, format: 'tflite' }

// 6 elements so the panel's default top-5 comparison is a genuine partial
// overlap (4/5) rather than trivially "all of them" at k >= length.
function makeResult(ok: boolean): ValidationRunResult {
  return {
    inputSource: 'generated',
    original: { loaded: true, inferenceOk: true, outputs: { y: { shape: [1, 6], data: new Float32Array([5, 4, 3, 2, 1, 0]) } } },
    modified: {
      loaded: ok,
      inferenceOk: ok,
      error: ok ? undefined : 'boom',
      outputs: ok ? { y: { shape: [1, 6], data: new Float32Array([0, 4, 3, 2, 1, 5]) } } : {},
    },
  }
}

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

function loadGraph(g: OnnxGraph) {
  act(() => {
    mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: g } } as MessageEvent)
  })
}

describe('v2.1 behavioral validation UI', () => {
  it('hides the Validate tab for read-only TFLite models', () => {
    render(<App />)
    loadGraph(tfliteGraph)
    expect(screen.queryByTestId('validation-tab')).not.toBeInTheDocument()
  })

  it('runs validation with generated inputs by default and renders the result', async () => {
    render(<App />)
    loadGraph(graph)

    fireEvent.click(screen.getByTestId('validation-tab'))
    expect(screen.getByText(/uses deterministic generated inputs/i)).toBeInTheDocument()

    fireEvent.click(screen.getByTestId('validate-run'))
    await waitFor(() => expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'VALIDATE', payload: expect.objectContaining({ providedInputs: null }) }),
    ))

    act(() => {
      mockWorker.onmessage?.({ data: { type: 'VALIDATION_RESULT', payload: makeResult(true) } } as MessageEvent)
    })

    const panel = await screen.findByTestId('validation-panel')
    expect(within(panel).getAllByText('Loaded / Inference OK')).toHaveLength(2)
    expect(within(panel).getByTestId('validation-output-y')).toBeInTheDocument()
    expect(within(panel).getByText('4/5')).toBeInTheDocument()
  })

  it('surfaces a failed modified-model run distinctly from a passing one', async () => {
    render(<App />)
    loadGraph(graph)
    fireEvent.click(screen.getByTestId('validation-tab'))
    fireEvent.click(screen.getByTestId('validate-run'))
    await waitFor(() => expect(mockWorker.postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'VALIDATE' })))

    act(() => {
      mockWorker.onmessage?.({ data: { type: 'VALIDATION_RESULT', payload: makeResult(false) } } as MessageEvent)
    })

    expect(await screen.findByText('boom')).toBeInTheDocument()
    expect(screen.getByText('Load failed / Inference failed')).toBeInTheDocument()
  })

  it('loads a single .npy onto the model\'s sole input and sends it as the validation feed', async () => {
    render(<App />)
    loadGraph(graph)
    fireEvent.click(screen.getByTestId('validation-tab'))

    const npy = buildNpy([1, 2, 3, 4], [1, 4])
    const file = new File([npy], 'x.npy')
    const input = screen.getByTestId('validate-file-input') as HTMLInputElement
    fireEvent.change(input, { target: { files: [file] } })

    expect(await screen.findByText(/Using loaded inputs: x/)).toBeInTheDocument()

    fireEvent.click(screen.getByTestId('validate-run'))
    await waitFor(() => expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'VALIDATE',
        payload: expect.objectContaining({
          providedInputs: { x: { data: new Float32Array([1, 2, 3, 4]), shape: [1, 4] } },
        }),
      }),
    ))
  })

  it('rejects a .npy upload when the model has more than one input', async () => {
    const twoInputGraph: OnnxGraph = {
      ...graph,
      graphInputs: [
        { name: 'x', shape: [{ value: 1 }], elemType: 1 },
        { name: 'z', shape: [{ value: 1 }], elemType: 1 },
      ],
    }
    render(<App />)
    loadGraph(twoInputGraph)
    fireEvent.click(screen.getByTestId('validation-tab'))

    const file = new File([buildNpy([1], [1])], 'x.npy')
    const input = screen.getByTestId('validate-file-input') as HTMLInputElement
    fireEvent.change(input, { target: { files: [file] } })

    expect(await screen.findByText(/exactly one input/i)).toBeInTheDocument()
  })

  it('records validation results per history state instead of per index', async () => {
    render(<App />)
    loadGraph(graph)
    fireEvent.click(screen.getByTestId('validation-tab'))

    // Run and resolve at the original state (index 0).
    fireEvent.click(screen.getByTestId('validate-run'))
    await waitFor(() => expect(mockWorker.postMessage).toHaveBeenCalledTimes(1))
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'VALIDATION_RESULT', payload: makeResult(true) } } as MessageEvent)
    })
    expect(await screen.findByTestId('validation-state-0')).toBeInTheDocument()

    // Add a node -> index 1, a state with no cached result yet.
    fireEvent.click(screen.getByText('Add Node'))
    fireEvent.mouseDown(screen.getByTestId('add-node-option-Tanh'))
    const pane = document.querySelector('.react-flow__pane')
    fireEvent.click(pane as Element)
    expect(screen.getByText('Not yet run for this edit state.')).toBeInTheDocument()

    fireEvent.click(screen.getByTestId('validate-run'))
    await waitFor(() => expect(mockWorker.postMessage).toHaveBeenCalledTimes(2))
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'VALIDATION_RESULT', payload: makeResult(false) } } as MessageEvent)
    })
    expect(await screen.findByText('boom')).toBeInTheDocument()
    expect(screen.getByTestId('validation-state-1')).toBeInTheDocument()

    // Undo back to index 0: the earlier cached result reappears without a third VALIDATE call.
    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    await waitFor(() => expect(screen.queryByText('boom')).not.toBeInTheDocument())
    expect(screen.queryByText('Load failed / Inference failed')).not.toBeInTheDocument()
    expect(mockWorker.postMessage).toHaveBeenCalledTimes(2)
  })
})
