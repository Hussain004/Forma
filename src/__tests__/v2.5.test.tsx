import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import type { OnnxGraph } from '../lib/onnxTypes'
import {
  compareModels,
  diffOpCounts,
  diffGraphIO,
  diffMetadata,
  attributeOnlyEditRecipe,
  formatComparisonReport,
} from '../lib/modelComparison'

// ---- Pure modelComparison.ts -------------------------------------------------

function baseNode(overrides: Partial<OnnxGraph['nodes'][number]>): OnnxGraph['nodes'][number] {
  return { id: 'x', opType: 'Noop', inputs: [], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0, ...overrides }
}

const baselineA: OnnxGraph = {
  modelName: 'base.onnx',
  totalParams: 27,
  totalSizeMB: 0.001,
  nodes: [
    baseNode({ id: 'input_0', opType: 'Input', outputs: ['x'] }),
    baseNode({
      id: 'node_0_Conv', name: 'conv1', opType: 'Conv', inputs: ['x', 'W'], outputs: ['y'],
      attributes: { strides: '[1, 1]' }, paramCount: 27, estimatedSizeMB: 0.001,
      inputMetadata: [{}, { shape: [{ value: 4 }, { value: 3 }, { value: 3 }, { value: 3 }], elemType: 1 }],
    }),
    baseNode({ id: 'node_1_Relu', name: 'relu1', opType: 'Relu', inputs: ['y'], outputs: ['z'] }),
    baseNode({ id: 'output_0', opType: 'Output', inputs: ['z'] }),
  ],
  edges: [],
  graphInputs: [{ name: 'x', shape: [{ value: 1 }, { value: 3 }, { value: 8 }, { value: 8 }], elemType: 1 }],
  metadata: { irVersion: 8, producerName: 'pytorch', producerVersion: '1.0', opsetVersion: 13, docString: '' },
}

// Same architecture, same node names: only Conv's `strides` attribute differs.
const candidateA: OnnxGraph = {
  ...baselineA,
  modelName: 'candidate.onnx',
  nodes: baselineA.nodes.map((n) => (n.name === 'conv1' ? { ...n, attributes: { strides: '[2, 2]' } } : n)),
}

describe('v2.5 modelComparison: attribute-only diff (matched by name)', () => {
  const comparison = compareModels(baselineA, candidateA)

  it('matches nodes by name and finds exactly the one attribute change', () => {
    expect(comparison.nodeMatch.matchedByName).toBe(true)
    expect(comparison.nodeMatch.addedNodes).toHaveLength(0)
    expect(comparison.nodeMatch.removedNodes).toHaveLength(0)
    expect(comparison.attributes).toEqual([
      { baselineNodeId: 'node_0_Conv', candidateNodeId: 'node_0_Conv', opType: 'Conv', attrName: 'strides', baselineValue: '[1, 1]', candidateValue: '[2, 2]' },
    ])
    expect(comparison.opCounts).toHaveLength(0)
    expect(comparison.initializers).toHaveLength(0)
  })

  it('offers an applicable edit recipe addressed to the baseline node', () => {
    const recipe = attributeOnlyEditRecipe(comparison)
    expect(recipe).toEqual([{ type: 'attr', nodeId: 'node_0_Conv', attrName: 'strides', value: '[2, 2]' }])
  })

  it('formats a plain-text report including the attribute change and both model names', () => {
    const report = formatComparisonReport(comparison, baselineA.modelName, candidateA.modelName)
    expect(report).toContain('Forma Model Comparison Report')
    expect(report).toContain('Baseline:  base.onnx')
    expect(report).toContain('Candidate: candidate.onnx')
    expect(report).toContain('Conv strides: [1, 1] -> [2, 2]')
  })

  it('includes latency and output sections in the report when supplied', () => {
    const report = formatComparisonReport(comparison, baselineA.modelName, candidateA.modelName, {
      latency: { baseline: { avgMs: 1.5, medianMs: 1.4 }, candidate: { avgMs: 1.6, medianMs: 1.5 } },
      outputs: [{ name: 'z', shapeMatch: true, maxAbsErr: 0.01, cosineSim: 0.999 }],
    })
    expect(report).toContain('1.50 / 1.40')
    expect(report).toContain('max abs err 0.010000')
  })
})

describe('v2.5 modelComparison: structural changes disqualify the edit recipe', () => {
  it('an added node blocks attributeOnlyEditRecipe even though attributes also changed', () => {
    const candidateWithExtra: OnnxGraph = {
      ...candidateA,
      nodes: [...candidateA.nodes, baseNode({ id: 'node_2_Sigmoid', name: 'sig1', opType: 'Sigmoid', inputs: ['z'], outputs: ['w'] })],
    }
    const comparison = compareModels(baselineA, candidateWithExtra)
    expect(comparison.nodeMatch.addedNodes.map((n) => n.opType)).toEqual(['Sigmoid'])
    expect(attributeOnlyEditRecipe(comparison)).toBeNull()
  })

  it('falls back to op-type-position matching when nodes are unnamed, but withholds the edit recipe', () => {
    const stripNames = (g: OnnxGraph): OnnxGraph => ({ ...g, nodes: g.nodes.map((n) => ({ ...n, name: undefined })) })
    const comparison = compareModels(stripNames(baselineA), stripNames(candidateA))
    expect(comparison.nodeMatch.matchedByName).toBe(false)
    expect(comparison.nodeMatch.matches).toHaveLength(2) // Conv+Relu match positionally; Input/Output pseudo-nodes are excluded from matching entirely
    expect(comparison.attributes.some((a) => a.attrName === 'strides')).toBe(true)
    // Structurally sound but unverified correspondence, so don't offer to auto-apply it.
    expect(attributeOnlyEditRecipe(comparison)).toBeNull()
  })
})

describe('v2.5 modelComparison: initializers, I/O, metadata, op counts', () => {
  const baselineB: OnnxGraph = {
    modelName: 'b.onnx', totalParams: 0, totalSizeMB: 0,
    nodes: [
      baseNode({ id: 'input_0', opType: 'Input', outputs: ['x'] }),
      baseNode({
        id: 'node_0_Conv', opType: 'Conv', inputs: ['x', 'W', 'b'], outputs: ['y'],
        inputMetadata: [{}, { shape: [{ value: 4 }, { value: 3 }, { value: 3 }, { value: 3 }], elemType: 1 }, { shape: [{ value: 4 }], elemType: 1 }],
      }),
      baseNode({ id: 'output_0', opType: 'Output', inputs: ['y'] }),
    ],
    edges: [],
    graphInputs: [{ name: 'x', shape: [{ value: 1 }, { value: 3 }, { value: 8 }, { value: 8 }], elemType: 1 }],
    metadata: { irVersion: 8, producerName: 'pytorch', producerVersion: '1.0', opsetVersion: 13, docString: '' },
  }

  const candidateB: OnnxGraph = {
    modelName: 'c.onnx', totalParams: 0, totalSizeMB: 0,
    nodes: [
      baseNode({ id: 'input_0', opType: 'Input', outputs: ['x'] }),
      baseNode({
        id: 'node_0_Conv', opType: 'Conv', inputs: ['x', 'W', 'scale'], outputs: ['y'],
        // W's shape changed (more output channels); 'b' removed, 'scale' added.
        inputMetadata: [{}, { shape: [{ value: 8 }, { value: 3 }, { value: 3 }, { value: 3 }], elemType: 1 }, { shape: [{ value: 8 }], elemType: 1 }],
      }),
      baseNode({ id: 'node_1_Relu', opType: 'Relu', inputs: ['y'], outputs: ['y2'] }),
      baseNode({ id: 'output_0', opType: 'Output', inputs: ['y2'] }),
      baseNode({ id: 'output_1', opType: 'Output', inputs: ['y'] }),
    ],
    edges: [],
    graphInputs: [{ name: 'x', shape: [{ value: 1 }, { value: 3 }, { value: 16 }, { value: 16 }], elemType: 1 }],
    metadata: { irVersion: 9, producerName: 'onnxruntime', producerVersion: '1.0', opsetVersion: 17, docString: '' },
  }

  it('diffInitializers reports added, removed, and changed by name', () => {
    const comparison = compareModels(baselineB, candidateB)
    const byName = Object.fromEntries(comparison.initializers.map((i) => [i.name, i.status]))
    expect(byName.W).toBe('changed')
    expect(byName.b).toBe('removed')
    expect(byName.scale).toBe('added')
  })

  it('diffGraphIO reports a changed input shape and an added output', () => {
    const io = diffGraphIO(baselineB, candidateB)
    const input = io.find((c) => c.ioKind === 'input' && c.name === 'x')!
    expect(input.status).toBe('changed')
    expect(input.baselineShape).toEqual([1, 3, 8, 8])
    expect(input.candidateShape).toEqual([1, 3, 16, 16])
    const addedOutput = io.find((c) => c.ioKind === 'output' && c.status === 'added')
    expect(addedOutput?.name).toBe('y2')
  })

  it('diffMetadata reports every differing field', () => {
    const changes = diffMetadata(baselineB, candidateB)
    const byField = Object.fromEntries(changes.map((c) => [c.field, [c.baselineValue, c.candidateValue]]))
    expect(byField['Opset version']).toEqual([13, 17])
    expect(byField.Producer).toEqual(['pytorch', 'onnxruntime'])
  })

  it('diffOpCounts only reports op types whose count actually differs', () => {
    const changes = diffOpCounts(baselineB, candidateB)
    expect(changes).toEqual([{ opType: 'Relu', baselineCount: 0, candidateCount: 1 }])
  })
})

// ---- App wiring: entering and leaving compare mode ---------------------------

const makeMockWorker = () => ({
  postMessage: vi.fn(),
  terminate: vi.fn(),
  onmessage: null as ((event: MessageEvent) => void) | null,
  onerror: null as ((event: unknown) => void) | null,
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
})

let workers: ReturnType<typeof makeMockWorker>[]

beforeEach(() => {
  workers = []
  vi.stubGlobal('Worker', vi.fn(function () {
    const w = makeMockWorker()
    workers.push(w)
    return w
  }))
  window.history.replaceState(null, '', '/')
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
  window.history.replaceState(null, '', '/')
})

describe('v2.5 App wiring: entering and leaving compare mode', () => {
  it('the landing page offers a way into compare mode, and back out again', async () => {
    render(<App />)
    fireEvent.click(await screen.findByTestId('compare-mode-entry'))
    expect(await screen.findByTestId('model-compare-page')).toBeInTheDocument()

    fireEvent.click(screen.getByTestId('compare-back'))
    expect(screen.queryByTestId('model-compare-page')).not.toBeInTheDocument()
  })
})

// ---- ModelComparePage: loading both models, diffing, latency/output runs -----

const compareBaseline: OnnxGraph = {
  modelName: 'baseline.onnx', totalParams: 0, totalSizeMB: 0,
  nodes: [
    baseNode({ id: 'input_0', opType: 'Input', outputs: ['x'] }),
    baseNode({ id: 'node_0_Relu', name: 'relu1', opType: 'Relu', inputs: ['x'], outputs: ['y'], attributes: { alpha: 1 } }),
    baseNode({ id: 'output_0', opType: 'Output', inputs: ['y'] }),
  ],
  edges: [],
  graphInputs: [{ name: 'x', shape: [{ value: 1 }, { value: 4 }], elemType: 1 }],
  metadata: { irVersion: 8, producerName: '', producerVersion: '', opsetVersion: 13, docString: '' },
}

const compareCandidate: OnnxGraph = {
  ...compareBaseline,
  modelName: 'candidate.onnx',
  nodes: compareBaseline.nodes.map((n) => (n.name === 'relu1' ? { ...n, attributes: { alpha: 2 } } : n)),
}

async function loadBothModels() {
  render(<App />)
  fireEvent.click(await screen.findByTestId('compare-mode-entry'))
  await screen.findByTestId('model-compare-page')

  const fileInputs = document.querySelectorAll('input[type="file"]')
  expect(fileInputs).toHaveLength(2)

  fireEvent.change(fileInputs[0], { target: { files: [new File([new TextEncoder().encode('abc')], 'baseline.onnx')] } })
  await waitFor(() => expect(workers[1].postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'LOAD_MODEL' }), expect.anything()))
  act(() => { workers[1].onmessage?.({ data: { type: 'MODEL_LOADED', payload: compareBaseline } } as MessageEvent) })

  fireEvent.change(fileInputs[1], { target: { files: [new File([new TextEncoder().encode('def')], 'candidate.onnx')] } })
  await waitFor(() => expect(workers[2].postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'LOAD_MODEL' }), expect.anything()))
  act(() => { workers[2].onmessage?.({ data: { type: 'MODEL_LOADED', payload: compareCandidate } } as MessageEvent) })
}

describe('v2.5 ModelComparePage', () => {
  it('loads both models and renders the attribute diff', async () => {
    await loadBothModels()
    expect(await screen.findByTestId('compare-attribute-change')).toHaveTextContent('Relu alpha: 1 -> 2')
    expect(screen.getByTestId('compare-export-recipe')).toBeInTheDocument()
  })

  it('runs a latency comparison against both workers independently', async () => {
    await loadBothModels()
    fireEvent.click(screen.getByTestId('compare-run-latency'))

    await waitFor(() => expect(workers[1].postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'BENCHMARK' })))
    await waitFor(() => expect(workers[2].postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'BENCHMARK' })))
    act(() => { workers[1].onmessage?.({ data: { type: 'BENCHMARK_RESULT', payload: { avgMs: 1, medianMs: 1, minMs: 1, maxMs: 1, runs: 10 } } } as MessageEvent) })
    act(() => { workers[2].onmessage?.({ data: { type: 'BENCHMARK_RESULT', payload: { avgMs: 2, medianMs: 2, minMs: 2, maxMs: 2, runs: 10 } } } as MessageEvent) })

    expect(await screen.findByText('1.00 / 1.00')).toBeInTheDocument()
    expect(screen.getByText('2.00 / 2.00')).toBeInTheDocument()
  })

  it('runs an output comparison via the single-sided RUN_GENERATED command on each worker', async () => {
    await loadBothModels()
    fireEvent.click(screen.getByTestId('compare-run-outputs'))

    await waitFor(() => expect(workers[1].postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'RUN_GENERATED' })))
    await waitFor(() => expect(workers[2].postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'RUN_GENERATED' })))

    const outA = { loaded: true, inferenceOk: true, outputs: { y: { shape: [1, 4], data: new Float32Array([1, 2, 3, 4]) } } }
    const outB = { loaded: true, inferenceOk: true, outputs: { y: { shape: [1, 4], data: new Float32Array([1, 2, 3, 4]) } } }
    act(() => { workers[1].onmessage?.({ data: { type: 'GENERATED_RESULT', payload: outA } } as MessageEvent) })
    act(() => { workers[2].onmessage?.({ data: { type: 'GENERATED_RESULT', payload: outB } } as MessageEvent) })

    expect(await screen.findByTestId('compare-output-y')).toHaveTextContent('Max abs error0')
  })

  it('exports an edit recipe as a share link scoped to the baseline model hash', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { ...navigator, clipboard: { writeText } })
    await loadBothModels()

    fireEvent.click(await screen.findByTestId('compare-export-recipe'))
    await waitFor(() => expect(writeText).toHaveBeenCalledOnce())
    const copied = new URL(writeText.mock.calls[0][0])
    expect(copied.hash).toContain('#s=')
    expect(await screen.findByText(/Edit recipe link copied \(1 attribute change\)/)).toBeInTheDocument()
  })

  it('withholds the edit recipe button once a structural difference exists', async () => {
    render(<App />)
    fireEvent.click(await screen.findByTestId('compare-mode-entry'))
    await screen.findByTestId('model-compare-page')
    const fileInputs = document.querySelectorAll('input[type="file"]')

    fireEvent.change(fileInputs[0], { target: { files: [new File([new TextEncoder().encode('abc')], 'baseline.onnx')] } })
    await waitFor(() => expect(workers[1].postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'LOAD_MODEL' }), expect.anything()))
    act(() => { workers[1].onmessage?.({ data: { type: 'MODEL_LOADED', payload: compareBaseline } } as MessageEvent) })

    const candidateWithExtraNode: OnnxGraph = {
      ...compareCandidate,
      nodes: [...compareCandidate.nodes, baseNode({ id: 'node_1_Sigmoid', name: 'sig1', opType: 'Sigmoid', inputs: ['y'], outputs: ['w'] })],
    }
    fireEvent.change(fileInputs[1], { target: { files: [new File([new TextEncoder().encode('def')], 'candidate.onnx')] } })
    await waitFor(() => expect(workers[2].postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'LOAD_MODEL' }), expect.anything()))
    act(() => { workers[2].onmessage?.({ data: { type: 'MODEL_LOADED', payload: candidateWithExtraNode } } as MessageEvent) })

    await screen.findByTestId('compare-attribute-change')
    expect(screen.queryByTestId('compare-export-recipe')).not.toBeInTheDocument()
  })

  it('blocks comparison and explains why when one side is a TFLite model', async () => {
    render(<App />)
    fireEvent.click(await screen.findByTestId('compare-mode-entry'))
    await screen.findByTestId('model-compare-page')
    const fileInputs = document.querySelectorAll('input[type="file"]')

    fireEvent.change(fileInputs[0], { target: { files: [new File([new TextEncoder().encode('abc')], 'baseline.onnx')] } })
    await waitFor(() => expect(workers[1].postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'LOAD_MODEL' }), expect.anything()))
    act(() => { workers[1].onmessage?.({ data: { type: 'MODEL_LOADED', payload: compareBaseline } } as MessageEvent) })

    fireEvent.change(fileInputs[1], { target: { files: [new File([new TextEncoder().encode('def')], 'candidate.tflite')] } })
    await waitFor(() => expect(workers[2].postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'LOAD_MODEL' }), expect.anything()))
    act(() => { workers[2].onmessage?.({ data: { type: 'MODEL_LOADED', payload: { ...compareCandidate, format: 'tflite' as const } } } as MessageEvent) })

    expect(await screen.findByText(/only supports ONNX models/)).toBeInTheDocument()
    expect(screen.queryByTestId('compare-run-latency')).not.toBeInTheDocument()
  })
})
