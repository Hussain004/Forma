import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createElement } from 'react'
import App from '../App'
import { buildGraphFromParsed } from '../lib/onnxParser'
import {
  rewireEdge,
  toSelectableGraph,
  validateRewire,
  validateTensorCompatibility,
} from '../lib/graphUtils'
import type { ParsedGraph } from '../lib/onnxProtoParser'
import type { OnnxGraph, TensorMetadata } from '../lib/onnxTypes'

const graphCanvasState = vi.hoisted(() => ({
  props: null as null | {
    onRewire?: (sourceNodeId: string, targetNodeId: string, inputPosition: number) => void
  },
}))

vi.mock('../components/GraphCanvas', () => ({
  GraphCanvas: (props: typeof graphCanvasState.props) => {
    graphCanvasState.props = props
    return null
  },
}))

const shape = (...values: number[]): TensorMetadata['shape'] => values.map((value) => ({ value }))

function makeCompatibilityGraph(
  sourceMetadata?: TensorMetadata,
  targetMetadata?: TensorMetadata,
): OnnxGraph {
  return {
    modelName: 'compatibility.onnx',
    totalParams: 0,
    totalSizeMB: 0,
    nodes: [
      {
        id: 'node_0_Identity',
        opType: 'Identity',
        inputs: ['x'],
        outputs: ['source_out'],
        attributes: {},
        paramCount: 0,
        estimatedSizeMB: 0,
        outputMetadata: sourceMetadata ? [sourceMetadata] : undefined,
      },
      {
        id: 'node_1_Relu',
        opType: 'Relu',
        inputs: ['x'],
        outputs: ['target_out'],
        attributes: {},
        paramCount: 0,
        estimatedSizeMB: 0,
        inputMetadata: targetMetadata ? [targetMetadata] : undefined,
      },
    ],
    edges: [],
  }
}

describe('v1.8 aligned tensor metadata', () => {
  it('preserves input positions including unknown and initializer metadata', () => {
    const proto: ParsedGraph = {
      name: 'metadata',
      nodes: [{
        inputs: ['missing', 'x', 'weight'],
        outputs: ['y'],
        name: 'add',
        opType: 'Add',
        attributes: {},
      }],
      initializers: [{
        name: 'weight',
        dims: [4, 4],
        elemType: 1,
        elemCount: 16,
        sizeMB: 0.000061,
      }],
      inputs: [{ name: 'x', shape: [{ param: 'batch' }, { value: 4 }], elemType: 1 }],
      outputs: [{ name: 'y', shape: [{ param: 'batch' }, { value: 4 }], elemType: 1 }],
      valueInfo: [],
    }

    const graph = buildGraphFromParsed(proto, 'metadata.onnx', 'onnx')
    const node = graph.nodes.find((item) => item.id === 'node_0_Add')

    expect(node?.inputMetadata).toHaveLength(3)
    expect(node?.inputMetadata?.[0]).toEqual({})
    expect(node?.inputMetadata?.[1]).toEqual({
      shape: [{ param: 'batch' }, { value: 4 }],
      elemType: 1,
    })
    expect(node?.inputMetadata?.[2]).toEqual({
      shape: [{ value: 4 }, { value: 4 }],
      elemType: 1,
    })
    expect(node?.outputMetadata?.[0]?.elemType).toBe(1)
  })
})

describe('v1.8 tensor compatibility', () => {
  it('rejects a known dtype mismatch', () => {
    const result = validateTensorCompatibility(
      { shape: shape(1, 4), elemType: 1 },
      { shape: shape(1, 4), elemType: 7 },
    )

    expect(result).toEqual({
      valid: false,
      reason: 'Tensor type mismatch: source FLOAT, target expects INT64',
    })
  })

  it('rejects a known rank mismatch', () => {
    const result = validateTensorCompatibility(
      { shape: shape(1, 4), elemType: 1 },
      { shape: shape(1, 4, 1), elemType: 1 },
    )

    expect(result.valid).toBe(false)
    expect(result.reason).toMatch(/rank 2.*rank 3/i)
  })

  it('rejects a known concrete dimension mismatch', () => {
    const result = validateTensorCompatibility(
      { shape: shape(1, 8), elemType: 1 },
      { shape: shape(1, 4), elemType: 1 },
    )

    expect(result.valid).toBe(false)
    expect(result.reason).toMatch(/dimension 2.*source 8.*expects 4/i)
  })

  it('allows symbolic dimensions and unknown metadata', () => {
    expect(validateTensorCompatibility(
      { shape: [{ param: 'batch' }, { value: 4 }], elemType: 1 },
      { shape: [{ value: 8 }, { value: 4 }], elemType: 1 },
    )).toEqual({ valid: true })
    expect(validateTensorCompatibility(undefined, { shape: shape(1, 4), elemType: 1 })).toEqual({ valid: true })
  })

  it('accepts an exact known contract', () => {
    expect(validateTensorCompatibility(
      { shape: shape(1, 4), elemType: 1 },
      { shape: shape(1, 4), elemType: 1 },
    )).toEqual({ valid: true })
  })
})

describe('v1.8 rewire validation', () => {
  it('checks the source output and target input slots', () => {
    const graph = makeCompatibilityGraph(
      { shape: shape(1, 4), elemType: 1 },
      { shape: shape(1, 4), elemType: 7 },
    )

    const result = validateRewire(graph, 'node_0_Identity', 'node_1_Relu', 0)
    expect(result.valid).toBe(false)
    expect(result.reason).toMatch(/FLOAT.*INT64/)
  })

  it('keeps rewiring permissive when either slot lacks metadata', () => {
    const graph = makeCompatibilityGraph(undefined, { shape: shape(1, 4), elemType: 1 })
    expect(validateRewire(graph, 'node_0_Identity', 'node_1_Relu', 0)).toEqual({ valid: true })
  })

  it('carries source shape metadata onto the new edge', () => {
    const graph = toSelectableGraph(makeCompatibilityGraph(
      { shape: shape(1, 4), elemType: 1 },
      { shape: shape(1, 4), elemType: 1 },
    ))

    const result = rewireEdge(graph, 'node_0_Identity', 'node_1_Relu', 0)
    expect(result.edges[0]?.shape).toEqual(shape(1, 4))
  })
})

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
  graphCanvasState.props = null
  vi.stubGlobal('Worker', vi.fn(function () { return mockWorker }))
  localStorage.clear()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

describe('v1.8 application feedback', () => {
  it('announces a rejected incompatible rewire without adding history', async () => {
    const graph = makeCompatibilityGraph(
      { shape: shape(1, 4), elemType: 1 },
      { shape: shape(1, 4), elemType: 7 },
    )

    render(createElement(App))
    act(() => {
      mockWorker.onmessage?.({ data: { type: 'MODEL_LOADED', payload: graph } } as MessageEvent)
    })
    await act(() => new Promise((resolve) => setTimeout(resolve, 0)))

    act(() => {
      graphCanvasState.props?.onRewire?.('node_0_Identity', 'node_1_Relu', 0)
    })

    expect(screen.getByTestId('announcement')).toHaveTextContent(
      'Rewire rejected: Tensor type mismatch: source FLOAT, target expects INT64',
    )
    expect(screen.getByTestId('diff-toggle')).toBeDisabled()
  })
})
