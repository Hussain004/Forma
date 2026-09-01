import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import { parseOnnxProto } from '../lib/onnxProtoParser'
import { writeModifiedOnnx, type StructuralOp } from '../lib/onnxProtoWriter'
import {
  currentInputBoundaryTensor,
  describeHistoryEntry,
  friendlyNodeLabel,
  insertRecipeNode,
  structuralNodeIndex,
  toSelectableGraph,
  type SelectableGraph,
} from '../lib/graphUtils'
import { createShareHash } from '../lib/shareLinks'
import { PIPELINE_RECIPES, resolveRecipe } from '../lib/pipelineRecipes'
import type { OnnxGraph } from '../lib/onnxTypes'

// ---- Minimal ONNX protobuf builder (for test fixtures only) -----------------
// Independent of onnxProtoWriter.ts, mirroring v2.2/v2.3's fixture builders.

function encodeVarint(value: number): number[] {
  const bytes: number[] = []
  let v = value >>> 0
  do {
    let b = v & 0x7f
    v >>>= 7
    if (v !== 0) b |= 0x80
    bytes.push(b)
  } while (v !== 0)
  return bytes
}
function lenField(field: number, data: number[]): number[] {
  const tag = (field << 3) | 2
  return [...encodeVarint(tag), ...encodeVarint(data.length), ...data]
}
function varintField(field: number, value: number): number[] {
  const tag = (field << 3) | 0
  return [...encodeVarint(tag), ...encodeVarint(value)]
}
function strField(field: number, value: string): number[] {
  return lenField(field, Array.from(new TextEncoder().encode(value)))
}
function float32Bytes(value: number): number[] {
  const buf = new ArrayBuffer(4)
  new DataView(buf).setFloat32(0, value, true)
  return Array.from(new Uint8Array(buf))
}
function makeNode(opType: string, inputs: string[], outputs: string[], name: string): number[] {
  const bytes: number[] = []
  for (const s of inputs) bytes.push(...strField(1, s))
  for (const s of outputs) bytes.push(...strField(2, s))
  bytes.push(...strField(3, name))
  bytes.push(...strField(4, opType))
  return bytes
}
function makeInitializer(name: string, dims: number[], values: number[]): number[] {
  const dimBytes = dims.flatMap((d) => encodeVarint(d))
  const raw = values.flatMap(float32Bytes)
  return [...lenField(1, dimBytes), ...varintField(2, 1), ...lenField(9, raw), ...strField(8, name)]
}
function makeValueInfo(name: string, elemType?: number, dims?: number[]): number[] {
  const bytes = [...strField(1, name)]
  if (elemType !== undefined) {
    const dimBytes = (dims ?? []).flatMap((d) => lenField(1, varintField(1, d)))
    const tensorType = [...varintField(1, elemType), ...(dims ? lenField(2, dimBytes) : [])]
    bytes.push(...lenField(2, lenField(1, tensorType)))
  }
  return bytes
}
function makeGraph(nodes: number[][], inputs: number[][], outputs: number[][], initializers: number[][]): number[] {
  const bytes: number[] = []
  for (const n of nodes) bytes.push(...lenField(1, n))
  for (const i of initializers) bytes.push(...lenField(5, i))
  for (const s of inputs) bytes.push(...lenField(11, s))
  for (const s of outputs) bytes.push(...lenField(12, s))
  return bytes
}
function makeModel(graph: number[]): ArrayBuffer {
  return new Uint8Array([...varintField(1, 8), ...lenField(7, graph)]).buffer
}

// x -[Conv,W,b]-> y -[Relu]-> z
function makeFixture(): ArrayBuffer {
  const conv = makeNode('Conv', ['x', 'W', 'b'], ['y'], 'conv0')
  const relu = makeNode('Relu', ['y'], ['z'], 'relu0')
  const graph = makeGraph(
    [conv, relu],
    [makeValueInfo('x', 1, [1, 3, 8, 8])],
    [makeValueInfo('z', 1, [1, 3, 8, 8])],
    [makeInitializer('W', [4, 3, 3, 3], new Array(108).fill(0.1)), makeInitializer('b', [4], [1, 2, 3, 4])],
  )
  return makeModel(graph)
}

// x -[Relu]-> a -\
//   \-[Sigmoid]-> b -[Add]-> z    (x fans out to two direct consumers)
function makeFanoutFixture(): ArrayBuffer {
  const reluA = makeNode('Relu', ['x'], ['a'], 'reluA')
  const sigB = makeNode('Sigmoid', ['x'], ['b'], 'sigB')
  const add = makeNode('Add', ['a', 'b'], ['z'], 'add0')
  const graph = makeGraph(
    [reluA, sigB, add],
    [makeValueInfo('x', 1, [1, 4])],
    [makeValueInfo('z', 1, [1, 4])],
    [],
  )
  return makeModel(graph)
}

function applyOps(buffer: ArrayBuffer, ops: StructuralOp[], overrides = new Map<number, Record<string, string | number>>()): ReturnType<typeof parseOnnxProto> {
  return parseOnnxProto(writeModifiedOnnx(buffer, overrides, ops))
}

const castOnX: StructuralOp = {
  type: 'insertRecipe', anchorKind: 'input', ioIndex: 0, anchorTensor: 'x', newNodeIndex: 1,
  opType: 'Cast', attrs: [{ name: 'to', kind: 'I', value: 1 }], extraInputs: [], extraOutputCount: 0, extraOutputElemTypes: [],
}
const softmaxOnZ: StructuralOp = {
  type: 'insertRecipe', anchorKind: 'output', ioIndex: 0, anchorTensor: '', newNodeIndex: 2,
  opType: 'Softmax', attrs: [{ name: 'axis', kind: 'I', value: -1 }], extraInputs: [], extraOutputCount: 0, extraOutputElemTypes: [],
}

// ---- Writer-level: insertRecipe against real wire bytes ---------------------

describe('v2.4 writer: insertRecipe', () => {
  it('preprocess: prepends the node and rewires the sole consumer to its output', () => {
    const parsed = applyOps(makeFixture(), [castOnX])
    expect(parsed.nodes[0].opType).toBe('Cast')
    expect(parsed.nodes[0].inputs).toEqual(['x'])
    expect(parsed.nodes[0].outputs).toEqual(['recipe_1_out'])
    expect(parsed.nodes[0].attributes.to).toBe(1)
    expect(parsed.nodes[1].opType).toBe('Conv')
    expect(parsed.nodes[1].inputs[0]).toBe('recipe_1_out') // was 'x'
    expect(parsed.inputs[0].name).toBe('x') // public contract unchanged
  })

  it('preprocess: rewires every fan-out consumer of the graph input, not just one', () => {
    const parsed = applyOps(makeFanoutFixture(), [castOnX])
    const relu = parsed.nodes.find((n) => n.opType === 'Relu')!
    const sig = parsed.nodes.find((n) => n.opType === 'Sigmoid')!
    expect(relu.inputs).toEqual(['recipe_1_out'])
    expect(sig.inputs).toEqual(['recipe_1_out'])
  })

  it('postprocess: appends the node and preserves the graph output\'s public name', () => {
    const parsed = applyOps(makeFixture(), [softmaxOnZ])
    const last = parsed.nodes[parsed.nodes.length - 1]
    expect(last.opType).toBe('Softmax')
    expect(last.outputs).toEqual(['z'])
    expect(last.inputs).toEqual(['recipe_2_in'])
    const relu = parsed.nodes.find((n) => n.opType === 'Relu')!
    expect(relu.outputs).toEqual(['recipe_2_in']) // was 'z'
    expect(parsed.outputs[0].name).toBe('z') // public contract unchanged
  })

  it('chains a second preprocess recipe after the first, not both racing the raw input', () => {
    const second: StructuralOp = { ...castOnX, newNodeIndex: 3, anchorTensor: 'recipe_1_out', opType: 'LpNormalization', attrs: [{ name: 'axis', kind: 'I', value: 1 }] }
    const parsed = applyOps(makeFixture(), [castOnX, second])
    expect(parsed.nodes.map((n) => n.opType)).toEqual(['Cast', 'LpNormalization', 'Conv', 'Relu'])
    expect(parsed.nodes[0].outputs).toEqual(['recipe_1_out'])
    expect(parsed.nodes[1].inputs).toEqual(['recipe_1_out'])
    expect(parsed.nodes[1].outputs).toEqual(['recipe_3_out'])
    expect(parsed.nodes[2].inputs[0]).toBe('recipe_3_out') // Conv now reads the chain's tail
  })

  it('chains a second postprocess recipe before the first, keeping the public name last', () => {
    const second: StructuralOp = { ...softmaxOnZ, newNodeIndex: 3, opType: 'Cast', attrs: [{ name: 'to', kind: 'I', value: 6 }] }
    const parsed = applyOps(makeFixture(), [softmaxOnZ, second])
    expect(parsed.nodes.map((n) => n.opType)).toEqual(['Conv', 'Relu', 'Softmax', 'Cast'])
    expect(parsed.nodes[2].outputs).toEqual(['recipe_3_in']) // Softmax's public output got renamed for the chain
    expect(parsed.nodes[3].inputs).toEqual(['recipe_3_in'])
    expect(parsed.nodes[3].outputs).toEqual(['z'])
    expect(parsed.outputs[0].name).toBe('z')
  })

  it('an "empty" extra input slot encodes ONNX\'s own omitted-optional-input convention', () => {
    const resize: StructuralOp = {
      type: 'insertRecipe', anchorKind: 'input', ioIndex: 0, anchorTensor: 'x', newNodeIndex: 1,
      opType: 'Resize', attrs: [{ name: 'mode', kind: 'S', value: 'nearest' }],
      extraInputs: [{ kind: 'empty' }, { kind: 'const', elemType: 1, dims: [4], values: [1, 1, 2, 2] }],
      extraOutputCount: 0,
    }
    const parsed = applyOps(makeFixture(), [resize])
    expect(parsed.nodes[0].inputs).toEqual(['x', '', 'recipe_1_const1'])
    const scales = parsed.initializers.find((i) => i.name === 'recipe_1_const1')!
    expect(scales.values).toEqual([1, 1, 2, 2])
  })

  it('a "const" extra output is a real new initializer with the requested dtype and values', () => {
    const topK: StructuralOp = {
      type: 'insertRecipe', anchorKind: 'output', ioIndex: 0, anchorTensor: '', newNodeIndex: 2,
      opType: 'TopK', attrs: [{ name: 'axis', kind: 'I', value: -1 }, { name: 'largest', kind: 'I', value: 1 }],
      extraInputs: [{ kind: 'const', elemType: 7, dims: [1], values: [5] }],
      extraOutputCount: 1,
      extraOutputElemTypes: [7],
    }
    const parsed = applyOps(makeFixture(), [topK])
    const node = parsed.nodes[parsed.nodes.length - 1]
    expect(node.opType).toBe('TopK')
    expect(node.inputs).toEqual(['recipe_2_in', 'recipe_2_const0'])
    expect(node.outputs).toEqual(['z', 'recipe_2_extra0'])
    const k = parsed.initializers.find((i) => i.name === 'recipe_2_const0')!
    expect(k.dims).toEqual([1])
    expect(k.elemType).toBe(7)
    expect(k.values).toEqual([5])
    // extraOutputCount auto-promotes the second output as an additional graph output.
    expect(parsed.outputs.map((o) => o.name)).toEqual(['z', 'recipe_2_extra0'])
    // The primary output's stale pre-recipe shape is dropped (unranked), since
    // Top-K genuinely changes it -- but its dtype is preserved, and the new
    // Indices output is declared int64 (never inherited from the primary).
    expect(parsed.outputs[0].shape).toBeUndefined()
    expect(parsed.outputs[0].elemType).toBe(1)
    expect(parsed.outputs[1].elemType).toBe(7)
  })

  it('a recipe node is addressable by later ops -- attribute overrides apply to it at export time', () => {
    const overrides = new Map([[-1, { to: 6 }]])
    const parsed = applyOps(makeFixture(), [castOnX], overrides)
    expect(parsed.nodes[0].attributes.to).toBe(6)
  })

  it('legacy Top-K (opset < 10): single input, k as an attribute, no K initializer', () => {
    const legacyTopK: StructuralOp = {
      type: 'insertRecipe', anchorKind: 'output', ioIndex: 0, anchorTensor: '', newNodeIndex: 2,
      opType: 'TopK', attrs: [{ name: 'axis', kind: 'I', value: -1 }, { name: 'k', kind: 'I', value: 5 }],
      extraInputs: [], extraOutputCount: 1, extraOutputElemTypes: [7],
    }
    const parsed = applyOps(makeFixture(), [legacyTopK])
    const node = parsed.nodes[parsed.nodes.length - 1]
    expect(node.inputs).toEqual(['recipe_2_in'])
    expect(node.attributes.k).toBe(5)
    expect(parsed.initializers.some((i) => i.name.includes('recipe_2_const'))).toBe(false)
  })

  it('a recipe node can be renamed like any other addressable structural entry', () => {
    const rename: StructuralOp = { type: 'renameNode', nodeIndex: -1, name: 'my_cast' }
    const parsed = applyOps(makeFixture(), [castOnX, rename])
    expect(parsed.nodes[0].name).toBe('my_cast')
  })
})

// ---- graphUtils pure transforms and helpers ---------------------------------

const recipeGraph: SelectableGraph = toSelectableGraph({
  modelName: 'r.onnx',
  totalParams: 0,
  totalSizeMB: 0,
  nodes: [
    { id: 'input_0', opType: 'Input', inputs: [], outputs: ['x'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'node_0_Relu', opType: 'Relu', inputs: ['x'], outputs: ['y'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'output_0', opType: 'Output', inputs: ['y'], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
  ],
  edges: [
    { id: 'input_0->node_0_Relu@x', source: 'input_0', target: 'node_0_Relu', label: 'x' },
    { id: 'node_0_Relu->output_0@y', source: 'node_0_Relu', target: 'output_0', label: 'y' },
  ],
})

describe('v2.4 graphUtils', () => {
  it('currentInputBoundaryTensor returns the graph input\'s own name before any recipe', () => {
    expect(currentInputBoundaryTensor(recipeGraph, 0)).toBe('x')
  })

  it('insertRecipeNode (input anchor) rewires the consumer and wires the new node in between', () => {
    const next = insertRecipeNode(recipeGraph, 'input', 0, 'x', 'recipe_1', 'Cast', [{ name: 'to', value: 1 }], 0, 0)
    const recipe = next.nodes.find((n) => n.id === 'recipe_1')!
    expect(recipe.inputs).toEqual(['x'])
    expect(recipe.outputs).toEqual(['recipe_1_out'])
    expect(recipe.attributes).toEqual({ to: 1 })
    expect(next.nodes.find((n) => n.id === 'node_0_Relu')?.inputs).toEqual(['recipe_1_out'])
    expect(next.edges.find((e) => e.source === 'input_0')?.target).toBe('recipe_1')
  })

  it('insertRecipeNode chains correctly: currentInputBoundaryTensor advances after insertion', () => {
    const next = insertRecipeNode(recipeGraph, 'input', 0, 'x', 'recipe_1', 'Cast', [], 0, 0)
    expect(currentInputBoundaryTensor(next, 0)).toBe('recipe_1_out')
  })

  it('insertRecipeNode (output anchor) preserves the public output name on the new node', () => {
    const next = insertRecipeNode(recipeGraph, 'output', 0, '', 'recipe_2', 'Softmax', [{ name: 'axis', value: -1 }], 0, 0)
    const recipe = next.nodes.find((n) => n.id === 'recipe_2')!
    expect(recipe.inputs).toEqual(['recipe_2_in'])
    expect(recipe.outputs).toEqual(['y'])
    expect(next.nodes.find((n) => n.id === 'node_0_Relu')?.outputs).toEqual(['recipe_2_in'])
    expect(next.nodes.find((n) => n.id === 'output_0')?.inputs).toEqual(['y']) // unchanged
  })

  it('insertRecipeNode extraOutputCount adds trailing outputs without a new canvas pseudo-node', () => {
    const next = insertRecipeNode(recipeGraph, 'output', 0, '', 'recipe_2', 'TopK', [], 1, 1)
    const recipe = next.nodes.find((n) => n.id === 'recipe_2')!
    expect(recipe.outputs).toEqual(['y', 'recipe_2_extra0'])
    expect(next.nodes.some((n) => n.opType === 'Output' && n.id !== 'output_0')).toBe(false)
  })

  it('friendlyNodeLabel and structuralNodeIndex recognize recipe_N ids', () => {
    expect(friendlyNodeLabel('recipe_3')).toBe('Recipe node 3')
    expect(structuralNodeIndex('recipe_3')).toBe(-3)
  })

  it('describeHistoryEntry summarizes an insertRecipe entry by anchor kind', () => {
    const text = describeHistoryEntry({
      type: 'insertRecipe', anchorKind: 'input', ioIndex: 0, anchorTensor: 'x', newNodeId: 'recipe_1',
      newNodeIndex: 1, opType: 'Cast', recipeLabel: 'Cast to float32', attrs: [], extraInputs: [], extraOutputCount: 0, extraOutputElemTypes: [],
    })
    expect(text).toMatch(/Cast to float32.*preprocessing/)
  })
})

describe('v2.4 resolveRecipe (opset adaptation)', () => {
  const topK = PIPELINE_RECIPES.find((r) => r.id === 'top-k-5')!

  it('leaves Top-K on the modern 2-input encoding for opset 10+', () => {
    expect(resolveRecipe(topK, 13)).toBe(topK)
    expect(resolveRecipe(topK, 10)).toBe(topK)
  })

  it('falls back to the legacy 1-input + k-attribute encoding below opset 10', () => {
    const resolved = resolveRecipe(topK, 7)
    expect(resolved.extraInputs).toEqual([])
    expect(resolved.attrs.find((a) => a.name === 'k')).toEqual({ name: 'k', kind: 'I', value: 5 })
    expect(resolved.attrs.some((a) => a.name === 'largest')).toBe(false)
  })

  it('leaves every other recipe untouched regardless of opset', () => {
    const cast = PIPELINE_RECIPES.find((r) => r.id === 'cast-float32')!
    expect(resolveRecipe(cast, 7)).toBe(cast)
  })

  it('leaves Top-K untouched when the opset is unknown', () => {
    expect(resolveRecipe(topK, undefined)).toBe(topK)
  })

  it('Resize (no legacy encoding exists) declares a minOpset instead of trying to adapt', () => {
    const resize = PIPELINE_RECIPES.find((r) => r.id === 'resize-2x-nearest')!
    expect(resize.minOpset).toBe(10)
  })
})

describe('v2.4 share links', () => {
  it('refuses to encode a share hash containing an insertRecipe edit', () => {
    expect(() => createShareHash(
      'ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0',
      'model.onnx',
      [{ type: 'insertRecipe', anchorKind: 'input', ioIndex: 0, anchorTensor: 'x', newNodeId: 'recipe_1', newNodeIndex: 1, opType: 'Cast', recipeLabel: 'Cast to float32', attrs: [], extraInputs: [], extraOutputCount: 0, extraOutputElemTypes: [] }],
    )).toThrow(/insertRecipe/)
  })
})

// ---- App wiring --------------------------------------------------------------

const appGraph: OnnxGraph = {
  modelName: 'io.onnx',
  totalParams: 0,
  totalSizeMB: 0,
  nodes: [
    { id: 'input_0', opType: 'Input', inputs: [], outputs: ['x'], attributes: {}, paramCount: 0, estimatedSizeMB: 0, outputShapes: [[{ value: 1 }, { value: 4 }]], outputMetadata: [{ elemType: 1, shape: [{ value: 1 }, { value: 4 }] }] },
    { id: 'node_0_Relu', opType: 'Relu', inputs: ['x'], outputs: ['y'], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
    { id: 'output_0', opType: 'Output', inputs: ['y'], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 },
  ],
  edges: [
    { id: 'input_0->node_0_Relu@x', source: 'input_0', target: 'node_0_Relu', label: 'x' },
    { id: 'node_0_Relu->output_0@y', source: 'node_0_Relu', target: 'output_0', label: 'y' },
  ],
  metadata: { irVersion: 8, producerName: '', producerVersion: '', opsetVersion: 13, docString: '' },
}

const legacyOpsetGraph: OnnxGraph = {
  ...appGraph,
  metadata: { irVersion: 3, producerName: '', producerVersion: '', opsetVersion: 7, docString: '' },
}

const graphCanvasState = vi.hoisted(() => ({
  props: null as null | { onNodeSelect?: (id: string) => void },
}))

vi.mock('../components/GraphCanvas', () => ({
  GraphCanvas: (props: typeof graphCanvasState.props) => {
    graphCanvasState.props = props
    return null
  },
}))

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
  graphCanvasState.props = null
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

function selectNode(id: string) {
  act(() => { graphCanvasState.props?.onNodeSelect?.(id) })
}

describe('v2.4 App wiring', () => {
  it('shows preprocessing recipes on an Input pseudo-node, not postprocessing ones', async () => {
    render(<App />)
    loadGraph(appGraph)
    selectNode('input_0')

    expect(await screen.findByTestId('insert-recipe-cast-float32')).toBeInTheDocument()
    expect(screen.queryByTestId('insert-recipe-softmax')).not.toBeInTheDocument()
  })

  it('shows postprocessing recipes on an Output pseudo-node, not preprocessing ones', async () => {
    render(<App />)
    loadGraph(appGraph)
    selectNode('output_0')

    expect(await screen.findByTestId('insert-recipe-softmax')).toBeInTheDocument()
    expect(screen.queryByTestId('insert-recipe-cast-float32')).not.toBeInTheDocument()
  })

  it('inserting a recipe records a history entry and announces it', async () => {
    render(<App />)
    loadGraph(appGraph)
    selectNode('input_0')

    fireEvent.click(await screen.findByTestId('insert-recipe-cast-float32'))
    expect(await screen.findByTestId('announcement')).toHaveTextContent(/Inserted Cast to float32/)
    expect(screen.getByTestId('history-tab')).toHaveTextContent('History (1)')
    fireEvent.click(screen.getByTestId('history-tab'))
    expect(screen.getByText(/Cast to float32.*preprocessing/)).toBeInTheDocument()
  })

  it('chains a second preprocessing recipe onto the same input without error', async () => {
    render(<App />)
    loadGraph(appGraph)
    selectNode('input_0')

    fireEvent.click(await screen.findByTestId('insert-recipe-cast-float32'))
    await screen.findByTestId('announcement')
    fireEvent.click(await screen.findByTestId('insert-recipe-l2-normalize'))
    expect(await screen.findByTestId('announcement')).toHaveTextContent(/Inserted L2 Normalize/)
    expect(screen.getByTestId('history-tab')).toHaveTextContent('History (2)')
  })

  it('undo removes the inserted recipe from history', async () => {
    render(<App />)
    loadGraph(appGraph)
    selectNode('input_0')

    fireEvent.click(await screen.findByTestId('insert-recipe-cast-float32'))
    expect(await screen.findByText(/Export Modified \(1\)/)).toBeInTheDocument()

    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    expect(await screen.findByTestId('announcement')).toHaveTextContent(/Undo/)
    expect(screen.queryByText(/Export Modified/)).not.toBeInTheDocument()
  })

  it('a freshly inserted recipe node\'s attribute is editable, and the edit is a separate history entry', async () => {
    render(<App />)
    loadGraph(appGraph)
    selectNode('input_0')

    fireEvent.click(await screen.findByTestId('insert-recipe-cast-float32'))
    await screen.findByTestId('announcement')
    expect(screen.getByTestId('history-tab')).toHaveTextContent('History (1)')

    selectNode('recipe_1')
    const attrButton = await screen.findByTestId('attr-value-to')
    expect(attrButton).toHaveTextContent('1')
    fireEvent.click(attrButton)
    fireEvent.change(screen.getByTestId('attr-input-to'), { target: { value: '6' } })
    fireEvent.keyDown(screen.getByTestId('attr-input-to'), { key: 'Enter' })

    expect(await screen.findByTestId('attr-value-to')).toHaveTextContent('6')
    expect(screen.getByTestId('history-tab')).toHaveTextContent('History (2)')
  })

  it('hides Resize on a model whose declared opset predates it, unlike the opset-adaptable Top-K', async () => {
    render(<App />)
    loadGraph(legacyOpsetGraph)
    selectNode('input_0')
    expect(await screen.findByTestId('insert-recipe-cast-float32')).toBeInTheDocument()
    expect(screen.queryByTestId('insert-recipe-resize-2x-nearest')).not.toBeInTheDocument()
  })
})
