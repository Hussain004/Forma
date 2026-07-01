import { describe, it, expect } from 'vitest'
import { createElement } from 'react'
import { render, screen } from '@testing-library/react'
import { parseOnnxProto } from '../lib/onnxProtoParser'
import { LayerInspector } from '../components/LayerInspector'
import type { OnnxNode, ModelMetadata } from '../lib/onnxTypes'

// ---- Proto fixture helpers (mirrors onnx.test.ts) ----

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

function strField(field: number, value: string): number[] {
  return lenField(field, Array.from(new TextEncoder().encode(value)))
}

function varintField(field: number, value: number): number[] {
  return [...encodeVarint((field << 3) | 0), ...encodeVarint(value)]
}

function makeOpset(domain: string, version: number): number[] {
  return [...strField(1, domain), ...varintField(2, version)]
}

function makeNode(opType: string, inputs: string[], outputs: string[], name = ''): number[] {
  const bytes: number[] = []
  for (const s of inputs) bytes.push(...strField(1, s))
  for (const s of outputs) bytes.push(...strField(2, s))
  if (name) bytes.push(...strField(3, name))
  bytes.push(...strField(4, opType))
  return bytes
}

function makeGraph(nodes: number[][], inputs: string[], outputs: string[]): number[] {
  const bytes: number[] = []
  for (const n of nodes) bytes.push(...lenField(1, n))
  for (const s of inputs) bytes.push(...lenField(11, strField(1, s)))
  for (const s of outputs) bytes.push(...lenField(12, strField(1, s)))
  return bytes
}

function makeModel(graph: number[], opts: { irVersion?: number; producer?: string; producerVersion?: string; opset?: number; docString?: string } = {}): ArrayBuffer {
  const bytes: number[] = []
  if (opts.irVersion !== undefined) bytes.push(...varintField(1, opts.irVersion))
  if (opts.producer) bytes.push(...strField(2, opts.producer))
  if (opts.producerVersion) bytes.push(...strField(3, opts.producerVersion))
  if (opts.docString) bytes.push(...strField(6, opts.docString))
  bytes.push(...lenField(7, graph))
  if (opts.opset !== undefined) bytes.push(...lenField(8, makeOpset('', opts.opset)))
  return new Uint8Array(bytes).buffer
}

// ---- Group 1: parseOnnxProto metadata ----

describe('parseOnnxProto metadata (v0.10)', () => {
  it('parses ir_version from ModelProto', () => {
    const buf = makeModel(makeGraph([makeNode('Relu', ['x'], ['y'])], ['x'], ['y']), { irVersion: 8 })
    const result = parseOnnxProto(buf)
    expect(result.metadata.irVersion).toBe(8)
  })

  it('parses producer_name from ModelProto', () => {
    const buf = makeModel(makeGraph([], [], []), { producer: 'pytorch', producerVersion: '2.1.0' })
    const result = parseOnnxProto(buf)
    expect(result.metadata.producerName).toBe('pytorch')
    expect(result.metadata.producerVersion).toBe('2.1.0')
  })

  it('parses default-domain opset version', () => {
    const buf = makeModel(makeGraph([], [], []), { opset: 17 })
    const result = parseOnnxProto(buf)
    expect(result.metadata.opsetVersion).toBe(17)
  })

  it('parses doc_string from ModelProto', () => {
    const buf = makeModel(makeGraph([], [], []), { docString: 'test model' })
    const result = parseOnnxProto(buf)
    expect(result.metadata.docString).toBe('test model')
  })

  it('returns zero/empty metadata for an empty buffer', () => {
    const result = parseOnnxProto(new ArrayBuffer(0))
    expect(result.metadata.irVersion).toBe(0)
    expect(result.metadata.producerName).toBe('')
    expect(result.metadata.opsetVersion).toBe(0)
  })

  it('existing tests still pass: parses nodes correctly alongside metadata', () => {
    const buf = makeModel(
      makeGraph([makeNode('Conv', ['x', 'W'], ['y'])], ['x'], ['y']),
      { irVersion: 9, producer: 'torch.onnx', opset: 18 },
    )
    const result = parseOnnxProto(buf)
    expect(result.nodes).toHaveLength(1)
    expect(result.nodes[0].opType).toBe('Conv')
    expect(result.metadata.irVersion).toBe(9)
    expect(result.metadata.producerName).toBe('torch.onnx')
    expect(result.metadata.opsetVersion).toBe(18)
  })
})

// ---- Group 2: node name parsing ----

describe('parseOnnxProto node name (v0.10)', () => {
  it('parses node name when present', () => {
    const node = makeNode('Conv', ['x'], ['y'], 'resnet/layer1/conv1')
    const buf = makeModel(makeGraph([node], ['x'], ['y']))
    const result = parseOnnxProto(buf)
    expect(result.nodes[0].name).toBe('resnet/layer1/conv1')
  })

  it('returns empty string for nodes without a name', () => {
    const node = makeNode('Relu', ['x'], ['y'])
    const buf = makeModel(makeGraph([node], ['x'], ['y']))
    const result = parseOnnxProto(buf)
    expect(result.nodes[0].name).toBe('')
  })
})

// ---- Group 3: LayerInspector metadata display ----

const mockMetadata: ModelMetadata = {
  irVersion: 9,
  producerName: 'pytorch',
  producerVersion: '2.1.0',
  opsetVersion: 18,
  docString: '',
}

describe('LayerInspector metadata rows (v0.10)', () => {
  it('shows PRODUCER row when producerName is set', () => {
    render(createElement(LayerInspector, {
      node: null,
      modelStats: { opCounts: {}, totalNodes: 5, metadata: mockMetadata },
    }))
    expect(screen.getByText('PRODUCER')).toBeInTheDocument()
    expect(screen.getByText(/pytorch/)).toBeInTheDocument()
  })

  it('includes producer version alongside producer name', () => {
    render(createElement(LayerInspector, {
      node: null,
      modelStats: { opCounts: {}, totalNodes: 5, metadata: mockMetadata },
    }))
    expect(screen.getByText('pytorch 2.1.0')).toBeInTheDocument()
  })

  it('shows OPSET row when opsetVersion is set', () => {
    render(createElement(LayerInspector, {
      node: null,
      modelStats: { opCounts: {}, totalNodes: 5, metadata: mockMetadata },
    }))
    expect(screen.getByText('OPSET')).toBeInTheDocument()
    expect(screen.getByText('18')).toBeInTheDocument()
  })

  it('shows IR VER row when irVersion is set', () => {
    render(createElement(LayerInspector, {
      node: null,
      modelStats: { opCounts: {}, totalNodes: 5, metadata: mockMetadata },
    }))
    expect(screen.getByText('IR VER')).toBeInTheDocument()
    expect(screen.getByText('9')).toBeInTheDocument()
  })

  it('omits PRODUCER row when producerName is empty', () => {
    const meta: ModelMetadata = { ...mockMetadata, producerName: '', producerVersion: '' }
    render(createElement(LayerInspector, {
      node: null,
      modelStats: { opCounts: {}, totalNodes: 5, metadata: meta },
    }))
    expect(screen.queryByText('PRODUCER')).not.toBeInTheDocument()
  })
})

// ---- Group 4: LayerInspector node name display ----

describe('LayerInspector node name row (v0.10)', () => {
  it('shows NODE NAME row when node.name is set', () => {
    const node: OnnxNode = {
      id: 'conv_0',
      name: 'resnet/layer1/conv1',
      opType: 'Conv',
      inputs: ['x'],
      outputs: ['y'],
      attributes: {},
      paramCount: 0,
      estimatedSizeMB: 0,
    }
    render(createElement(LayerInspector, { node }))
    expect(screen.getByText('NODE NAME')).toBeInTheDocument()
    expect(screen.getByText('resnet/layer1/conv1')).toBeInTheDocument()
  })

  it('omits NODE NAME row when node.name is absent', () => {
    const node: OnnxNode = {
      id: 'relu_0',
      opType: 'Relu',
      inputs: ['x'],
      outputs: ['y'],
      attributes: {},
      paramCount: 0,
      estimatedSizeMB: 0,
    }
    render(createElement(LayerInspector, { node }))
    expect(screen.queryByText('NODE NAME')).not.toBeInTheDocument()
  })
})
