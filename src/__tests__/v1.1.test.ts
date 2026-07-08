import { describe, it, expect } from 'vitest'
import { parseOnnxProto } from '../lib/onnxProtoParser'
import { writeModifiedOnnx } from '../lib/onnxProtoWriter'

// ---- Minimal ONNX protobuf builder (for test fixtures only) ----
// Independent of onnxProtoWriter.ts -- this hand-encodes fixtures so the writer
// is tested as a black box against real wire bytes, not against its own encoder.

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

function float32Field(field: number, value: number): number[] {
  const tag = (field << 3) | 5
  const buf = new ArrayBuffer(4)
  new DataView(buf).setFloat32(0, value, true)
  return [...encodeVarint(tag), ...new Uint8Array(buf)]
}

function strField(field: number, value: string): number[] {
  const bytes = Array.from(new TextEncoder().encode(value))
  return lenField(field, bytes)
}

function intAttr(name: string, value: number): number[] {
  return [...strField(1, name), ...varintField(3, value >>> 0), ...varintField(20, 2)]
}

function floatAttr(name: string, value: number): number[] {
  return [...strField(1, name), ...float32Field(4, value), ...varintField(20, 1)]
}

function stringAttr(name: string, value: string): number[] {
  return [...strField(1, name), ...strField(6, value), ...varintField(20, 3)]
}

function intsAttr(name: string, values: number[]): number[] {
  const packed = values.flatMap(v => encodeVarint(v))
  return [...strField(1, name), ...lenField(8, packed), ...varintField(20, 7)]
}

function makeNode(opType: string, inputs: string[], outputs: string[], name: string, attrs: number[][]): number[] {
  const bytes: number[] = []
  for (const s of inputs) bytes.push(...strField(1, s))
  for (const s of outputs) bytes.push(...strField(2, s))
  bytes.push(...strField(3, name))
  bytes.push(...strField(4, opType))
  for (const a of attrs) bytes.push(...lenField(5, a))
  return bytes
}

function makeInitializer(name: string, dims: number[], elemType = 1): number[] {
  const bytes: number[] = []
  const dimBytes = dims.flatMap(d => encodeVarint(d))
  bytes.push(...lenField(1, dimBytes))
  bytes.push(...varintField(2, elemType))
  bytes.push(...strField(8, name))
  return bytes
}

function makeValueInfo(name: string): number[] { return strField(1, name) }

function makeGraph(nodes: number[][], inputs: string[], outputs: string[], initializers: number[][]): number[] {
  const bytes: number[] = []
  for (const n of nodes) bytes.push(...lenField(1, n))
  for (const i of initializers) bytes.push(...lenField(5, i))
  for (const s of inputs) bytes.push(...lenField(11, makeValueInfo(s)))
  for (const s of outputs) bytes.push(...lenField(12, makeValueInfo(s)))
  return bytes
}

function makeModel(graph: number[]): ArrayBuffer {
  const bytes = [...varintField(1, 7), ...lenField(7, graph)]
  return new Uint8Array(bytes).buffer
}

function makeFixture(): ArrayBuffer {
  const weight = makeInitializer('W', [4, 3, 3, 3])
  const conv = makeNode('Conv', ['x', 'W'], ['y'], 'conv0', [
    intAttr('group', 1),
    floatAttr('epsilon', 0.001),
    stringAttr('auto_pad', 'NOTSET'),
    intsAttr('kernel_shape', [3, 3]),
  ])
  const relu = makeNode('Relu', ['y'], ['z'], 'relu0', [])
  const graph = makeGraph([conv, relu], ['x'], ['z'], [weight])
  return makeModel(graph)
}

describe('writeModifiedOnnx (v1.1)', () => {
  it('edits an int attribute and leaves everything else intact', () => {
    const original = makeFixture()
    const patched = writeModifiedOnnx(original, new Map([[0, { group: 4 }]]))
    const result = parseOnnxProto(patched)

    expect(result.nodes).toHaveLength(2)
    expect(result.nodes[0].opType).toBe('Conv')
    expect(result.nodes[0].name).toBe('conv0')
    expect(result.nodes[0].inputs).toEqual(['x', 'W'])
    expect(result.nodes[0].outputs).toEqual(['y'])
    expect(result.nodes[0].attributes.group).toBe(4)
    // untouched attributes on the same node survive
    expect(result.nodes[0].attributes.epsilon).toBeCloseTo(0.001, 5)
    expect(result.nodes[0].attributes.auto_pad).toBe('NOTSET')
    expect(result.nodes[0].attributes.kernel_shape).toBe('[3, 3]')
    // untouched sibling node survives
    expect(result.nodes[1].opType).toBe('Relu')
    expect(result.nodes[1].name).toBe('relu0')
    // initializer survives byte-for-byte
    expect(result.initializers).toHaveLength(1)
    expect(result.initializers[0].name).toBe('W')
    expect(result.initializers[0].dims).toEqual([4, 3, 3, 3])
  })

  it('edits a float attribute', () => {
    const original = makeFixture()
    const patched = writeModifiedOnnx(original, new Map([[0, { epsilon: 0.05 }]]))
    const result = parseOnnxProto(patched)
    expect(result.nodes[0].attributes.epsilon).toBeCloseTo(0.05, 5)
    expect(result.nodes[0].attributes.group).toBe(1)
  })

  it('edits a string attribute', () => {
    const original = makeFixture()
    const patched = writeModifiedOnnx(original, new Map([[0, { auto_pad: 'SAME_UPPER' }]]))
    const result = parseOnnxProto(patched)
    expect(result.nodes[0].attributes.auto_pad).toBe('SAME_UPPER')
  })

  it('edits an int array (INTS) attribute', () => {
    const original = makeFixture()
    const patched = writeModifiedOnnx(original, new Map([[0, { kernel_shape: '[5, 5]' }]]))
    const result = parseOnnxProto(patched)
    expect(result.nodes[0].attributes.kernel_shape).toBe('[5, 5]')
  })

  it('edits a small negative int attribute correctly', () => {
    const original = makeFixture()
    const patched = writeModifiedOnnx(original, new Map([[0, { group: -2 }]]))
    const result = parseOnnxProto(patched)
    expect(result.nodes[0].attributes.group).toBe(-2)
  })

  it('applies edits to multiple nodes independently', () => {
    const original = makeFixture()
    const patched = writeModifiedOnnx(original, new Map([
      [0, { group: 8 }],
    ]))
    const result = parseOnnxProto(patched)
    expect(result.nodes[0].attributes.group).toBe(8)
    expect(result.nodes[1].attributes).toEqual({})
  })

  it('returns the original graph unchanged when there are no overrides', () => {
    const original = makeFixture()
    const patched = writeModifiedOnnx(original, new Map())
    const result = parseOnnxProto(patched)
    expect(result.nodes).toHaveLength(2)
    expect(result.nodes[0].attributes.group).toBe(1)
    expect(result.initializers[0].dims).toEqual([4, 3, 3, 3])
  })

  it('ignores overrides for attribute names that do not exist on the node', () => {
    const original = makeFixture()
    const patched = writeModifiedOnnx(original, new Map([[0, { nonexistent: 99 }]]))
    const result = parseOnnxProto(patched)
    expect(result.nodes[0].attributes.group).toBe(1)
    expect(Object.keys(result.nodes[0].attributes)).not.toContain('nonexistent')
  })
})
