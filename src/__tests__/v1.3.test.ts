import { describe, it, expect } from 'vitest'
import { isTfliteBuffer, parseTfliteGraph } from '../lib/tfliteParser'

// ---- Minimal FlatBuffers encoder (for test fixtures only) ----
// Independent of tfliteParser.ts's own reader -- this hand-encodes fixtures so the
// parser is tested as a black box against real wire bytes, matching the same
// philosophy as the protobuf fixture builders in v1.1.test.ts/v1.2.test.ts.
//
// FlatBuffers forward offsets require the referenced data to sit at a HIGHER byte
// position than the field slot that points to it (uoffset is unsigned: target =
// slotPos + uoffsetValue >= slotPos). So every table here is built in two steps:
// reserve the table's vtable+body first (fixing its slot positions at the current,
// lower write position), then write its children afterward (at higher positions,
// since the writer only ever appends) and patch each slot with the now-known child
// position. This is the opposite order from a naive "build children first" approach,
// which would produce negative (invalid) offsets.

class FbWriter {
  private view: DataView
  private bytes: Uint8Array
  private pos = 0

  constructor(size = 1 << 16) {
    const buf = new ArrayBuffer(size)
    this.view = new DataView(buf)
    this.bytes = new Uint8Array(buf)
  }

  private alloc(n: number): number {
    const p = this.pos
    this.pos += n
    return p
  }

  reserveHeader(): void {
    this.alloc(4) // root offset, patched in finish()
    const p = this.alloc(4)
    this.bytes.set(new TextEncoder().encode('TFL3'), p)
  }

  finish(rootPos: number): ArrayBuffer {
    this.view.setUint32(0, rootPos, true)
    return this.view.buffer.slice(0, this.pos)
  }

  writeString(s: string): number {
    const utf8 = new TextEncoder().encode(s)
    const p = this.alloc(4 + utf8.length)
    this.view.setUint32(p, utf8.length, true)
    this.bytes.set(utf8, p + 4)
    return p
  }

  writeInt32Vector(values: number[]): number {
    const p = this.alloc(4 + values.length * 4)
    this.view.setUint32(p, values.length, true)
    values.forEach((v, i) => this.view.setInt32(p + 4 + i * 4, v, true))
    return p
  }

  // The parser only reads this vector's element COUNT (byte length), never the raw
  // bytes, so the content is left zero-filled.
  writeByteVectorOfLength(byteLength: number): number {
    const p = this.alloc(4 + byteLength)
    this.view.setUint32(p, byteLength, true)
    return p
  }

  // Reserves a table's vtable + body. `fieldWidths[i]` is the byte width of field i
  // (1 for int8 scalars, 4 for uint32/int32 scalars and all offset fields). `scalars`
  // supplies immediate values for genuinely-scalar fields; any field not present in
  // `scalars` (including every offset field) is written as 0 and expected to be
  // patched via `patchOffset` once its child is written.
  startTable(fieldWidths: number[], scalars: Record<number, number>): { pos: number; patchOffset: (fieldId: number, targetPos: number) => void } {
    const fieldOffsets: number[] = []
    let cursor = 4 // past the soffset
    for (const w of fieldWidths) {
      fieldOffsets.push(cursor)
      cursor += w
    }
    const tableSize = cursor
    const vtableSize = 4 + fieldWidths.length * 2

    const vtablePos = this.alloc(vtableSize)
    this.view.setUint16(vtablePos, vtableSize, true)
    this.view.setUint16(vtablePos + 2, tableSize, true)
    fieldOffsets.forEach((off, i) => this.view.setUint16(vtablePos + 4 + i * 2, off, true))

    const tablePos = this.alloc(tableSize)
    this.view.setInt32(tablePos, tablePos - vtablePos, true)
    fieldWidths.forEach((width, i) => {
      const slotPos = tablePos + fieldOffsets[i]
      const value = scalars[i] ?? 0
      if (width === 1) this.view.setInt8(slotPos, value)
      else this.view.setUint32(slotPos, value >>> 0, true)
    })

    return {
      pos: tablePos,
      patchOffset: (fieldId: number, targetPos: number) => {
        const slotPos = tablePos + fieldOffsets[fieldId]
        this.view.setUint32(slotPos, targetPos - slotPos, true)
      },
    }
  }

  startTableVector(count: number): { pos: number; patchSlot: (i: number, targetPos: number) => void } {
    const p = this.alloc(4 + count * 4)
    this.view.setUint32(p, count, true)
    return {
      pos: p,
      patchSlot: (i: number, targetPos: number) => {
        const slotPos = p + 4 + i * 4
        this.view.setUint32(slotPos, targetPos - slotPos, true)
      },
    }
  }
}

// Builds a minimal single-op TFLite model: input(shape [1,4,4,3]) + weight(shape
// [3,3,3,1], 12 bytes) -> CONV_2D (or whatever builtinCode is given) -> output(shape
// [1,2,2,1]). Mirrors the shape of a single conv layer.
function makeTfliteFixture(builtinCode: number): ArrayBuffer {
  const w = new FbWriter()
  w.reserveHeader()

  // Model: version(0,skip) operator_codes(1) subgraphs(2) description(3,skip) buffers(4)
  const model = w.startTable([4, 4, 4, 4, 4], {})

  const opCodesVec = w.startTableVector(1)
  // OperatorCode: deprecated_builtin_code(0,i8) custom_code(1,skip) version(2,skip) builtin_code(3,i32)
  const opCode0 = w.startTable([1, 4, 4, 4], { 3: builtinCode })
  opCodesVec.patchSlot(0, opCode0.pos)
  model.patchOffset(1, opCodesVec.pos)

  const buffersVec = w.startTableVector(2)
  // Buffer 0: the mandatory empty sentinel -- data field left absent entirely.
  const buf0 = w.startTable([4], {})
  buffersVec.patchSlot(0, buf0.pos)
  // Buffer 1: 12 bytes of weight data (3x float32).
  const buf1 = w.startTable([4], {})
  const buf1Data = w.writeByteVectorOfLength(12)
  buf1.patchOffset(0, buf1Data)
  buffersVec.patchSlot(1, buf1.pos)
  model.patchOffset(4, buffersVec.pos)

  const subgraphsVec = w.startTableVector(1)
  // SubGraph: tensors(0) inputs(1) outputs(2) operators(3)
  const sg = w.startTable([4, 4, 4, 4], {})
  subgraphsVec.patchSlot(0, sg.pos)
  model.patchOffset(2, subgraphsVec.pos)

  const tensorsVec = w.startTableVector(3)
  // Tensor: shape(0) type(1,i8) buffer(2,u32) name(3)
  const t0 = w.startTable([4, 1, 4, 4], { 1: 0, 2: 0 }) // input: FLOAT32, buffer 0 (empty)
  t0.patchOffset(0, w.writeInt32Vector([1, 4, 4, 3]))
  t0.patchOffset(3, w.writeString('input'))
  tensorsVec.patchSlot(0, t0.pos)

  const t1 = w.startTable([4, 1, 4, 4], { 1: 0, 2: 1 }) // weight: FLOAT32, buffer 1
  t1.patchOffset(0, w.writeInt32Vector([3, 3, 3, 1]))
  t1.patchOffset(3, w.writeString('weight'))
  tensorsVec.patchSlot(1, t1.pos)

  const t2 = w.startTable([4, 1, 4, 4], { 1: 0, 2: 0 }) // output: FLOAT32, buffer 0 (empty)
  t2.patchOffset(0, w.writeInt32Vector([1, 2, 2, 1]))
  t2.patchOffset(3, w.writeString('output'))
  tensorsVec.patchSlot(2, t2.pos)
  sg.patchOffset(0, tensorsVec.pos)

  sg.patchOffset(1, w.writeInt32Vector([0])) // graph input: tensor index 0
  sg.patchOffset(2, w.writeInt32Vector([2])) // graph output: tensor index 2

  const opsVec = w.startTableVector(1)
  // Operator: opcode_index(0,u32) inputs(1) outputs(2)
  const op0 = w.startTable([4, 4, 4], { 0: 0 })
  op0.patchOffset(1, w.writeInt32Vector([0, 1])) // consumes input(0) and weight(1)
  op0.patchOffset(2, w.writeInt32Vector([2])) // produces output(2)
  opsVec.patchSlot(0, op0.pos)
  sg.patchOffset(3, opsVec.pos)

  return w.finish(model.pos)
}

describe('isTfliteBuffer (v1.3)', () => {
  it('recognizes the TFL3 file identifier at offset 4', () => {
    expect(isTfliteBuffer(makeTfliteFixture(3))).toBe(true)
  })

  it('rejects a buffer without the identifier', () => {
    const onnxLike = new Uint8Array([8, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4])
    expect(isTfliteBuffer(onnxLike.buffer)).toBe(false)
  })

  it('rejects a too-short buffer', () => {
    expect(isTfliteBuffer(new ArrayBuffer(4))).toBe(false)
  })
})

describe('parseTfliteGraph (v1.3)', () => {
  it('parses a single-op graph with correct topology, op name, shapes, and weight size', () => {
    const graph = parseTfliteGraph(makeTfliteFixture(3), 'conv.tflite') // 3 = CONV_2D

    expect(graph.format).toBe('tflite')
    const opNode = graph.nodes.find((n) => n.opType === 'CONV_2D')
    expect(opNode).toBeDefined()
    expect(opNode?.paramCount).toBe(27) // 3*3*3*1 weight elements
    expect(opNode?.estimatedSizeMB).toBeCloseTo(12 / (1024 * 1024), 10)
    expect(opNode?.inputShapes?.[0]).toEqual([{ value: 1 }, { value: 4 }, { value: 4 }, { value: 3 }])
    expect(opNode?.outputShapes?.[0]).toEqual([{ value: 1 }, { value: 2 }, { value: 2 }, { value: 1 }])

    // input -> CONV_2D -> output wiring
    const inputNode = graph.nodes.find((n) => n.opType === 'Input')
    const outputNode = graph.nodes.find((n) => n.opType === 'Output')
    expect(inputNode).toBeDefined()
    expect(outputNode).toBeDefined()
    expect(graph.edges.some((e) => e.source === inputNode!.id && e.target === opNode!.id)).toBe(true)
    expect(graph.edges.some((e) => e.source === opNode!.id && e.target === outputNode!.id)).toBe(true)

    expect(graph.totalParams).toBe(27)
  })

  it('falls back to OP_<code> for an unrecognized builtin opcode instead of throwing', () => {
    const graph = parseTfliteGraph(makeTfliteFixture(9999), 'unknown-op.tflite')
    const opNode = graph.nodes.find((n) => n.opType.startsWith('OP_'))
    expect(opNode?.opType).toBe('OP_9999')
  })

  it('recognizes a known low-numbered opcode correctly (RELU = 19)', () => {
    const graph = parseTfliteGraph(makeTfliteFixture(19), 'relu.tflite')
    expect(graph.nodes.some((n) => n.opType === 'RELU')).toBe(true)
  })
})
