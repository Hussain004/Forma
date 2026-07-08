// Patches edited node attributes back into a raw ONNX protobuf buffer.
//
// Strategy: walk the original bytes field-by-field (reusing the parser's ProtoReader)
// and copy every field through verbatim except the ones that changed. Only the
// AttributeProto entries the user actually edited are re-encoded; initializer weights,
// tensor/graph-valued attributes, and everything else pass through byte-for-byte
// untouched. This avoids writing a full ONNX serializer and guarantees nothing we
// don't understand gets silently dropped.

import {
  ProtoReader,
  WIRE_VARINT,
  WIRE_LEN,
  WIRE_32BIT,
  MODEL_GRAPH,
  GRAPH_NODE,
  NODE_ATTR,
  ATTR_NAME,
  ATTR_I,
  ATTR_F,
  ATTR_S,
  ATTR_INTS,
  ATTR_FLOATS,
  ATTR_TYPE,
} from './onnxProtoParser'

type AttrKind = 'I' | 'F' | 'S' | 'INTS' | 'FLOATS' | 'OTHER'

// AttributeProto.AttributeType enum values (onnx.proto3)
const ATTR_TYPE_ENUM: Record<AttrKind, number> = { F: 1, I: 2, S: 3, FLOATS: 6, INTS: 7, OTHER: 0 }

function concatBytes(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((sum, c) => sum + c.length, 0)
  const out = new Uint8Array(total)
  let offset = 0
  for (const c of chunks) { out.set(c, offset); offset += c.length }
  return out
}

function encodeVarint(value: number): Uint8Array {
  const bytes: number[] = []
  let v = value >>> 0
  do {
    let b = v & 0x7f
    v >>>= 7
    if (v !== 0) b |= 0x80
    bytes.push(b)
  } while (v !== 0)
  return new Uint8Array(bytes)
}

// Full 64-bit two's complement varint, matching real protobuf encoders for negative int64.
function encodeInt64Varint(value: number): Uint8Array {
  let v = BigInt(Math.trunc(value))
  if (v < 0n) v &= 0xFFFFFFFFFFFFFFFFn
  const bytes: number[] = []
  do {
    let b = Number(v & 0x7Fn)
    v >>= 7n
    if (v !== 0n) b |= 0x80
    bytes.push(b)
  } while (v !== 0n)
  return new Uint8Array(bytes)
}

function encodeFloat32(value: number): Uint8Array {
  const buf = new ArrayBuffer(4)
  new DataView(buf).setFloat32(0, value, true)
  return new Uint8Array(buf)
}

function encodeTag(field: number, wire: number): Uint8Array {
  return encodeVarint((field << 3) | wire)
}

function encodeLenField(field: number, content: Uint8Array): Uint8Array {
  return concatBytes([encodeTag(field, WIRE_LEN), encodeVarint(content.length), content])
}

function encodeVarintField(field: number, value: number): Uint8Array {
  return concatBytes([encodeTag(field, WIRE_VARINT), encodeVarint(value)])
}

function encodeInt64Field(field: number, value: number): Uint8Array {
  return concatBytes([encodeTag(field, WIRE_VARINT), encodeInt64Varint(value)])
}

function encode32BitField(field: number, value: number): Uint8Array {
  return concatBytes([encodeTag(field, WIRE_32BIT), encodeFloat32(value)])
}

function encodeStringField(field: number, value: string): Uint8Array {
  return encodeLenField(field, new TextEncoder().encode(value))
}

function parseArrayString(s: string): number[] {
  return s.replace(/^\[|\]$/g, '').split(',').map(p => parseFloat(p.trim()))
}

// Walk `content` at the top level, replacing every occurrence of `targetField` (a
// length-delimited field) for which `patcher` returns non-null. Everything else --
// other fields, and occurrences where patcher returns null -- is copied through
// verbatim, byte for byte.
function patchLenFields(
  content: Uint8Array,
  targetField: number,
  patcher: (occurrenceIndex: number, subContent: Uint8Array) => Uint8Array | null,
): Uint8Array {
  const r = new ProtoReader(content)
  const chunks: Uint8Array[] = []
  let occurrence = 0
  while (!r.done) {
    const fieldStart = r.pos
    const tag = r.readTag()
    if (!tag) break
    if (tag.wire === WIRE_LEN) {
      const len = r.readVarint()
      const subStart = r.pos
      r.skip(len)
      const fieldEnd = r.pos
      if (tag.field === targetField) {
        const sub = content.subarray(subStart, fieldEnd)
        const replacement = patcher(occurrence, sub)
        occurrence++
        chunks.push(replacement ? encodeLenField(tag.field, replacement) : content.subarray(fieldStart, fieldEnd))
      } else {
        chunks.push(content.subarray(fieldStart, fieldEnd))
      }
    } else {
      r.skipField(tag.wire)
      chunks.push(content.subarray(fieldStart, r.pos))
    }
  }
  return concatBytes(chunks)
}

function identifyAttr(sub: Uint8Array): { name: string; kind: AttrKind } {
  const r = new ProtoReader(sub)
  let name = ''
  let kind: AttrKind = 'OTHER'
  while (!r.done) {
    const tag = r.readTag()
    if (!tag) break
    if (tag.wire === WIRE_LEN) {
      const len = r.readVarint()
      if (tag.field === ATTR_NAME) {
        name = r.readString(len)
      } else {
        if (tag.field === ATTR_S) kind = 'S'
        else if (tag.field === ATTR_INTS) kind = 'INTS'
        else if (tag.field === ATTR_FLOATS) kind = 'FLOATS'
        r.skip(len)
      }
    } else if (tag.wire === WIRE_VARINT) {
      const v = r.readVarint()
      if (tag.field === ATTR_I) kind = 'I'
      void v
    } else if (tag.wire === WIRE_32BIT) {
      r.skip(4)
      if (tag.field === ATTR_F) kind = 'F'
    } else {
      r.skipField(tag.wire)
    }
  }
  return { name, kind }
}

function buildEditedAttrContent(name: string, kind: AttrKind, newValue: string | number): Uint8Array {
  const parts: Uint8Array[] = [encodeStringField(ATTR_NAME, name)]
  if (kind === 'I') {
    parts.push(encodeInt64Field(ATTR_I, Number(newValue)))
  } else if (kind === 'F') {
    parts.push(encode32BitField(ATTR_F, Number(newValue)))
  } else if (kind === 'S') {
    parts.push(encodeStringField(ATTR_S, String(newValue)))
  } else if (kind === 'INTS') {
    const nums = parseArrayString(String(newValue)).map(n => Math.trunc(n))
    parts.push(encodeLenField(ATTR_INTS, concatBytes(nums.map(n => encodeInt64Varint(n)))))
  } else if (kind === 'FLOATS') {
    const nums = parseArrayString(String(newValue))
    parts.push(encodeLenField(ATTR_FLOATS, concatBytes(nums.map(n => encodeFloat32(n)))))
  }
  parts.push(encodeVarintField(ATTR_TYPE, ATTR_TYPE_ENUM[kind]))
  return concatBytes(parts)
}

// overridesByNodeIndex is keyed by the node's position in GraphProto.node (0-based),
// matching the order onnxParser.ts assigns as `node_${idx}_${opType}` ids.
export function writeModifiedOnnx(
  buffer: ArrayBuffer,
  overridesByNodeIndex: Map<number, Record<string, string | number>>,
): ArrayBuffer {
  const full = new Uint8Array(buffer)

  const nodePatcher = (idx: number, nodeContent: Uint8Array): Uint8Array | null => {
    const overrides = overridesByNodeIndex.get(idx)
    if (!overrides) return null
    return patchLenFields(nodeContent, NODE_ATTR, (_attrIdx, attrContent) => {
      const { name, kind } = identifyAttr(attrContent)
      if (kind === 'OTHER' || !(name in overrides)) return null
      return buildEditedAttrContent(name, kind, overrides[name])
    })
  }

  const graphPatcher = (_idx: number, graphContent: Uint8Array): Uint8Array | null =>
    patchLenFields(graphContent, GRAPH_NODE, nodePatcher)

  const patched = patchLenFields(full, MODEL_GRAPH, graphPatcher)
  return patched.buffer as ArrayBuffer
}
