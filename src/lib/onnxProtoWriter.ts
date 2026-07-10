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
  NODE_INPUT,
  NODE_OUTPUT,
  NODE_NAME,
  NODE_OP_TYPE,
  NODE_ATTR,
  ATTR_NAME,
  ATTR_I,
  ATTR_F,
  ATTR_S,
  ATTR_INTS,
  ATTR_FLOATS,
  ATTR_TYPE,
} from './onnxProtoParser'

// Structural edits are keyed by the node's original position in GraphProto.node
// (0-based, matching the `node_<idx>_<opType>` id scheme from onnxParser.ts).
// keepInputPosition/inputPosition index into that node's CURRENT .inputs[] array
// (not a captured tensor-name value) so ops replay correctly in sequence even when
// an earlier op already renamed the tensor at that position -- positions within a
// single node's own input list stay stable across edits elsewhere in the graph;
// tensor-name values do not.
export type StructuralOp =
  | { type: 'delete'; nodeIndex: number; keepInputPosition: number | null }
  | { type: 'insertPassthrough'; targetNodeIndex: number; inputPosition: number; newNodeName: string }

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

// ---- Structural editing (delete node / insert passthrough) ----
//
// patchLenFields only supports "replace this occurrence" or "keep verbatim" -- it
// has no way to omit or append occurrences, so it can't delete or insert a
// GRAPH_NODE entry on its own. The structural path below decodes each node's raw
// bytes into an array (one entry per NodeProto, in original order), performs
// array-level splice/rewire operations, then re-encodes. Every entry always tracks
// its ORIGINAL index (its position in GraphProto.node before any edits), which is
// what structuralOps reference -- deletion never renumbers survivors.

interface NodeEntry {
  origIndex: number
  bytes: Uint8Array
  inputs: string[]
  outputs: string[]
}

function decodeNodeIO(nodeBytes: Uint8Array): { inputs: string[]; outputs: string[] } {
  const r = new ProtoReader(nodeBytes)
  const inputs: string[] = []
  const outputs: string[] = []
  while (!r.done) {
    const tag = r.readTag()
    if (!tag) break
    if (tag.wire === WIRE_LEN) {
      const len = r.readVarint()
      if (tag.field === NODE_INPUT) inputs.push(r.readString(len))
      else if (tag.field === NODE_OUTPUT) outputs.push(r.readString(len))
      else r.skip(len)
    } else {
      r.skipField(tag.wire)
    }
  }
  return { inputs, outputs }
}

// The bytes are always the source of truth; inputs/outputs are re-derived from
// them after every mutation rather than patched separately, so the two can never
// drift out of sync.
function decodeEntry(origIndex: number, bytes: Uint8Array): NodeEntry {
  return { origIndex, bytes, ...decodeNodeIO(bytes) }
}

function buildIdentityNodeBytes(inputTensor: string, outputTensor: string, name: string): Uint8Array {
  return concatBytes([
    encodeStringField(NODE_INPUT, inputTensor),
    encodeStringField(NODE_OUTPUT, outputTensor),
    encodeStringField(NODE_NAME, name),
    encodeStringField(NODE_OP_TYPE, 'Identity'),
  ])
}

// Replaces every NODE_INPUT occurrence whose VALUE equals `from` with `to`. Used
// for delete-reconnection: a deleted node's output tensor name may be consumed at
// more than one input position on the same downstream node (e.g. Add(x, x)), and
// every such reference must be rewired, not just the first.
function rewireInputsByValue(bytes: Uint8Array, from: string, to: string): Uint8Array {
  return patchLenFields(bytes, NODE_INPUT, (_occ, sub) =>
    new TextDecoder().decode(sub) === from ? new TextEncoder().encode(to) : null,
  )
}

// Replaces the NODE_INPUT occurrence at a specific POSITION (not value) with a new
// tensor name. Used for passthrough insertion, which targets one specific input
// slot the user clicked -- other consumers of the same original tensor, or other
// positions on the same node that happen to share that tensor's name, are untouched.
function rewireInputAtPosition(bytes: Uint8Array, position: number, to: string): Uint8Array {
  return patchLenFields(bytes, NODE_INPUT, (occ, _sub) =>
    occ === position ? new TextEncoder().encode(to) : null,
  )
}

function applyStructuralOps(nodeEntries: NodeEntry[], structuralOps: StructuralOp[]): NodeEntry[] {
  let entries = nodeEntries
  let insertCounter = 0

  for (const op of structuralOps) {
    if (op.type === 'delete') {
      const idx = entries.findIndex((e) => e.origIndex === op.nodeIndex)
      if (idx === -1) continue
      const entry = entries[idx]
      const removedTensor = entry.outputs[0]
      const keepTensor = op.keepInputPosition !== null ? entry.inputs[op.keepInputPosition] : undefined

      entries = entries.filter((_, i) => i !== idx)
      if (keepTensor !== undefined && removedTensor) {
        entries = entries.map((e) =>
          e.inputs.includes(removedTensor) ? decodeEntry(e.origIndex, rewireInputsByValue(e.bytes, removedTensor, keepTensor)) : e,
        )
      }
    } else if (op.type === 'insertPassthrough') {
      const idx = entries.findIndex((e) => e.origIndex === op.targetNodeIndex)
      if (idx === -1) continue
      const target = entries[idx]
      const tensorName = target.inputs[op.inputPosition]
      if (!tensorName) continue

      insertCounter++
      const newTensorName = `${tensorName}__identity_${insertCounter}_${op.newNodeName}`
      const rewiredTarget = decodeEntry(target.origIndex, rewireInputAtPosition(target.bytes, op.inputPosition, newTensorName))
      // Sentinel negative origIndex: structural ops only ever address original
      // (non-negative) indices, so inserted entries are never themselves a valid
      // op target -- chaining onto a generated node is out of scope by construction.
      const identityEntry = decodeEntry(-insertCounter, buildIdentityNodeBytes(tensorName, newTensorName, op.newNodeName))

      // Splice the new node immediately before its consumer, not appended at the
      // end: ONNX requires GraphProto.node to be topologically ordered (a node's
      // inputs must be defined by an earlier node), and the target already follows
      // its own producer, so inserting directly before it preserves that ordering.
      entries = [...entries.slice(0, idx), identityEntry, rewiredTarget, ...entries.slice(idx + 1)]
    }
  }

  return entries
}

// Decodes graphContent into per-node entries (applying attribute overrides along
// the way) plus every other field's bytes untouched, applies structuralOps, then
// re-encodes. Node entries are hoisted before other fields in the output, which is
// safe -- protobuf field order across distinct field numbers is not meaningful,
// only the relative order AMONG node entries (topological) matters, and that's
// preserved by construction above.
function rewriteGraphContent(
  graphContent: Uint8Array,
  overridesByNodeIndex: Map<number, Record<string, string | number>>,
  structuralOps: StructuralOp[],
): Uint8Array {
  const r = new ProtoReader(graphContent)
  const otherChunks: Uint8Array[] = []
  let entries: NodeEntry[] = []
  let nodeOccurrence = 0

  while (!r.done) {
    const fieldStart = r.pos
    const tag = r.readTag()
    if (!tag) break
    if (tag.wire === WIRE_LEN) {
      const len = r.readVarint()
      const subStart = r.pos
      r.skip(len)
      const fieldEnd = r.pos
      if (tag.field === GRAPH_NODE) {
        const nodeBytes = graphContent.subarray(subStart, fieldEnd)
        const overrides = overridesByNodeIndex.get(nodeOccurrence)
        const patchedBytes = overrides
          ? patchLenFields(nodeBytes, NODE_ATTR, (_attrIdx, attrContent) => {
              const { name, kind } = identifyAttr(attrContent)
              if (kind === 'OTHER' || !(name in overrides)) return null
              return buildEditedAttrContent(name, kind, overrides[name])
            })
          : nodeBytes
        entries.push(decodeEntry(nodeOccurrence, patchedBytes))
        nodeOccurrence++
      } else {
        otherChunks.push(graphContent.subarray(fieldStart, fieldEnd))
      }
    } else {
      r.skipField(tag.wire)
      otherChunks.push(graphContent.subarray(fieldStart, r.pos))
    }
  }

  entries = applyStructuralOps(entries, structuralOps)
  const nodeChunks = entries.map((e) => encodeLenField(GRAPH_NODE, e.bytes))
  return concatBytes([...nodeChunks, ...otherChunks])
}

// overridesByNodeIndex is keyed by the node's position in GraphProto.node (0-based),
// matching the order onnxParser.ts assigns as `node_${idx}_${opType}` ids.
//
// When structuralOps is empty, this uses the original streaming patchLenFields pass
// (model -> graph -> node -> attr, replace-or-verbatim only) unchanged from before
// structural editing existed -- zero behavior change for the attribute-only export
// path. structuralOps only engages the array-based rewrite in rewriteGraphContent.
export function writeModifiedOnnx(
  buffer: ArrayBuffer,
  overridesByNodeIndex: Map<number, Record<string, string | number>>,
  structuralOps: StructuralOp[] = [],
): ArrayBuffer {
  const full = new Uint8Array(buffer)

  if (structuralOps.length === 0) {
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

  const graphPatcher = (_idx: number, graphContent: Uint8Array): Uint8Array | null =>
    rewriteGraphContent(graphContent, overridesByNodeIndex, structuralOps)
  const patched = patchLenFields(full, MODEL_GRAPH, graphPatcher)
  return patched.buffer as ArrayBuffer
}
