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
  GRAPH_INIT,
  GRAPH_INPUT,
  GRAPH_OUTPUT,
  GRAPH_VALUE_INFO,
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
  INIT_DIMS,
  INIT_DATA_TYPE,
  INIT_NAME,
  INIT_RAW_DATA,
  VINFO_NAME,
  VINFO_TYPE,
  TYPE_TENSOR,
  TENSOR_ELEM_TYPE,
  TENSOR_SHAPE,
  SHAPE_DIM,
  DIM_VALUE,
  DIM_PARAM,
  DATA_TYPE_BYTES,
  readTopLevelStringField,
  parseInitializer,
  parseValueInfo,
  type OnnxDim,
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
  | { type: 'rewire'; targetNodeIndex: number; inputPosition: number; sourceNodeIndex: number }
  // newNodeIndex is a positive counter minted by the UI (graphUtils.structuralNodeIndex
  // negates it to get the entry's origIndex) -- rewire ops reference the resulting
  // node the same way they reference any original node, via that negative index.
  | { type: 'addNode'; newNodeIndex: number; opType: string; inputCount: number }
  | { type: 'renameNode'; nodeIndex: number; name: string }
  | { type: 'renameTensor'; oldName: string; newName: string }
  // ioIndex is the 0-based position within GraphProto.input/output, matching
  // graphUtils.graphIOIndex. dims: null omits the shape field (unranked), not
  // an empty/scalar shape.
  | { type: 'setGraphIO'; ioKind: 'input' | 'output'; ioIndex: number; elemType: number; dims: OnnxDim[] | null }
  | { type: 'promoteOutput'; tensorName: string }
  // values.length must equal the initializer's current element count -- see
  // SMALL_TENSOR_MAX_ELEMENTS in onnxProtoParser.ts.
  | { type: 'replaceConstant'; initializerName: string; values: number[] }
  // v2.4 guided recipe insertion (see pipelineRecipes.ts and graphUtils.ts's
  // insertRecipeNode, which this mirrors byte-for-byte). newNodeIndex mints
  // this entry's origIndex the same way addNode's does (origIndex = -newNodeIndex),
  // which is what makes a recipe node addressable by later ops (attr edits,
  // rename, delete) for free -- see applyStructuralOps below. anchorTensor is
  // only consulted for an 'input' anchor; an 'output' anchor re-derives its
  // target fresh from graphOutputs[ioIndex], which never changes across edits.
  | {
      type: 'insertRecipe'
      anchorKind: 'input' | 'output'
      ioIndex: number
      anchorTensor: string
      newNodeIndex: number
      opType: string
      attrs: { name: string; kind: 'I' | 'F' | 'S' | 'INTS' | 'FLOATS'; value: string | number }[]
      extraInputs: { kind: 'empty' | 'const'; elemType?: number; dims?: number[]; values?: number[] }[]
      extraOutputCount: number
      extraOutputElemTypes: number[]
    }

type AttrKind = 'I' | 'F' | 'S' | 'INTS' | 'FLOATS' | 'OTHER'

// AttributeProto.AttributeType enum values (onnx.proto3)
const ATTR_TYPE_ENUM: Record<AttrKind, number> = { F: 1, I: 2, S: 3, FLOATS: 6, INTS: 7, OTHER: 0 }

export function concatBytes(chunks: Uint8Array[]): Uint8Array {
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

export function encodeLenField(field: number, content: Uint8Array): Uint8Array {
  return concatBytes([encodeTag(field, WIRE_LEN), encodeVarint(content.length), content])
}

export function encodeVarintField(field: number, value: number): Uint8Array {
  return concatBytes([encodeTag(field, WIRE_VARINT), encodeVarint(value)])
}

function encodeInt64Field(field: number, value: number): Uint8Array {
  return concatBytes([encodeTag(field, WIRE_VARINT), encodeInt64Varint(value)])
}

function encode32BitField(field: number, value: number): Uint8Array {
  return concatBytes([encodeTag(field, WIRE_32BIT), encodeFloat32(value)])
}

export function encodeStringField(field: number, value: string): Uint8Array {
  return encodeLenField(field, new TextEncoder().encode(value))
}

function parseArrayString(s: string): number[] {
  return s.replace(/^\[|\]$/g, '').split(',').map(p => parseFloat(p.trim()))
}

// Walk `content` at the top level, replacing every occurrence of `targetField` (a
// length-delimited field) for which `patcher` returns non-null. Everything else --
// other fields, and occurrences where patcher returns null -- is copied through
// verbatim, byte for byte.
export function patchLenFields(
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

export function decodeNodeIO(nodeBytes: Uint8Array): { inputs: string[]; outputs: string[] } {
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

// A freshly added node (v1.5) starts with inputCount empty-string inputs -- ONNX's
// own convention for "no input at this position" (already used elsewhere in this
// codebase for TFLite's -1/optional-input translation), not a placeholder tensor
// name. Any position the user wires up before export gets overwritten in place by
// rewireInputAtPosition below; anything left unwired exports as a harmlessly
// omitted optional input rather than a dangling reference to a name nothing
// produces. No attributes: the curated op list is chosen to need none.
function buildCustomNodeBytes(opType: string, inputCount: number, outputTensor: string, name: string): Uint8Array {
  const parts: Uint8Array[] = []
  for (let i = 0; i < inputCount; i++) parts.push(encodeStringField(NODE_INPUT, ''))
  parts.push(encodeStringField(NODE_OUTPUT, outputTensor))
  parts.push(encodeStringField(NODE_NAME, name))
  parts.push(encodeStringField(NODE_OP_TYPE, opType))
  return concatBytes(parts)
}

// A recipe node (v2.4) always has real, caller-specified inputs/outputs and
// initial attributes -- unlike buildCustomNodeBytes's empty-placeholder,
// attribute-free nodes, everything here is baked in up front. attrBytes are
// already-encoded NODE_ATTR occurrences (see buildEditedAttrContent, reused
// as-is since a fresh attribute and a patched one are the same bytes).
function buildRecipeNodeBytes(opType: string, inputs: string[], outputs: string[], name: string, attrBytes: Uint8Array[]): Uint8Array {
  const parts: Uint8Array[] = []
  for (const inp of inputs) parts.push(encodeStringField(NODE_INPUT, inp))
  for (const out of outputs) parts.push(encodeStringField(NODE_OUTPUT, out))
  parts.push(encodeStringField(NODE_NAME, name))
  parts.push(encodeStringField(NODE_OP_TYPE, opType))
  parts.push(...attrBytes)
  return concatBytes(parts)
}

// A small new TensorProto for one of a recipe's constant extra inputs (e.g.
// Resize's scales, TopK's K) -- dims are unpacked int64 varints, matching
// onnxProtoParser's own read of INIT_DIMS (see its "unpacked (proto2 compat)"
// comment).
function buildInitializerBytes(name: string, elemType: number, dims: number[], values: number[]): Uint8Array {
  const parts = dims.map((d) => encodeInt64Field(INIT_DIMS, d))
  parts.push(encodeVarintField(INIT_DATA_TYPE, elemType))
  parts.push(encodeStringField(INIT_NAME, name))
  parts.push(encodeLenField(INIT_RAW_DATA, encodeRawData(elemType, values)))
  return concatBytes(parts)
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

// Replaces every NODE_OUTPUT occurrence whose value equals `from` -- the
// output-side counterpart to rewireInputsByValue above, used by renameTensor.
function renameOutputsByValue(bytes: Uint8Array, from: string, to: string): Uint8Array {
  return patchLenFields(bytes, NODE_OUTPUT, (_occ, sub) =>
    new TextDecoder().decode(sub) === from ? new TextEncoder().encode(to) : null,
  )
}

// Insert-or-replace a top-level string field -- patchLenFields alone can only
// replace an EXISTING occurrence, but a node that never had a name (a common,
// valid ONNX state) needs one appended instead. Used for both NodeProto.name
// (renameNode) and TensorProto/ValueInfoProto.name (renameTensor).
function setStringField(bytes: Uint8Array, field: number, value: string): Uint8Array {
  let found = false
  const patched = patchLenFields(bytes, field, () => { found = true; return new TextEncoder().encode(value) })
  return found ? patched : concatBytes([bytes, encodeStringField(field, value)])
}

function encodeDimension(dim: OnnxDim): Uint8Array {
  return 'value' in dim ? encodeInt64Field(DIM_VALUE, dim.value) : encodeStringField(DIM_PARAM, dim.param)
}

// dims: null omits TypeProto.Tensor.shape entirely (unranked); [] encodes an
// empty-but-present TensorShapeProto (rank 0, a scalar) -- those are different
// things in ONNX and callers must pick deliberately, not default one to the other.
export function encodeValueInfo(name: string, elemType: number, dims: OnnxDim[] | null): Uint8Array {
  const shapeField = dims ? encodeLenField(TENSOR_SHAPE, concatBytes(dims.map((d) => encodeLenField(SHAPE_DIM, encodeDimension(d))))) : new Uint8Array(0)
  const tensorType = concatBytes([encodeVarintField(TENSOR_ELEM_TYPE, elemType), shapeField])
  const typeProto = encodeLenField(TYPE_TENSOR, tensorType)
  return concatBytes([encodeStringField(VINFO_NAME, name), encodeLenField(VINFO_TYPE, typeProto)])
}

// Fixed-width little-endian writers, the encode-side mirror of
// onnxProtoParser.ts's ELEM_TYPE_READERS. FLOAT16/BFLOAT16 are absent there
// (no half-float decode in this codebase) so a small constant of that dtype
// never gets an inspectable `values` array in the first place -- replaceConstant
// can only be reached for a dtype this table also covers.
const ELEM_TYPE_WRITERS: Record<number, (view: DataView, offset: number, value: number) => void> = {
  1: (v, o, x) => v.setFloat32(o, x, true),
  2: (v, o, x) => v.setUint8(o, x),
  3: (v, o, x) => v.setInt8(o, x),
  4: (v, o, x) => v.setUint16(o, x, true),
  5: (v, o, x) => v.setInt16(o, x, true),
  6: (v, o, x) => v.setInt32(o, x, true),
  7: (v, o, x) => v.setBigInt64(o, BigInt(Math.trunc(x)), true),
  9: (v, o, x) => v.setUint8(o, x ? 1 : 0),
  11: (v, o, x) => v.setFloat64(o, x, true),
  12: (v, o, x) => v.setUint32(o, x, true),
  13: (v, o, x) => v.setBigUint64(o, BigInt(Math.max(0, Math.trunc(x))), true),
}

function encodeRawData(elemType: number, values: number[]): Uint8Array {
  const writer = ELEM_TYPE_WRITERS[elemType]
  const bytesPerElem = DATA_TYPE_BYTES[elemType]
  if (!writer || !bytesPerElem) throw new Error(`Unsupported constant data type: ${elemType}`)
  const buf = new ArrayBuffer(values.length * bytesPerElem)
  const view = new DataView(buf)
  values.forEach((value, i) => writer(view, i * bytesPerElem, value))
  return new Uint8Array(buf)
}

function setRawData(bytes: Uint8Array, rawData: Uint8Array): Uint8Array {
  let found = false
  const patched = patchLenFields(bytes, INIT_RAW_DATA, () => { found = true; return rawData })
  return found ? patched : concatBytes([bytes, encodeLenField(INIT_RAW_DATA, rawData)])
}

// Delete and insertPassthrough never change relative node order (delete only
// removes; insert always splices its new node immediately before its consumer),
// so they can't break ONNX's topological-order requirement. Rewire is different:
// it can point a node's input at ANY other original node's output, including one
// that currently sits later in the list, which the raw entries array would then
// serialize out of order. DFS postorder over the producer graph fixes that --
// cycles can't occur here since validateRewire (graphUtils.ts) already rejects
// them before an op ever reaches the writer, so this never needs cycle recovery.
function topologicalSort(entries: NodeEntry[]): NodeEntry[] {
  const producerOf = new Map<string, NodeEntry>()
  for (const e of entries) {
    for (const out of e.outputs) if (out) producerOf.set(out, e)
  }
  const visited = new Set<NodeEntry>()
  const result: NodeEntry[] = []
  function visit(e: NodeEntry) {
    if (visited.has(e)) return
    visited.add(e)
    for (const inp of e.inputs) {
      const producer = inp ? producerOf.get(inp) : undefined
      if (producer) visit(producer)
    }
    result.push(e)
  }
  for (const e of entries) visit(e)
  return result
}

// insertPassthrough's generated Identity entries use a negative sentinel origIndex
// purely to stay unique within one export call (they're never a valid op target,
// so nothing outside this function ever needs to address them). Offset well clear
// of addNode's negative range (structuralNodeIndex negates the UI's small custom-node
// counter, e.g. -1, -2, ...) so the two generated-entry schemes can never collide
// within the same export.
const PASSTHROUGH_SENTINEL_BASE = -1_000_000

interface NamedEntry { name: string; bytes: Uint8Array }

// GraphProto's node/initializer/input/output lists, decoded into named or
// indexed entries so v2.3's deployment ops (rename, retype, promote, replace)
// can address and rewrite any of them -- not just nodes, which is all the
// pre-v2.3 structural ops (delete/insertPassthrough/rewire/addNode) ever
// needed. Everything else in GraphProto (name, doc_string, value_info,
// sparse_initializer, ...) stays in otherChunks, untouched and unindexed --
// none of the ops below target it, and value_info in particular is informational
// only, so a rename leaving it stale under the old name is a cosmetic gap, not
// a correctness one.
interface DecodedGraph {
  nodes: NodeEntry[]
  initializers: NamedEntry[]
  graphInputs: NamedEntry[]
  graphOutputs: NamedEntry[]
  // Read-only: never itself rewritten, only consulted for promoteOutput's
  // knownValueInfo lookup. Its raw bytes are also always present in
  // otherChunks, which is what actually preserves it in the output.
  valueInfo: NamedEntry[]
  otherChunks: Uint8Array[]
}

function applyStructuralOps(decoded: DecodedGraph, structuralOps: StructuralOp[]): DecodedGraph {
  let entries = decoded.nodes
  let initializers = decoded.initializers
  let graphInputs = decoded.graphInputs
  let graphOutputs = decoded.graphOutputs
  let insertCounter = 0
  let needsTopoSort = false

  // Static snapshot from decode time (see decodeGraphContent) -- promoteOutput's
  // best-effort reuse of a tensor's existing declared type, not re-derived after
  // every op, so it can miss a same-batch rename's new name and fall back to an
  // unranked declaration. A valid result either way, just possibly less precise.
  const knownValueInfo = new Map<string, Uint8Array>()
  for (const entry of [...decoded.graphInputs, ...decoded.graphOutputs, ...decoded.valueInfo]) knownValueInfo.set(entry.name, entry.bytes)

  for (const op of structuralOps) {
    if (op.type === 'renameNode') {
      entries = entries.map((e) => (e.origIndex === op.nodeIndex ? decodeEntry(e.origIndex, setStringField(e.bytes, NODE_NAME, op.name)) : e))
    } else if (op.type === 'renameTensor') {
      const { oldName, newName } = op
      entries = entries.map((e) => decodeEntry(e.origIndex, renameOutputsByValue(rewireInputsByValue(e.bytes, oldName, newName), oldName, newName)))
      initializers = initializers.map((i) => (i.name === oldName ? { name: newName, bytes: setStringField(i.bytes, INIT_NAME, newName) } : i))
      graphInputs = graphInputs.map((i) => (i.name === oldName ? { name: newName, bytes: setStringField(i.bytes, VINFO_NAME, newName) } : i))
      graphOutputs = graphOutputs.map((o) => (o.name === oldName ? { name: newName, bytes: setStringField(o.bytes, VINFO_NAME, newName) } : o))
    } else if (op.type === 'setGraphIO') {
      const pool = op.ioKind === 'input' ? graphInputs : graphOutputs
      const entry = pool[op.ioIndex]
      if (!entry) continue
      const next = { name: entry.name, bytes: encodeValueInfo(entry.name, op.elemType, op.dims) }
      if (op.ioKind === 'input') graphInputs = graphInputs.map((e, i) => (i === op.ioIndex ? next : e))
      else graphOutputs = graphOutputs.map((e, i) => (i === op.ioIndex ? next : e))
    } else if (op.type === 'promoteOutput') {
      if (graphOutputs.some((o) => o.name === op.tensorName)) continue // already an output
      const bytes = knownValueInfo.get(op.tensorName) ?? encodeValueInfo(op.tensorName, 1, null)
      graphOutputs = [...graphOutputs, { name: op.tensorName, bytes }]
    } else if (op.type === 'replaceConstant') {
      const idx = initializers.findIndex((i) => i.name === op.initializerName)
      if (idx === -1) continue
      const entry = initializers[idx]
      const parsed = parseInitializer(new ProtoReader(entry.bytes))
      if (op.values.length !== parsed.elemCount) continue
      const rawData = encodeRawData(parsed.elemType, op.values)
      initializers = initializers.map((i, i2) => (i2 === idx ? { name: i.name, bytes: setRawData(entry.bytes, rawData) } : i))
    } else if (op.type === 'delete') {
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
      // Sentinel negative origIndex (see PASSTHROUGH_SENTINEL_BASE above): inserted
      // entries are never themselves a valid op target -- chaining onto a generated
      // passthrough is out of scope by construction.
      const identityEntry = decodeEntry(PASSTHROUGH_SENTINEL_BASE - insertCounter, buildIdentityNodeBytes(tensorName, newTensorName, op.newNodeName))

      // Splice the new node immediately before its consumer, not appended at the
      // end: ONNX requires GraphProto.node to be topologically ordered (a node's
      // inputs must be defined by an earlier node), and the target already follows
      // its own producer, so inserting directly before it preserves that ordering.
      entries = [...entries.slice(0, idx), identityEntry, rewiredTarget, ...entries.slice(idx + 1)]
    } else if (op.type === 'rewire') {
      const sourceEntry = entries.find((e) => e.origIndex === op.sourceNodeIndex)
      const newTensorName = sourceEntry?.outputs[0]
      if (!newTensorName) continue
      entries = entries.map((e) =>
        e.origIndex === op.targetNodeIndex ? decodeEntry(e.origIndex, rewireInputAtPosition(e.bytes, op.inputPosition, newTensorName)) : e,
      )
      needsTopoSort = true
    } else if (op.type === 'addNode') {
      const origIndex = -op.newNodeIndex
      const outputTensor = `custom_${op.newNodeIndex}_out`
      const name = `custom_${op.newNodeIndex}`
      entries = [...entries, decodeEntry(origIndex, buildCustomNodeBytes(op.opType, op.inputCount, outputTensor, name))]
      // A freshly appended node needs no reordering by itself (nothing yet depends
      // on it, it depends on nothing), but a later rewire in the same batch can make
      // an earlier-positioned node consume its output, which does need the sort.
      needsTopoSort = true
    } else if (op.type === 'insertRecipe') {
      const origIndex = -op.newNodeIndex
      const name = `recipe_${op.newNodeIndex}`
      const attrBytes = op.attrs.map((a) => encodeLenField(NODE_ATTR, buildEditedAttrContent(a.name, a.kind, a.value)))
      const extraInputNames = op.extraInputs.map((slot, i) => {
        if (slot.kind === 'empty') return ''
        const constName = `${name}_const${i}`
        initializers = [...initializers, { name: constName, bytes: buildInitializerBytes(constName, slot.elemType ?? 1, slot.dims ?? [], slot.values ?? []) }]
        return constName
      })

      if (op.anchorKind === 'input') {
        const tensorName = op.anchorTensor
        if (!tensorName) continue
        const newTensorName = `${name}_out`
        entries = entries.map((e) => (
          e.inputs.includes(tensorName) ? decodeEntry(e.origIndex, rewireInputsByValue(e.bytes, tensorName, newTensorName)) : e
        ))
        const newEntry = decodeEntry(origIndex, buildRecipeNodeBytes(op.opType, [tensorName, ...extraInputNames], [newTensorName], name, attrBytes))
        // Splice immediately after whatever currently produces tensorName (a
        // prior chained recipe), or prepend if nothing does (a raw graph
        // input has no producer entry at all) -- same ordering rationale as
        // insertPassthrough's "splice before its consumer", mirrored for a
        // node spliced after its producer instead.
        const producerIdx = entries.findIndex((e) => e.outputs.includes(tensorName))
        entries = producerIdx === -1
          ? [newEntry, ...entries]
          : [...entries.slice(0, producerIdx + 1), newEntry, ...entries.slice(producerIdx + 1)]
      } else {
        const anchor = graphOutputs[op.ioIndex]
        if (!anchor) continue
        const publicName = anchor.name
        const internalName = `${name}_in`
        entries = entries.map((e) => decodeEntry(e.origIndex, renameOutputsByValue(rewireInputsByValue(e.bytes, publicName, internalName), publicName, internalName)))
        const extraOutputNames = Array.from({ length: op.extraOutputCount }, (_, i) => `${name}_extra${i}`)
        const newEntry = decodeEntry(origIndex, buildRecipeNodeBytes(op.opType, [internalName, ...extraInputNames], [publicName, ...extraOutputNames], name, attrBytes))
        entries = [...entries, newEntry] // its only input is the just-renamed producer, always already earlier
        // The op that used to declare publicName's shape (e.g. [N, 1000]) is
        // gone from this position -- a recipe like Top-K genuinely produces a
        // different shape (top-K along an axis), so the stale declaration
        // would conflict with onnxruntime's own shape inference and fail
        // session creation. Re-declare unranked (dtype preserved, shape
        // dropped) rather than leave a declaration nothing still backs.
        const knownElemType = parseValueInfo(new ProtoReader(anchor.bytes)).elemType ?? 1
        graphOutputs = graphOutputs.map((o, i) => (i === op.ioIndex ? { name: publicName, bytes: encodeValueInfo(publicName, knownElemType, null) } : o))
        if (extraOutputNames.length > 0) {
          graphOutputs = [...graphOutputs, ...extraOutputNames.map((n, i) => ({ name: n, bytes: encodeValueInfo(n, op.extraOutputElemTypes[i] ?? 1, null) }))]
        }
      }
      needsTopoSort = true
    }
  }

  return {
    nodes: needsTopoSort ? topologicalSort(entries) : entries,
    initializers,
    graphInputs,
    graphOutputs,
    valueInfo: decoded.valueInfo,
    otherChunks: decoded.otherChunks,
  }
}

// Decodes graphContent into named/indexed entry pools (applying attribute
// overrides to nodes along the way) plus every other field's bytes untouched,
// applies structuralOps, then re-encodes. Node entries are hoisted before
// other fields in the output, which is safe -- protobuf field order across
// distinct field numbers is not meaningful, only the relative order AMONG
// node entries (topological) matters, and that's preserved by construction
// in applyStructuralOps.
function rewriteGraphContent(
  graphContent: Uint8Array,
  overridesByNodeIndex: Map<number, Record<string, string | number>>,
  structuralOps: StructuralOp[],
): Uint8Array {
  const r = new ProtoReader(graphContent)
  const otherChunks: Uint8Array[] = []
  const nodes: NodeEntry[] = []
  const initializers: NamedEntry[] = []
  const graphInputs: NamedEntry[] = []
  const graphOutputs: NamedEntry[] = []
  const valueInfo: NamedEntry[] = []
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
      const sub = graphContent.subarray(subStart, fieldEnd)
      if (tag.field === GRAPH_NODE) {
        nodes.push(decodeEntry(nodeOccurrence, sub))
        nodeOccurrence++
      } else if (tag.field === GRAPH_INIT) {
        initializers.push({ name: readTopLevelStringField(sub, INIT_NAME), bytes: sub })
      } else if (tag.field === GRAPH_INPUT) {
        graphInputs.push({ name: readTopLevelStringField(sub, VINFO_NAME), bytes: sub })
      } else if (tag.field === GRAPH_OUTPUT) {
        graphOutputs.push({ name: readTopLevelStringField(sub, VINFO_NAME), bytes: sub })
      } else if (tag.field === GRAPH_VALUE_INFO) {
        // Captured for knownValueInfo (see DecodedGraph) AND still pushed to
        // otherChunks below -- it's read for lookups but never itself rewritten.
        valueInfo.push({ name: readTopLevelStringField(sub, VINFO_NAME), bytes: sub })
        otherChunks.push(graphContent.subarray(fieldStart, fieldEnd))
      } else {
        otherChunks.push(graphContent.subarray(fieldStart, fieldEnd))
      }
    } else {
      r.skipField(tag.wire)
      otherChunks.push(graphContent.subarray(fieldStart, r.pos))
    }
  }

  const result = applyStructuralOps({ nodes, initializers, graphInputs, graphOutputs, valueInfo, otherChunks }, structuralOps)
  // Attribute overrides are applied to the FINAL node set, keyed by origIndex
  // rather than during the initial GRAPH_NODE pass above -- origIndex is
  // stable across structural edits (delete/rewire/rename never renumber
  // survivors), and applying it here, after entries created by addNode or
  // insertRecipe already exist, is what makes a custom or recipe node's
  // attributes editable at export time at all, not just on the live canvas.
  const patchedNodes = result.nodes.map((entry) => {
    const overrides = overridesByNodeIndex.get(entry.origIndex)
    if (!overrides) return entry
    return decodeEntry(entry.origIndex, patchLenFields(entry.bytes, NODE_ATTR, (_attrIdx, attrContent) => {
      const { name, kind } = identifyAttr(attrContent)
      if (kind === 'OTHER' || !(name in overrides)) return null
      return buildEditedAttrContent(name, kind, overrides[name])
    }))
  })
  return concatBytes([
    ...patchedNodes.map((e) => encodeLenField(GRAPH_NODE, e.bytes)),
    ...result.initializers.map((i) => encodeLenField(GRAPH_INIT, i.bytes)),
    ...result.graphInputs.map((i) => encodeLenField(GRAPH_INPUT, i.bytes)),
    ...result.graphOutputs.map((o) => encodeLenField(GRAPH_OUTPUT, o.bytes)),
    ...result.otherChunks,
  ])
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
