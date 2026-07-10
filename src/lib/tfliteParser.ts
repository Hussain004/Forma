// Minimal binary FlatBuffers parser scoped to the subset of the TFLite schema needed
// for read-only graph visualization: topology, tensor shapes/types, op names, weight
// sizes. Schema field orderings verified against the authoritative schema.fbs from
// google-ai-edge/LiteRT (the project TFLite moved to) -- FlatBuffers field IDs are
// assigned by declaration order, so getting a table's field order wrong silently
// misreads data rather than throwing, unlike most parsing bugs.
//
// FlatBuffers wire format (schema-independent):
// - All integers little-endian. No alignment requirement for a reader (the writer pads
//   to guarantee it; DataView reads unaligned offsets fine in JS).
// - Table: at position P, an int32 "soffset" at P points BACKWARD to its vtable:
//   vtablePos = P - soffset. Vtable: [vtable_size:u16, table_size:u16, N x field_offset:u16].
//   A field offset of 0 (or reading past the vtable's own length) means "absent, use
//   default" -- this is how forward/backward schema compatibility works. Otherwise the
//   field's value lives at P + offset.
// - Offset field (string/vector/table): a uint32 "uoffset" stored inline, FORWARD-
//   relative to its own slot: target = slotPos + uoffsetValue.
// - String: at target, length:u32 (excludes null terminator) then that many UTF-8 bytes.
// - Vector: at target, count:u32 then count elements. Scalar vectors pack inline;
//   vectors of offsets (tables/strings) store one uoffset per slot, each forward-
//   relative to ITS OWN slot position (not the vector start).

import { buildGraphFromParsed } from './onnxParser'
import type { ParsedGraph, ParsedNode, ParsedInitializer, ParsedValueInfo } from './onnxProtoParser'
import type { OnnxGraph, OnnxDim } from './onnxTypes'

export function isTfliteBuffer(buffer: ArrayBuffer): boolean {
  if (buffer.byteLength < 8) return false
  const bytes = new Uint8Array(buffer, 4, 4)
  return String.fromCharCode(...bytes) === 'TFL3'
}

// ---- Low-level reader ----

class FlatBufferReader {
  view: DataView
  buf: Uint8Array
  constructor(buffer: ArrayBuffer) {
    this.buf = new Uint8Array(buffer)
    this.view = new DataView(buffer)
  }

  rootTable(): number {
    return this.view.getUint32(0, true)
  }

  private vtable(tablePos: number): number {
    const soffset = this.view.getInt32(tablePos, true)
    return tablePos - soffset
  }

  // Absolute position of fieldId's value within tablePos, or null if the field is
  // absent (not present in this table's vtable, or explicitly omitted as a default).
  private fieldPos(tablePos: number, fieldId: number): number | null {
    const vt = this.vtable(tablePos)
    const vtableSize = this.view.getUint16(vt, true)
    const slot = 4 + fieldId * 2
    if (slot >= vtableSize) return null
    const relOffset = this.view.getUint16(vt + slot, true)
    return relOffset === 0 ? null : tablePos + relOffset
  }

  getInt8(tablePos: number, fieldId: number, def = 0): number {
    const p = this.fieldPos(tablePos, fieldId)
    return p === null ? def : this.view.getInt8(p)
  }

  getUint32(tablePos: number, fieldId: number, def = 0): number {
    const p = this.fieldPos(tablePos, fieldId)
    return p === null ? def : this.view.getUint32(p, true)
  }

  getInt32(tablePos: number, fieldId: number, def = 0): number {
    const p = this.fieldPos(tablePos, fieldId)
    return p === null ? def : this.view.getInt32(p, true)
  }

  private offsetTarget(tablePos: number, fieldId: number): number | null {
    const p = this.fieldPos(tablePos, fieldId)
    if (p === null) return null
    return p + this.view.getUint32(p, true)
  }

  getString(tablePos: number, fieldId: number): string {
    const target = this.offsetTarget(tablePos, fieldId)
    if (target === null) return ''
    const len = this.view.getUint32(target, true)
    return new TextDecoder().decode(this.buf.subarray(target + 4, target + 4 + len))
  }

  // Vector of scalars, e.g. [int]/[ubyte]. Returns the position of element 0 and the
  // element count; caller reads elements at the appropriate width.
  getVector(tablePos: number, fieldId: number): { pos: number; count: number } | null {
    const target = this.offsetTarget(tablePos, fieldId)
    if (target === null) return null
    return { pos: target + 4, count: this.view.getUint32(target, true) }
  }

  readInt32At(pos: number, i: number): number {
    return this.view.getInt32(pos + i * 4, true)
  }

  // Vector of table offsets, e.g. [OperatorCode]/[Tensor]/[Operator]/[Buffer]. Each
  // 4-byte slot holds a uoffset forward-relative to its own position.
  getVectorTablePos(vec: { pos: number; count: number }, i: number): number {
    const slot = vec.pos + i * 4
    return slot + this.view.getUint32(slot, true)
  }
}

function getTableVectorPositions(r: FlatBufferReader, tablePos: number, fieldId: number): number[] {
  const vec = r.getVector(tablePos, fieldId)
  if (!vec) return []
  const positions: number[] = []
  for (let i = 0; i < vec.count; i++) positions.push(r.getVectorTablePos(vec, i))
  return positions
}

function decodeIntVector(r: FlatBufferReader, tablePos: number, fieldId: number): number[] {
  const vec = r.getVector(tablePos, fieldId)
  if (!vec) return []
  const out: number[] = []
  for (let i = 0; i < vec.count; i++) out.push(r.readInt32At(vec.pos, i))
  return out
}

// ---- TFLite BuiltinOperator enum ----
// Verified against google-ai-edge/LiteRT tflite/converter/schema/schema.fbs, codes
// 0-161 (every practical/production op). Codes 162+ are experimental STABLEHLO_* ops
// with little to no runtime support in real-world models, left unmapped -- an
// unrecognized code falls back to `OP_<code>` rather than throwing.
const BUILTIN_OPERATOR_NAMES: string[] = [
  'ADD', 'AVERAGE_POOL_2D', 'CONCATENATION', 'CONV_2D', 'DEPTHWISE_CONV_2D', 'DEPTH_TO_SPACE',
  'DEQUANTIZE', 'EMBEDDING_LOOKUP', 'FLOOR', 'FULLY_CONNECTED', 'HASHTABLE_LOOKUP',
  'L2_NORMALIZATION', 'L2_POOL_2D', 'LOCAL_RESPONSE_NORMALIZATION', 'LOGISTIC', 'LSH_PROJECTION',
  'LSTM', 'MAX_POOL_2D', 'MUL', 'RELU', 'RELU_N1_TO_1', 'RELU6', 'RESHAPE', 'RESIZE_BILINEAR',
  'RNN', 'SOFTMAX', 'SPACE_TO_DEPTH', 'SVDF', 'TANH', 'CONCAT_EMBEDDINGS', 'SKIP_GRAM', 'CALL',
  'CUSTOM', 'EMBEDDING_LOOKUP_SPARSE', 'PAD', 'UNIDIRECTIONAL_SEQUENCE_RNN', 'GATHER',
  'BATCH_TO_SPACE_ND', 'SPACE_TO_BATCH_ND', 'TRANSPOSE', 'MEAN', 'SUB', 'DIV', 'SQUEEZE',
  'UNIDIRECTIONAL_SEQUENCE_LSTM', 'STRIDED_SLICE', 'BIDIRECTIONAL_SEQUENCE_RNN', 'EXP', 'TOPK_V2',
  'SPLIT', 'LOG_SOFTMAX', 'DELEGATE', 'BIDIRECTIONAL_SEQUENCE_LSTM', 'CAST', 'PRELU', 'MAXIMUM',
  'ARG_MAX', 'MINIMUM', 'LESS', 'NEG', 'PADV2', 'GREATER', 'GREATER_EQUAL', 'LESS_EQUAL', 'SELECT',
  'SLICE', 'SIN', 'TRANSPOSE_CONV', 'SPARSE_TO_DENSE', 'TILE', 'EXPAND_DIMS', 'EQUAL', 'NOT_EQUAL',
  'LOG', 'SUM', 'SQRT', 'RSQRT', 'SHAPE', 'POW', 'ARG_MIN', 'FAKE_QUANT', 'REDUCE_PROD',
  'REDUCE_MAX', 'PACK', 'LOGICAL_OR', 'ONE_HOT', 'LOGICAL_AND', 'LOGICAL_NOT', 'UNPACK',
  'REDUCE_MIN', 'FLOOR_DIV', 'REDUCE_ANY', 'SQUARE', 'ZEROS_LIKE', 'FILL', 'FLOOR_MOD', 'RANGE',
  'RESIZE_NEAREST_NEIGHBOR', 'LEAKY_RELU', 'SQUARED_DIFFERENCE', 'MIRROR_PAD', 'ABS', 'SPLIT_V',
  'UNIQUE', 'CEIL', 'REVERSE_V2', 'ADD_N', 'GATHER_ND', 'COS', 'WHERE', 'RANK', 'ELU',
  'REVERSE_SEQUENCE', 'MATRIX_DIAG', 'QUANTIZE', 'MATRIX_SET_DIAG', 'ROUND', 'HARD_SWISH', 'IF',
  'WHILE', 'NON_MAX_SUPPRESSION_V4', 'NON_MAX_SUPPRESSION_V5', 'SCATTER_ND', 'SELECT_V2',
  'DENSIFY', 'SEGMENT_SUM', 'BATCH_MATMUL', 'PLACEHOLDER_FOR_GREATER_OP_CODES', 'CUMSUM',
  'CALL_ONCE', 'BROADCAST_TO', 'RFFT2D', 'CONV_3D', 'IMAG', 'REAL', 'COMPLEX_ABS', 'HASHTABLE',
  'HASHTABLE_FIND', 'HASHTABLE_IMPORT', 'HASHTABLE_SIZE', 'REDUCE_ALL', 'CONV_3D_TRANSPOSE',
  'VAR_HANDLE', 'READ_VARIABLE', 'ASSIGN_VARIABLE', 'BROADCAST_ARGS', 'RANDOM_STANDARD_NORMAL',
  'BUCKETIZE', 'RANDOM_UNIFORM', 'MULTINOMIAL', 'GELU', 'DYNAMIC_UPDATE_SLICE', 'RELU_0_TO_1',
  'UNSORTED_SEGMENT_PROD', 'UNSORTED_SEGMENT_MAX', 'UNSORTED_SEGMENT_SUM', 'ATAN2',
  'UNSORTED_SEGMENT_MIN', 'SIGN', 'BITCAST', 'BITWISE_XOR', 'RIGHT_SHIFT',
]
const CUSTOM_OP_CODE = 32

function builtinOpName(code: number): string {
  return BUILTIN_OPERATOR_NAMES[code] ?? `OP_${code}`
}

// TFLite TensorType -> ONNX TensorProto.DataType equivalent, indexed by TFLite code.
// The two enums are NOT numerically aligned (e.g. TFLite FLOAT32=0 vs ONNX FLOAT=1), so
// this is an explicit translation, not a pass-through -- kept for consistency with the
// rest of the app, which expects ONNX-style elemType codes wherever one is surfaced.
// 0 = no ONNX equivalent (TFLite's RESOURCE/VARIANT types).
const TFLITE_TO_ONNX_ELEM_TYPE: number[] = [
  1, 10, 6, 2, 7, 8, 9, 5, 14, 3, 11, 15, 13, 0, 0, 12, 4,
]

// ---- TFLite schema decoding ----
// Field IDs below match schema.fbs declaration order exactly (see file header).

function decodeOperatorCode(r: FlatBufferReader, pos: number): { name: string } {
  const deprecatedCode = r.getInt8(pos, 0, 0)
  const customCode = r.getString(pos, 1)
  const builtinCode = r.getInt32(pos, 3, 0)
  // builtin_code (the newer int32 field) takes precedence when set; older files only
  // ever wrote deprecated_builtin_code, leaving builtin_code absent (reads as 0).
  const code = builtinCode !== 0 ? builtinCode : deprecatedCode
  if (code === CUSTOM_OP_CODE) return { name: customCode || 'CUSTOM' }
  return { name: builtinOpName(code) }
}

interface TfliteTensor {
  name: string
  shape: OnnxDim[]
  elemType: number
  bufferIndex: number
}

function decodeTensor(r: FlatBufferReader, pos: number, idx: number): TfliteTensor {
  const shapeVec = r.getVector(pos, 0)
  const shape: OnnxDim[] = []
  if (shapeVec) {
    for (let i = 0; i < shapeVec.count; i++) shape.push({ value: r.readInt32At(shapeVec.pos, i) })
  }
  const tfType = r.getInt8(pos, 1, 0)
  const bufferIndex = r.getUint32(pos, 2, 0)
  const rawName = r.getString(pos, 3)
  return {
    name: rawName || `tensor_${idx}`,
    shape,
    elemType: TFLITE_TO_ONNX_ELEM_TYPE[tfType] ?? 0,
    bufferIndex,
  }
}

interface TfliteOperator {
  opcodeIndex: number
  inputs: number[]
  outputs: number[]
}

function decodeOperator(r: FlatBufferReader, pos: number): TfliteOperator {
  return {
    opcodeIndex: r.getUint32(pos, 0, 0),
    inputs: decodeIntVector(r, pos, 1),
    outputs: decodeIntVector(r, pos, 2),
  }
}

// ---- Public API ----

export function parseTfliteGraph(buffer: ArrayBuffer, modelName: string): OnnxGraph {
  const r = new FlatBufferReader(buffer)
  const modelPos = r.rootTable()

  const opCodeNames = getTableVectorPositions(r, modelPos, 1).map((pos) => decodeOperatorCode(r, pos).name)

  // Buffers (Model field 4): only the byte length of each buffer's data vector is
  // needed for weight-size estimation, never the raw bytes themselves.
  const bufferByteLengths = getTableVectorPositions(r, modelPos, 4).map((pos) => {
    const vec = r.getVector(pos, 0)
    return vec ? vec.count : 0
  })

  const subgraphPositions = getTableVectorPositions(r, modelPos, 2)
  const emptyProto: ParsedGraph = { name: modelName, nodes: [], initializers: [], inputs: [], outputs: [], valueInfo: [] }
  if (subgraphPositions.length === 0) return buildGraphFromParsed(emptyProto, modelName, 'tflite')

  // TFLite models can contain multiple subgraphs for control flow (IF/WHILE branches);
  // subgraph 0 is always the main graph per TFLite convention, and the only one
  // visualized here -- following into branch subgraphs is out of scope.
  const sgPos = subgraphPositions[0]

  const tensors = getTableVectorPositions(r, sgPos, 0).map((pos, idx) => decodeTensor(r, pos, idx))
  const graphInputIndices = decodeIntVector(r, sgPos, 1)
  const graphOutputIndices = decodeIntVector(r, sgPos, 2)
  const operators = getTableVectorPositions(r, sgPos, 3).map((pos) => decodeOperator(r, pos))

  // Operator.inputs/outputs are tensor INDICES (-1 = absent/optional), unlike ONNX's
  // string tensor names. Translating index -> name here (rather than threading indices
  // through buildGraphFromParsed) means the rest of the pipeline -- edge synthesis by
  // shared tensor name, node id scheme, param/size rollups -- is reused completely
  // unmodified. '' represents an absent input, exactly matching how ONNX represents an
  // omitted optional input; the generic edge-wiring logic already treats '' as "no
  // producer, no edge" so no special-casing is needed downstream.
  const parsedNodes: ParsedNode[] = operators.map((op) => ({
    inputs: op.inputs.map((i) => (i >= 0 && tensors[i] ? tensors[i].name : '')),
    outputs: op.outputs.map((i) => (i >= 0 && tensors[i] ? tensors[i].name : '')),
    name: '',
    opType: opCodeNames[op.opcodeIndex] ?? `OP_${op.opcodeIndex}`,
    attributes: {},
  }))

  // A tensor is a constant/weight (initializer) when it references a non-empty buffer
  // other than the reserved index-0 empty sentinel (mandated by the schema itself).
  const initializers: ParsedInitializer[] = []
  for (const t of tensors) {
    if (t.bufferIndex <= 0) continue
    const byteLength = bufferByteLengths[t.bufferIndex] ?? 0
    if (byteLength === 0) continue
    const elemCount = t.shape.reduce((acc, d) => acc * ('value' in d ? d.value : 1), 1)
    initializers.push({
      name: t.name,
      dims: t.shape.map((d) => ('value' in d ? d.value : 0)),
      elemType: t.elemType,
      elemCount,
      // TFLite buffers store raw bytes directly, so the exact size is already known --
      // no need to multiply elemCount by a bytes-per-type table as the ONNX path does.
      sizeMB: byteLength / (1024 * 1024),
    })
  }

  const valueInfo: ParsedValueInfo[] = tensors.map((t) => ({ name: t.name, shape: t.shape, elemType: t.elemType }))
  const graphInputs = graphInputIndices.map((i) => valueInfo[i]).filter((v): v is ParsedValueInfo => Boolean(v))
  const graphOutputs = graphOutputIndices.map((i) => valueInfo[i]).filter((v): v is ParsedValueInfo => Boolean(v))

  const proto: ParsedGraph = {
    name: modelName,
    nodes: parsedNodes,
    initializers,
    inputs: graphInputs,
    outputs: graphOutputs,
    valueInfo,
  }

  return buildGraphFromParsed(proto, modelName, 'tflite')
}
