// Minimal binary protobuf parser scoped to ONNX graph topology.
// Decodes the raw ArrayBuffer without going through onnxruntime-web internals,
// which are private and not accessible in all runtime configurations.
//
// ONNX uses proto3. Field numbers referenced below come from onnx.proto3.

const WIRE_VARINT = 0
const WIRE_64BIT = 1
const WIRE_LEN = 2
const WIRE_32BIT = 5

// ModelProto fields
const MODEL_GRAPH = 7

// GraphProto fields
const GRAPH_NODE = 1
const GRAPH_NAME = 2
const GRAPH_INIT = 5
const GRAPH_INPUT = 11
const GRAPH_OUTPUT = 12

// NodeProto fields
const NODE_INPUT = 1
const NODE_OUTPUT = 2
const NODE_NAME = 3
const NODE_OP_TYPE = 4

// TensorProto fields (initializers)
const INIT_DIMS = 1
const INIT_DATA_TYPE = 2
const INIT_NAME = 8

// ValueInfoProto fields
const VINFO_NAME = 1
const VINFO_TYPE = 2

// TypeProto fields
const TYPE_TENSOR = 1

// TypeProto.Tensor fields
const TENSOR_ELEM_TYPE = 1
const TENSOR_SHAPE = 2

// TensorShapeProto fields
const SHAPE_DIM = 1

// Dimension fields
const DIM_VALUE = 1
const DIM_PARAM = 2

// Bytes per element for each ONNX data type
const DATA_TYPE_BYTES: Record<number, number> = {
  1: 4,  // FLOAT
  2: 1,  // UINT8
  3: 1,  // INT8
  4: 2,  // UINT16
  5: 2,  // INT16
  6: 4,  // INT32
  7: 8,  // INT64
  9: 1,  // BOOL
  10: 2, // FLOAT16
  11: 8, // DOUBLE
  12: 4, // UINT32
  13: 8, // UINT64
  16: 2, // BFLOAT16
}

export type OnnxDim = { value: number } | { param: string }

export interface ParsedValueInfo {
  name: string
  shape?: OnnxDim[]
  elemType?: number
}

export interface ParsedInitializer {
  name: string
  dims: number[]
  elemType: number
  elemCount: number
  sizeMB: number
}

export interface ParsedNode {
  inputs: string[]
  outputs: string[]
  name: string
  opType: string
}

export interface ParsedGraph {
  name: string
  nodes: ParsedNode[]
  initializers: ParsedInitializer[]
  inputs: ParsedValueInfo[]
  outputs: ParsedValueInfo[]
}

// ---- Low-level reader ----

class ProtoReader {
  pos = 0
  buf: Uint8Array
  constructor(buf: Uint8Array) { this.buf = buf }

  get done() { return this.pos >= this.buf.length }

  readVarint(): number {
    let r = 0, shift = 0
    while (this.pos < this.buf.length) {
      const b = this.buf[this.pos++]
      r |= (b & 0x7f) << shift
      shift += 7
      if (!(b & 0x80)) break
    }
    return r >>> 0
  }

  skipVarint() {
    while (this.pos < this.buf.length) {
      if (!(this.buf[this.pos++] & 0x80)) break
    }
  }

  readString(len: number): string {
    const bytes = this.buf.subarray(this.pos, this.pos + len)
    this.pos += len
    return new TextDecoder().decode(bytes)
  }

  subReader(len: number): ProtoReader {
    const r = new ProtoReader(this.buf.subarray(this.pos, this.pos + len))
    this.pos += len
    return r
  }

  skip(len: number) { this.pos += len }

  readTag(): { field: number; wire: number } | null {
    if (this.done) return null
    const tag = this.readVarint()
    return { field: tag >>> 3, wire: tag & 0x7 }
  }

  skipField(wire: number, len?: number) {
    if (wire === WIRE_VARINT) { this.skipVarint() }
    else if (wire === WIRE_64BIT) { this.skip(8) }
    else if (wire === WIRE_32BIT) { this.skip(4) }
    else if (wire === WIRE_LEN) {
      const n = len ?? this.readVarint()
      this.skip(n)
    }
  }
}

// ---- ONNX-specific decoders ----

function parseDimension(r: ProtoReader): OnnxDim {
  let dimValue = 0
  let dimParam = ''
  while (!r.done) {
    const tag = r.readTag()
    if (!tag) break
    if (tag.wire === WIRE_LEN) {
      const len = r.readVarint()
      if (tag.field === DIM_PARAM) {
        dimParam = r.readString(len)
      } else {
        r.skip(len)
      }
    } else if (tag.wire === WIRE_VARINT) {
      const v = r.readVarint()
      if (tag.field === DIM_VALUE) dimValue = v
    } else {
      r.skipField(tag.wire)
    }
  }
  return dimParam ? { param: dimParam } : { value: dimValue }
}

function parseTensorShape(r: ProtoReader): OnnxDim[] {
  const dims: OnnxDim[] = []
  while (!r.done) {
    const tag = r.readTag()
    if (!tag) break
    if (tag.wire === WIRE_LEN) {
      const len = r.readVarint()
      if (tag.field === SHAPE_DIM) {
        dims.push(parseDimension(r.subReader(len)))
      } else {
        r.skip(len)
      }
    } else {
      r.skipField(tag.wire)
    }
  }
  return dims
}

function parseTensorType(r: ProtoReader): { elemType: number; shape?: OnnxDim[] } {
  let elemType = 0
  let shape: OnnxDim[] | undefined
  while (!r.done) {
    const tag = r.readTag()
    if (!tag) break
    if (tag.wire === WIRE_VARINT) {
      const v = r.readVarint()
      if (tag.field === TENSOR_ELEM_TYPE) elemType = v
    } else if (tag.wire === WIRE_LEN) {
      const len = r.readVarint()
      if (tag.field === TENSOR_SHAPE) {
        shape = parseTensorShape(r.subReader(len))
      } else {
        r.skip(len)
      }
    } else {
      r.skipField(tag.wire)
    }
  }
  return { elemType, shape }
}

function parseTypeProto(r: ProtoReader): { elemType: number; shape?: OnnxDim[] } | null {
  while (!r.done) {
    const tag = r.readTag()
    if (!tag) break
    if (tag.wire === WIRE_LEN) {
      const len = r.readVarint()
      if (tag.field === TYPE_TENSOR) {
        return parseTensorType(r.subReader(len))
      } else {
        r.skip(len)
      }
    } else {
      r.skipField(tag.wire)
    }
  }
  return null
}

function parseValueInfo(r: ProtoReader): ParsedValueInfo {
  let name = ''
  let shape: OnnxDim[] | undefined
  let elemType: number | undefined
  while (!r.done) {
    const tag = r.readTag()
    if (!tag) break
    if (tag.wire === WIRE_LEN) {
      const len = r.readVarint()
      if (tag.field === VINFO_NAME) {
        name = r.readString(len)
      } else if (tag.field === VINFO_TYPE) {
        const t = parseTypeProto(r.subReader(len))
        if (t) { shape = t.shape; elemType = t.elemType }
      } else {
        r.skip(len)
      }
    } else {
      r.skipField(tag.wire)
    }
  }
  return { name, shape, elemType }
}

function parseInitializer(r: ProtoReader): ParsedInitializer {
  let name = ''
  let elemType = 1
  const dims: number[] = []

  while (!r.done) {
    const tag = r.readTag()
    if (!tag) break

    if (tag.wire === WIRE_LEN) {
      const len = r.readVarint()
      if (tag.field === INIT_NAME) {
        name = r.readString(len)
      } else if (tag.field === INIT_DIMS) {
        // Packed int64: a sequence of varints
        const sub = r.subReader(len)
        while (!sub.done) dims.push(sub.readVarint())
      } else {
        r.skip(len)
      }
    } else if (tag.wire === WIRE_VARINT) {
      const v = r.readVarint()
      if (tag.field === INIT_DATA_TYPE) elemType = v
      else if (tag.field === INIT_DIMS) dims.push(v) // unpacked (proto2 compat)
    } else {
      r.skipField(tag.wire)
    }
  }

  const elemCount = dims.length === 0 ? 0 : dims.reduce((a, b) => a * b, 1)
  const bytesPerElem = DATA_TYPE_BYTES[elemType] ?? 4
  const sizeMB = (elemCount * bytesPerElem) / (1024 * 1024)
  return { name, dims, elemType, elemCount, sizeMB }
}

function parseNode(r: ProtoReader): ParsedNode {
  const inputs: string[] = []
  const outputs: string[] = []
  let name = ''
  let opType = 'Unknown'
  while (!r.done) {
    const tag = r.readTag()
    if (!tag) break
    if (tag.wire === WIRE_LEN) {
      const len = r.readVarint()
      if (tag.field === NODE_INPUT) inputs.push(r.readString(len))
      else if (tag.field === NODE_OUTPUT) outputs.push(r.readString(len))
      else if (tag.field === NODE_NAME) name = r.readString(len)
      else if (tag.field === NODE_OP_TYPE) opType = r.readString(len)
      else r.skip(len)
    } else {
      r.skipField(tag.wire)
    }
  }
  return { inputs, outputs, name, opType }
}

function parseGraph(r: ProtoReader): ParsedGraph {
  let graphName = ''
  const nodes: ParsedNode[] = []
  const initializers: ParsedInitializer[] = []
  const inputs: ParsedValueInfo[] = []
  const outputs: ParsedValueInfo[] = []

  while (!r.done) {
    const tag = r.readTag()
    if (!tag) break
    if (tag.wire === WIRE_LEN) {
      const len = r.readVarint()
      if (tag.field === GRAPH_NODE) nodes.push(parseNode(r.subReader(len)))
      else if (tag.field === GRAPH_NAME) graphName = r.readString(len)
      else if (tag.field === GRAPH_INIT) initializers.push(parseInitializer(r.subReader(len)))
      else if (tag.field === GRAPH_INPUT) inputs.push(parseValueInfo(r.subReader(len)))
      else if (tag.field === GRAPH_OUTPUT) outputs.push(parseValueInfo(r.subReader(len)))
      else r.skip(len)
    } else {
      r.skipField(tag.wire)
    }
  }

  return { name: graphName, nodes, initializers, inputs, outputs }
}

// ---- Public API ----

export function parseOnnxProto(buffer: ArrayBuffer): ParsedGraph {
  const buf = new Uint8Array(buffer)
  const r = new ProtoReader(buf)
  let graph: ParsedGraph | null = null

  while (!r.done) {
    const tag = r.readTag()
    if (!tag) break
    if (tag.wire === WIRE_LEN) {
      const len = r.readVarint()
      if (tag.field === MODEL_GRAPH) {
        graph = parseGraph(r.subReader(len))
      } else {
        r.skip(len)
      }
    } else {
      r.skipField(tag.wire)
    }
  }

  return graph ?? { name: '', nodes: [], initializers: [], inputs: [], outputs: [] }
}

export function formatShape(dims: OnnxDim[] | undefined): string {
  if (!dims || dims.length === 0) return ''
  return '[' + dims.map(d => 'value' in d ? d.value : d.param).join(', ') + ']'
}
