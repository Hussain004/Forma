// Extracts a minimal, self-contained ONNX model from a selected subset of the
// ORIGINAL model's nodes: only those nodes, only the initializers they
// actually reference, and boundary tensors (inputs produced outside the
// selection, outputs consumed outside it) promoted to fresh graph
// inputs/outputs. Always operates on the original loaded bytes, not any
// active edit state -- a "minimal reproduction" is a piece of the model as
// it was loaded, not a mix of that plus in-progress edits.
//
// Reuses the same low-level protobuf primitives as onnxProtoWriter.ts rather
// than a full ONNX serializer. Everything outside GraphProto (ir_version,
// opset_import, producer info) passes through byte-for-byte untouched, via
// the same patchLenFields(full, MODEL_GRAPH, ...) shape writeModifiedOnnx
// uses. Inside GraphProto, only node/initializer/input/output are emitted --
// value_info, doc_string, and other optional fields are dropped, since
// nothing downstream needs them and dropping avoids having to reconcile them
// against a node list that's now a strict subset.

import {
  ProtoReader,
  WIRE_LEN,
  MODEL_GRAPH,
  GRAPH_NODE,
  GRAPH_INIT,
  GRAPH_INPUT,
  GRAPH_OUTPUT,
  GRAPH_VALUE_INFO,
  INIT_NAME,
  VINFO_NAME,
  readTopLevelStringField,
} from './onnxProtoParser'
import {
  patchLenFields,
  decodeNodeIO,
  encodeLenField,
  encodeValueInfo,
  concatBytes,
} from './onnxProtoWriter'

export class SubgraphExtractionError extends Error {}

interface RawEntry { name: string; bytes: Uint8Array }
interface RawNode { origIndex: number; bytes: Uint8Array; inputs: string[]; outputs: string[] }

// Fallback for a boundary tensor with no existing ValueInfoProto to reuse:
// name plus a bare elem_type, shape omitted entirely (unranked, not
// zero-rank -- an empty TensorShapeProto would mean "scalar", which is wrong
// here). Defaults to FLOAT (1), the same default this codebase uses
// elsewhere (onnxWorker's benchmark/validation tensor construction) when a
// tensor's real dtype isn't known.
function encodeUnrankedValueInfo(name: string): Uint8Array {
  return encodeValueInfo(name, 1, null)
}

export function extractSubgraph(buffer: ArrayBuffer, selectedIndices: Set<number>): ArrayBuffer {
  if (selectedIndices.size === 0) throw new SubgraphExtractionError('No nodes selected')
  const full = new Uint8Array(buffer)

  const graphPatcher = (_occurrence: number, graphContent: Uint8Array): Uint8Array => {
    const r = new ProtoReader(graphContent)
    const allNodes: RawNode[] = []
    const initializers: RawEntry[] = []
    // ValueInfoProto is just {name, type} -- which repeated field it came
    // from (input/output/value_info) doesn't matter for reuse, only the name
    // match does, so all three collections merge into one lookup.
    const knownValueInfo = new Map<string, Uint8Array>()
    let nodeOccurrence = 0

    while (!r.done) {
      const tag = r.readTag()
      if (!tag) break
      if (tag.wire === WIRE_LEN) {
        const len = r.readVarint()
        const sub = graphContent.subarray(r.pos, r.pos + len)
        r.skip(len)
        if (tag.field === GRAPH_NODE) {
          allNodes.push({ origIndex: nodeOccurrence, bytes: sub, ...decodeNodeIO(sub) })
          nodeOccurrence++
        } else if (tag.field === GRAPH_INIT) {
          initializers.push({ name: readTopLevelStringField(sub, INIT_NAME), bytes: sub })
        } else if (tag.field === GRAPH_INPUT || tag.field === GRAPH_OUTPUT || tag.field === GRAPH_VALUE_INFO) {
          const name = readTopLevelStringField(sub, VINFO_NAME)
          if (name) knownValueInfo.set(name, sub)
        }
      } else {
        r.skipField(tag.wire)
      }
    }

    const selected = allNodes.filter((n) => selectedIndices.has(n.origIndex))
    if (selected.length !== selectedIndices.size) {
      throw new SubgraphExtractionError('Some selected nodes were not found in the model')
    }

    const selectedOutputs = new Set(selected.flatMap((n) => n.outputs.filter(Boolean)))
    const selectedInputs = new Set(selected.flatMap((n) => n.inputs.filter(Boolean)))
    const initializerNames = new Set(initializers.map((i) => i.name))

    // A boundary input is anything a selected node consumes that isn't
    // produced by another selected node and isn't a preserved constant.
    const boundaryInputs = [...selectedInputs].filter((name) => !selectedOutputs.has(name) && !initializerNames.has(name))
    // A boundary output is anything a selected node produces that no
    // selected node consumes -- covers both "the original graph output" and
    // "consumed by a node outside the selection" identically.
    const boundaryOutputs = [...selectedOutputs].filter((name) => !selectedInputs.has(name))
    const requiredInitializers = initializers.filter((i) => selectedInputs.has(i.name))

    const valueInfoBytes = (name: string): Uint8Array => knownValueInfo.get(name) ?? encodeUnrankedValueInfo(name)

    const chunks: Uint8Array[] = [
      ...selected.map((n) => encodeLenField(GRAPH_NODE, n.bytes)),
      ...requiredInitializers.map((i) => encodeLenField(GRAPH_INIT, i.bytes)),
      ...boundaryInputs.map((name) => encodeLenField(GRAPH_INPUT, valueInfoBytes(name))),
      ...boundaryOutputs.map((name) => encodeLenField(GRAPH_OUTPUT, valueInfoBytes(name))),
    ]
    return concatBytes(chunks)
  }

  const patched = patchLenFields(full, MODEL_GRAPH, graphPatcher)
  return patched.buffer as ArrayBuffer
}
