import type { OnnxNode, OnnxEdge, OnnxGraph } from './onnxTypes'
import { parseOnnxProto } from './onnxProtoParser'

export function parseOnnxGraph(buffer: ArrayBuffer, modelName: string): OnnxGraph {
  const proto = parseOnnxProto(buffer)

  const nodes: OnnxNode[] = []
  const edges: OnnxEdge[] = []

  // Map tensor name -> the node id that produces it
  const tensorProducer = new Map<string, string>()

  // Map initializer name -> elem count (weights owned by compute nodes)
  const initElemCounts = new Map<string, number>()
  const initSizes = new Map<string, number>()
  for (const init of proto.initializers) {
    initElemCounts.set(init.name, init.elemCount)
    initSizes.set(init.name, init.sizeMB)
  }

  // Build a map of tensor name -> shape from graph inputs, outputs, and intermediate value_info
  const tensorShapes = new Map<string, import('./onnxProtoParser').OnnxDim[]>()
  for (const vi of [...proto.inputs, ...proto.outputs, ...proto.valueInfo]) {
    if (vi.name && vi.shape) tensorShapes.set(vi.name, vi.shape)
  }

  // Graph input nodes (non-initializer inputs only)
  const initNames = new Set(proto.initializers.map(i => i.name))
  const graphInputs = proto.inputs.filter(vi => !initNames.has(vi.name))

  graphInputs.forEach((vi, i) => {
    const id = `input_${i}`
    nodes.push({
      id,
      opType: 'Input',
      inputs: [],
      outputs: [vi.name],
      attributes: {},
      paramCount: 0,
      estimatedSizeMB: 0,
      outputShapes: vi.shape ? [vi.shape] : undefined,
    })
    tensorProducer.set(vi.name, id)
  })

  // Compute nodes
  proto.nodes.forEach((rawNode, idx) => {
    const id = `node_${idx}_${rawNode.opType}`

    // Parameter count = sum of elem counts for weight inputs (initializers)
    let paramCount = 0
    let estimatedSizeMB = 0
    for (const inp of rawNode.inputs) {
      if (initElemCounts.has(inp)) paramCount += initElemCounts.get(inp)!
      if (initSizes.has(inp)) estimatedSizeMB += initSizes.get(inp)!
    }

    const inputShapes = rawNode.inputs
      .filter(name => tensorShapes.has(name))
      .map(name => tensorShapes.get(name)!)

    const outputShapes = rawNode.outputs
      .filter(name => tensorShapes.has(name))
      .map(name => tensorShapes.get(name)!)

    nodes.push({
      id,
      name: rawNode.name || undefined,
      opType: rawNode.opType,
      inputs: rawNode.inputs,
      outputs: rawNode.outputs,
      attributes: rawNode.attributes,
      paramCount,
      estimatedSizeMB,
      inputShapes: inputShapes.length > 0 ? inputShapes : undefined,
      outputShapes: outputShapes.length > 0 ? outputShapes : undefined,
    })

    // Wire edges from tensor producers to this node (skip initializers as visual sources)
    for (const inp of rawNode.inputs) {
      if (initNames.has(inp)) continue
      const sourceId = tensorProducer.get(inp)
      if (sourceId) {
        edges.push({ id: `${sourceId}->${id}@${inp}`, source: sourceId, target: id, label: inp, shape: tensorShapes.get(inp) })
      }
    }

    // Register this node as the producer of its output tensors
    for (const out of rawNode.outputs) {
      tensorProducer.set(out, id)
    }
  })

  // Graph output nodes
  proto.outputs.forEach((vi, i) => {
    const id = `output_${i}`
    const sourceId = tensorProducer.get(vi.name)
    nodes.push({
      id,
      opType: 'Output',
      inputs: [vi.name],
      outputs: [],
      attributes: {},
      paramCount: 0,
      estimatedSizeMB: 0,
      inputShapes: vi.shape ? [vi.shape] : undefined,
    })
    if (sourceId) {
      edges.push({ id: `${sourceId}->${id}@${vi.name}`, source: sourceId, target: id, label: vi.name, shape: tensorShapes.get(vi.name) })
    }
  })

  const totalParams = nodes.reduce((sum, n) => sum + n.paramCount, 0)
  const totalSizeMB = nodes.reduce((sum, n) => sum + n.estimatedSizeMB, 0)

  return { nodes, edges, modelName, totalParams, totalSizeMB, graphInputs, metadata: proto.metadata }
}
