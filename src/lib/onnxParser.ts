import type { InferenceSession } from 'onnxruntime-web'
import type { OnnxNode, OnnxEdge, OnnxGraph } from './onnxTypes'

export function parseOnnxGraph(session: InferenceSession, modelName: string): OnnxGraph {
  const nodes: OnnxNode[] = []
  const edges: OnnxEdge[] = []
  const tensorToNodeId = new Map<string, string>()

  // Input nodes
  session.inputNames.forEach((name, i) => {
    const id = `input_${i}`
    nodes.push({ id, opType: 'Input', inputs: [], outputs: [name], attributes: {}, paramCount: 0, estimatedSizeMB: 0 })
    tensorToNodeId.set(name, id)
  })

  // Try to access internal graph nodes via WASM backend internals
  const graphNodes: unknown[] = (session as any)?.handler?.model?.graph?.node ?? []
  const initializers: unknown[] = (session as any)?.handler?.model?.graph?.initializer ?? []

  // Build param count from initializers
  const initializerSizes = new Map<string, number>()
  for (const init of initializers) {
    const i = init as any
    if (i.name && Array.isArray(i.dims)) {
      const count = (i.dims as number[]).reduce((a: number, b: number) => a * b, 1)
      initializerSizes.set(i.name, count)
    }
  }

  if (graphNodes.length > 0) {
    graphNodes.forEach((rawNode, idx) => {
      const n = rawNode as any
      const opType: string = n.opType ?? n.op_type ?? 'Unknown'
      const inputNames: string[] = Array.isArray(n.input) ? n.input : []
      const outputNames: string[] = Array.isArray(n.output) ? n.output : []
      const nodeId = `node_${idx}_${opType}`

      const attrs: Record<string, string | number | boolean> = {}
      if (Array.isArray(n.attribute)) {
        for (const attr of n.attribute as any[]) {
          if (attr.name) attrs[attr.name] = attr.f ?? attr.i ?? attr.s ?? attr.t ?? ''
        }
      }

      // Sum param count from weight initializers used by this node
      let paramCount = 0
      for (const inp of inputNames) {
        if (initializerSizes.has(inp)) paramCount += initializerSizes.get(inp)!
      }

      const estimatedSizeMB = (paramCount * 4) / (1024 * 1024)
      nodes.push({ id: nodeId, opType, inputs: inputNames, outputs: outputNames, attributes: attrs, paramCount, estimatedSizeMB })

      // Wire edges from tensor producers to this node
      for (const inp of inputNames) {
        const sourceId = tensorToNodeId.get(inp)
        if (sourceId) {
          edges.push({ id: `${sourceId}->${nodeId}`, source: sourceId, target: nodeId, label: inp })
        }
      }

      for (const out of outputNames) {
        tensorToNodeId.set(out, nodeId)
      }
    })
  } else {
    // Fallback: minimal graph with only input/output nodes when internals not accessible
    const intermediateId = 'model_body'
    nodes.push({
      id: intermediateId,
      opType: 'Model',
      inputs: [...session.inputNames],
      outputs: [...session.outputNames],
      attributes: {},
      paramCount: 0,
      estimatedSizeMB: 0,
    })
    session.inputNames.forEach((name, i) => {
      edges.push({ id: `input_${i}->${intermediateId}`, source: `input_${i}`, target: intermediateId, label: name })
    })
    session.outputNames.forEach((name) => {
      tensorToNodeId.set(name, intermediateId)
    })
  }

  // Output nodes
  session.outputNames.forEach((name, i) => {
    const id = `output_${i}`
    const sourceId = tensorToNodeId.get(name)
    nodes.push({ id, opType: 'Output', inputs: [name], outputs: [], attributes: {}, paramCount: 0, estimatedSizeMB: 0 })
    if (sourceId) {
      edges.push({ id: `${sourceId}->${id}`, source: sourceId, target: id, label: name })
    }
  })

  const totalParams = nodes.reduce((sum, n) => sum + n.paramCount, 0)
  const totalSizeMB = (totalParams * 4) / (1024 * 1024)

  return { nodes, edges, modelName, totalParams, totalSizeMB }
}
