import type { OnnxGraph, OnnxNode } from './onnxTypes'

export interface SelectableNode extends OnnxNode {
  selected: boolean
  dimmed?: boolean
  excluded?: boolean
}

export interface SelectableGraph {
  nodes: SelectableNode[]
  edges: OnnxGraph['edges']
  modelName: string
  totalParams: number
  totalSizeMB: number
}

export function toSelectableGraph(graph: OnnxGraph): SelectableGraph {
  return { ...graph, nodes: graph.nodes.map((n) => ({ ...n, selected: false })) }
}

export function selectNode(graph: SelectableGraph, nodeId: string): SelectableGraph {
  return {
    ...graph,
    nodes: graph.nodes.map((n) => ({ ...n, selected: n.id === nodeId ? true : n.selected })),
  }
}

export function deselectAll(graph: SelectableGraph): SelectableGraph {
  return { ...graph, nodes: graph.nodes.map((n) => ({ ...n, selected: false })) }
}

export function setMultiSelection(graph: SelectableGraph, ids: Set<string>): SelectableGraph {
  return { ...graph, nodes: graph.nodes.map((n) => ({ ...n, selected: ids.has(n.id) })) }
}

export function getSelectedNodes(graph: SelectableGraph): SelectableNode[] {
  return graph.nodes.filter((n) => n.selected)
}

export function validateEdges(graph: OnnxGraph): OnnxGraph['edges'] {
  const nodeIds = new Set(graph.nodes.map((n) => n.id))
  return graph.edges.filter((e) => !nodeIds.has(e.source) || !nodeIds.has(e.target))
}

export function filterGraph(graph: SelectableGraph, query: string): SelectableGraph {
  const q = query.toLowerCase()
  return {
    ...graph,
    nodes: graph.nodes.map((n) => {
      const matches =
        n.opType.toLowerCase().includes(q) ||
        n.id.toLowerCase().includes(q) ||
        n.inputs.some(t => t.toLowerCase().includes(q)) ||
        n.outputs.some(t => t.toLowerCase().includes(q))
      return { ...n, dimmed: !matches }
    }),
  }
}

export function excludeNode(graph: SelectableGraph, nodeId: string): SelectableGraph {
  return {
    ...graph,
    nodes: graph.nodes.map((n) => (n.id === nodeId ? { ...n, excluded: true } : n)),
  }
}

export function includeNode(graph: SelectableGraph, nodeId: string): SelectableGraph {
  return {
    ...graph,
    nodes: graph.nodes.map((n) => (n.id === nodeId ? { ...n, excluded: false } : n)),
  }
}

export function bulkExclude(graph: SelectableGraph, ids: Set<string>): SelectableGraph {
  return { ...graph, nodes: graph.nodes.map((n) => (ids.has(n.id) ? { ...n, excluded: true } : n)) }
}

export function bulkInclude(graph: SelectableGraph, ids: Set<string>): SelectableGraph {
  return { ...graph, nodes: graph.nodes.map((n) => (ids.has(n.id) ? { ...n, excluded: false } : n)) }
}

export function computeOpCounts(nodes: OnnxNode[]): Record<string, number> {
  const counts: Record<string, number> = {}
  for (const n of nodes) {
    if (n.opType === 'Input' || n.opType === 'Output') continue
    counts[n.opType] = (counts[n.opType] ?? 0) + 1
  }
  return counts
}

export const OP_CATEGORIES: Record<string, string> = {
  Conv: '#C0392B', ConvTranspose: '#C0392B', DepthwiseConv: '#C0392B',
  Relu: '#52C57A', Sigmoid: '#52C57A', Tanh: '#52C57A', Elu: '#52C57A',
  LeakyRelu: '#52C57A', Gelu: '#52C57A', HardSwish: '#52C57A', Mish: '#52C57A',
  Selu: '#52C57A', Softmax: '#52C57A', LogSoftmax: '#52C57A', Swish: '#52C57A',
  BatchNormalization: '#3498DB', LayerNormalization: '#3498DB',
  InstanceNormalization: '#3498DB', GroupNormalization: '#3498DB', LpNormalization: '#3498DB',
  Gemm: '#E67E22', MatMul: '#E67E22', Linear: '#E67E22',
  MaxPool: '#9B59B6', AveragePool: '#9B59B6', GlobalAveragePool: '#9B59B6',
  GlobalMaxPool: '#9B59B6', LpPool: '#9B59B6',
  Reshape: '#1ABC9C', Transpose: '#1ABC9C', Flatten: '#1ABC9C',
  Squeeze: '#1ABC9C', Unsqueeze: '#1ABC9C', Gather: '#1ABC9C',
  Concat: '#1ABC9C', Split: '#1ABC9C', Slice: '#1ABC9C',
  // TFLite op-name equivalents (SCREAMING_SNAKE_CASE), same category colors as their
  // ONNX counterparts above -- without these every TFLite node falls through to the
  // gray "Other" category.
  CONV_2D: '#C0392B', DEPTHWISE_CONV_2D: '#C0392B', TRANSPOSE_CONV: '#C0392B', CONV_3D: '#C0392B',
  RELU: '#52C57A', RELU6: '#52C57A', RELU_N1_TO_1: '#52C57A', LOGISTIC: '#52C57A',
  TANH: '#52C57A', ELU: '#52C57A', LEAKY_RELU: '#52C57A', GELU: '#52C57A',
  HARD_SWISH: '#52C57A', SOFTMAX: '#52C57A', LOG_SOFTMAX: '#52C57A', PRELU: '#52C57A',
  L2_NORMALIZATION: '#3498DB', LOCAL_RESPONSE_NORMALIZATION: '#3498DB',
  FULLY_CONNECTED: '#E67E22', BATCH_MATMUL: '#E67E22',
  MAX_POOL_2D: '#9B59B6', AVERAGE_POOL_2D: '#9B59B6', L2_POOL_2D: '#9B59B6',
  RESHAPE: '#1ABC9C', TRANSPOSE: '#1ABC9C', SQUEEZE: '#1ABC9C', EXPAND_DIMS: '#1ABC9C',
  GATHER: '#1ABC9C', GATHER_ND: '#1ABC9C', CONCATENATION: '#1ABC9C', SPLIT: '#1ABC9C',
  SPLIT_V: '#1ABC9C', SLICE: '#1ABC9C', STRIDED_SLICE: '#1ABC9C',
}

export function opCategoryColor(opType: string): string {
  return OP_CATEGORIES[opType] ?? 'rgba(255,255,255,0.15)'
}

export function getAncestors(graph: OnnxGraph, nodeId: string): Set<string> {
  const result = new Set<string>()
  const stack = [nodeId]
  while (stack.length > 0) {
    const current = stack.pop() as string
    for (const e of graph.edges) {
      if (e.target === current && !result.has(e.source)) {
        result.add(e.source)
        stack.push(e.source)
      }
    }
  }
  result.delete(nodeId)
  return result
}

export function getDescendants(graph: OnnxGraph, nodeId: string): Set<string> {
  const result = new Set<string>()
  const stack = [nodeId]
  while (stack.length > 0) {
    const current = stack.pop() as string
    for (const e of graph.edges) {
      if (e.source === current && !result.has(e.target)) {
        result.add(e.target)
        stack.push(e.target)
      }
    }
  }
  result.delete(nodeId)
  return result
}

// Only nodes parsed directly from the loaded model (id format `node_<idx>_<opType>`)
// are valid structural-edit targets. Synthetic nodes created by insertPassthroughNode
// use a different id scheme and are deliberately excluded from further editing --
// chaining edits onto generated nodes would require resolving ops against ops rather
// than always against the original model, which is out of scope for now.
const ORIGINAL_NODE_ID_RE = /^node_\d+_/

export interface DeleteCandidate {
  tensorName: string
  position: number
}

export interface DeleteEligibility {
  eligible: boolean
  reason?: string
  candidateInputs: DeleteCandidate[]
}

export function getDeleteEligibility(graph: SelectableGraph, nodeId: string): DeleteEligibility {
  const node = graph.nodes.find((n) => n.id === nodeId)
  if (!node) return { eligible: false, reason: 'Node not found', candidateInputs: [] }
  if (!ORIGINAL_NODE_ID_RE.test(nodeId)) {
    return { eligible: false, reason: 'Generated nodes cannot be deleted', candidateInputs: [] }
  }
  const realOutputs = node.outputs.filter((o) => o !== '')
  if (realOutputs.length > 1) {
    return { eligible: false, reason: 'Node has multiple outputs', candidateInputs: [] }
  }
  const hasConsumers = graph.edges.some((e) => e.source === nodeId)
  if (!hasConsumers) {
    // Dead end: nothing downstream to reconnect, safe to remove outright.
    return { eligible: true, candidateInputs: [] }
  }
  const feedsGraphOutput = graph.edges.some((e) => e.source === nodeId && e.target.startsWith('output_'))
  if (feedsGraphOutput) {
    return { eligible: false, reason: 'Node produces a graph output', candidateInputs: [] }
  }
  const candidateInputs: DeleteCandidate[] = node.inputs
    .map((tensorName, position) => ({ tensorName, position }))
    .filter(({ tensorName }) => tensorName !== '' && graph.edges.some((e) => e.target === nodeId && e.label === tensorName))
  if (candidateInputs.length === 0) {
    return { eligible: false, reason: 'No upstream connection to reconnect', candidateInputs: [] }
  }
  return { eligible: true, candidateInputs }
}

// keepInputPosition indexes node.inputs; pass null for the dead-end case (no
// reconnection needed). Caller is expected to have checked getDeleteEligibility first.
export function deleteNodeWithReconnect(
  graph: SelectableGraph,
  nodeId: string,
  keepInputPosition: number | null,
): SelectableGraph {
  const node = graph.nodes.find((n) => n.id === nodeId)
  if (!node) return graph

  let sourceId: string | undefined
  let keepTensor: string | undefined
  if (keepInputPosition !== null) {
    keepTensor = node.inputs[keepInputPosition]
    const incoming = graph.edges.find((e) => e.target === nodeId && e.label === keepTensor)
    sourceId = incoming?.source
  }

  const reconnected =
    sourceId && keepTensor
      ? graph.edges
          .filter((e) => e.source === nodeId)
          .map((e) => ({
            id: `${sourceId}->${e.target}@${keepTensor}`,
            source: sourceId as string,
            target: e.target,
            label: keepTensor,
            shape: e.shape,
          }))
      : []

  return {
    ...graph,
    nodes: graph.nodes.filter((n) => n.id !== nodeId),
    edges: graph.edges.filter((e) => e.source !== nodeId && e.target !== nodeId).concat(reconnected),
  }
}

// inputPosition indexes targetNode.inputs (the current/live tensor name at that
// position, which may already reflect earlier structural edits). newNodeId must be
// a fresh, unique id -- callers should generate one that never collides with the
// `node_<idx>_<opType>` scheme (e.g. `passthrough_<counter>`).
export function insertPassthroughNode(
  graph: SelectableGraph,
  targetNodeId: string,
  inputPosition: number,
  newNodeId: string,
): SelectableGraph {
  const target = graph.nodes.find((n) => n.id === targetNodeId)
  if (!target) return graph
  const tensorName = target.inputs[inputPosition]
  if (!tensorName) return graph
  const incoming = graph.edges.find((e) => e.target === targetNodeId && e.label === tensorName)
  if (!incoming) return graph

  const newTensorName = `${tensorName}__identity_${newNodeId}`
  const passthroughNode: SelectableNode = {
    id: newNodeId,
    opType: 'Identity',
    inputs: [tensorName],
    outputs: [newTensorName],
    attributes: {},
    paramCount: 0,
    estimatedSizeMB: 0,
    selected: false,
  }
  const rewiredTarget = { ...target, inputs: target.inputs.map((inp, i) => (i === inputPosition ? newTensorName : inp)) }

  return {
    ...graph,
    nodes: graph.nodes.map((n) => (n.id === targetNodeId ? rewiredTarget : n)).concat(passthroughNode),
    edges: graph.edges
      .filter((e) => e !== incoming)
      .concat([
        { id: `${incoming.source}->${newNodeId}@${tensorName}`, source: incoming.source, target: newNodeId, label: tensorName, shape: incoming.shape },
        { id: `${newNodeId}->${targetNodeId}@${newTensorName}`, source: newNodeId, target: targetNodeId, label: newTensorName, shape: incoming.shape },
      ]),
  }
}

export function computeGraphDepth(graph: OnnxGraph): number {
  if (graph.nodes.length === 0) return 0
  const incoming = new Map<string, number>()
  const adjacency = new Map<string, string[]>()
  for (const n of graph.nodes) {
    incoming.set(n.id, 0)
    adjacency.set(n.id, [])
  }
  for (const e of graph.edges) {
    if (!incoming.has(e.source) || !incoming.has(e.target)) continue
    incoming.set(e.target, (incoming.get(e.target) ?? 0) + 1)
    adjacency.get(e.source)?.push(e.target)
  }
  const depth = new Map<string, number>()
  const queue: string[] = []
  for (const [id, count] of incoming) {
    if (count === 0) {
      depth.set(id, 0)
      queue.push(id)
    }
  }
  let maxDepth = 0
  while (queue.length > 0) {
    const current = queue.shift() as string
    const currentDepth = depth.get(current) ?? 0
    for (const next of adjacency.get(current) ?? []) {
      const candidate = currentDepth + 1
      if (candidate > (depth.get(next) ?? -1)) {
        depth.set(next, candidate)
        maxDepth = Math.max(maxDepth, candidate)
      }
      const remaining = (incoming.get(next) ?? 0) - 1
      incoming.set(next, remaining)
      if (remaining === 0) queue.push(next)
    }
  }
  return maxDepth
}
