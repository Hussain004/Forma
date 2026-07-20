import type { OnnxDim, OnnxGraph, OnnxNode, TensorMetadata } from './onnxTypes'

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
export interface AttrHistoryEntry {
  type: 'attr'
  nodeId: string
  attrName: string
  value: string | number
}

export interface DeleteHistoryEntry {
  type: 'delete'
  nodeId: string
  nodeIndex: number
  keepInputPosition: number | null
}

export interface BulkDeleteHistoryEntry {
  type: 'bulkDelete'
  deletions: DeleteHistoryEntry[]
}

export interface InsertPassthroughHistoryEntry {
  type: 'insertPassthrough'
  targetNodeId: string
  targetNodeIndex: number
  inputPosition: number
  newNodeId: string
}

export interface RewireHistoryEntry {
  type: 'rewire'
  sourceNodeId: string
  sourceNodeIndex: number
  targetNodeId: string
  targetNodeIndex: number
  inputPosition: number
}

export interface AddNodeHistoryEntry {
  type: 'addNode'
  newNodeId: string
  newNodeIndex: number
  opType: string
  inputCount: number
  position: { x: number; y: number }
}

export type StructuralHistoryEntry =
  | DeleteHistoryEntry
  | InsertPassthroughHistoryEntry
  | RewireHistoryEntry
  | AddNodeHistoryEntry

// The history log is the sole source of truth for model edits. Its entries
// preserve the live node ids needed by the canvas alongside the stable indexes
// needed by the protobuf writer, so replay can drive both consumers unchanged.
export type HistoryEntry =
  | AttrHistoryEntry
  | StructuralHistoryEntry
  | BulkDeleteHistoryEntry

export function friendlyNodeLabel(nodeId: string): string {
  const original = /^node_\d+_(.+)$/.exec(nodeId)
  if (original) return original[1]
  const custom = /^custom_(\d+)$/.exec(nodeId)
  if (custom) return `Custom node ${custom[1]}`
  const passthrough = /^passthrough_(\d+)$/.exec(nodeId)
  if (passthrough) return `Passthrough ${passthrough[1]}`
  return 'Node'
}

function formatHistoryValue(value: string | number): string {
  return typeof value === 'string' ? value : String(value)
}

export function describeHistoryEntry(entry: HistoryEntry): string {
  switch (entry.type) {
    case 'attr':
      return `Changed ${friendlyNodeLabel(entry.nodeId)} ${entry.attrName} to ${formatHistoryValue(entry.value)}`
    case 'delete':
      return `Deleted ${friendlyNodeLabel(entry.nodeId)}`
    case 'bulkDelete':
      return `Deleted ${entry.deletions.length} nodes`
    case 'insertPassthrough':
      return `Inserted passthrough before ${friendlyNodeLabel(entry.targetNodeId)}`
    case 'rewire':
      return `Rewired ${friendlyNodeLabel(entry.targetNodeId)} input ${entry.inputPosition + 1} to ${friendlyNodeLabel(entry.sourceNodeId)}`
    case 'addNode':
      return `Added ${entry.opType} node`
  }

}

function edgeSignature(edge: OnnxGraph['edges'][number]): string {
  return `${edge.source}\u0000${edge.target}\u0000${edge.label ?? ''}`
}

// Builds a comparison graph without mutating the edited graph used for export,
// stats, selection, or history. Deleted original nodes and connections are
// restored only as display artifacts. Current connections that do not exist in
// the original graph are marked as changed.
export function buildGraphDiff(original: SelectableGraph, current: SelectableGraph): SelectableGraph {
  const currentNodeIds = new Set(current.nodes.map((node) => node.id))
  const currentEdgeSignatures = new Set(current.edges.map(edgeSignature))
  const originalEdgeSignatures = new Set(original.edges.map(edgeSignature))

  const deletedNodes = original.nodes
    .filter((node) => !currentNodeIds.has(node.id))
    .map((node) => ({ ...node, selected: false, diffStatus: 'deleted' as const }))

  const currentEdges = current.edges.map((edge) => (
    originalEdgeSignatures.has(edgeSignature(edge))
      ? edge
      : { ...edge, diffStatus: 'changed' as const }
  ))

  const removedEdges = original.edges
    .filter((edge) => !currentEdgeSignatures.has(edgeSignature(edge)))
    .map((edge, index) => ({
      ...edge,
      id: `diff_removed_${index}_${edge.id}`,
      diffStatus: 'removed' as const,
    }))

  return {
    ...current,
    nodes: [...current.nodes, ...deletedNodes],
    edges: [...currentEdges, ...removedEdges],
  }
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

export function getAncestors(graph: Pick<OnnxGraph, 'nodes' | 'edges'>, nodeId: string): Set<string> {
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

export function getDescendants(graph: Pick<OnnxGraph, 'nodes' | 'edges'>, nodeId: string): Set<string> {
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
// or added via addCustomNode (id format `custom_<n>`) are valid structural-edit
// targets. Synthetic nodes created by insertPassthroughNode use a different id
// scheme and are deliberately excluded from further editing -- chaining edits onto
// a generated passthrough would require resolving ops against ops rather than
// always against the original model, which is out of scope for now. Custom-added
// nodes are the one exception: being addressable is the entire point of v1.5.
const ORIGINAL_NODE_ID_RE = /^node_(\d+)_/
const CUSTOM_NODE_ID_RE = /^custom_(\d+)$/

// Maps a live node id to the signed numeric index onnxProtoWriter.ts uses to
// address it: original nodes keep their non-negative position in the source
// file's GraphProto.node; custom-added nodes get the negation of their creation
// counter (matching the sentinel convention onnxProtoWriter already used for
// insertPassthrough's generated entries, just made addressable instead of
// terminal). Returns null for anything else (passthrough_N, Input/Output
// pseudo-nodes) -- those aren't valid structural-edit endpoints.
export function structuralNodeIndex(nodeId: string): number | null {
  const orig = ORIGINAL_NODE_ID_RE.exec(nodeId)
  if (orig) return Number(orig[1])
  const custom = CUSTOM_NODE_ID_RE.exec(nodeId)
  if (custom) return -Number(custom[1])
  return null
}

export function isStructuralEditTarget(nodeId: string): boolean {
  return structuralNodeIndex(nodeId) !== null
}

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
  if (!isStructuralEditTarget(nodeId)) {
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

// A small curated menu for the "Add Node" picker -- common ops with no required
// attributes, so they're immediately usable through the existing v1.0 inline-edit
// flow (which can only change an existing attribute's value, not add a new key).
// Free text always defaults to inputCount 1; the curated entries exist purely for
// discoverability and to save a click adjusting variadic-input ops like Add/Concat.
export const CURATED_NODE_TYPES: { opType: string; inputCount: number }[] = [
  { opType: 'Relu', inputCount: 1 },
  { opType: 'Sigmoid', inputCount: 1 },
  { opType: 'Tanh', inputCount: 1 },
  { opType: 'Softmax', inputCount: 1 },
  { opType: 'Identity', inputCount: 1 },
  { opType: 'Add', inputCount: 2 },
  { opType: 'Sub', inputCount: 2 },
  { opType: 'Mul', inputCount: 2 },
  { opType: 'Concat', inputCount: 2 },
  { opType: 'MatMul', inputCount: 2 },
]

// Adds a new, initially unconnected node to the canvas -- inputCount placeholder
// slots (non-empty synthetic names so rewireEdge's truthy check can target them,
// but matching no real producer, so no edge is drawn) and one real output tensor
// name other nodes can immediately rewire to consume. newNodeId must be a fresh,
// unique id from the `custom_<n>` scheme (see structuralNodeIndex above).
export function addCustomNode(
  graph: SelectableGraph,
  newNodeId: string,
  opType: string,
  inputCount: number,
  position: { x: number; y: number },
): SelectableGraph {
  const inputs = Array.from({ length: inputCount }, (_, i) => `__unwired_${newNodeId}_in${i}`)
  const newNode: SelectableNode = {
    id: newNodeId,
    opType,
    inputs,
    outputs: [`${newNodeId}_out`],
    attributes: {},
    paramCount: 0,
    estimatedSizeMB: 0,
    selected: false,
    position,
  }
  return { ...graph, nodes: [...graph.nodes, newNode] }
}

export interface RewireValidation {
  valid: boolean
  reason?: string
}

const ONNX_ELEM_TYPE_NAMES: Record<number, string> = {
  1: 'FLOAT', 2: 'UINT8', 3: 'INT8', 4: 'UINT16', 5: 'INT16',
  6: 'INT32', 7: 'INT64', 8: 'STRING', 9: 'BOOL', 10: 'FLOAT16',
  11: 'DOUBLE', 12: 'UINT32', 13: 'UINT64', 14: 'COMPLEX64',
  15: 'COMPLEX128', 16: 'BFLOAT16', 17: 'FLOAT8E4M3FN',
  18: 'FLOAT8E4M3FNUZ', 19: 'FLOAT8E5M2', 20: 'FLOAT8E5M2FNUZ',
  21: 'UINT4', 22: 'INT4', 23: 'FLOAT4E2M1', 24: 'FLOAT8E8M0',
}

function elemTypeLabel(elemType: number): string {
  return ONNX_ELEM_TYPE_NAMES[elemType] ?? 'TYPE_' + elemType
}

function concreteDimension(dim: OnnxDim): number | null {
  return 'value' in dim ? dim.value : null
}

export function validateTensorCompatibility(
  source: TensorMetadata | undefined,
  target: TensorMetadata | undefined,
): RewireValidation {
  if (source?.elemType && target?.elemType && source.elemType !== target.elemType) {
    return {
      valid: false,
      reason: 'Tensor type mismatch: source ' + elemTypeLabel(source.elemType) + ', target expects ' + elemTypeLabel(target.elemType),
    }
  }

  if (!source?.shape || !target?.shape) return { valid: true }
  if (source.shape.length !== target.shape.length) {
    return {
      valid: false,
      reason: 'Shape rank mismatch: source rank ' + source.shape.length + ', target expects rank ' + target.shape.length,
    }
  }

  for (let index = 0; index < source.shape.length; index += 1) {
    const sourceValue = concreteDimension(source.shape[index])
    const targetValue = concreteDimension(target.shape[index])
    if (sourceValue !== null && targetValue !== null && sourceValue !== targetValue) {
      return {
        valid: false,
        reason: 'Shape mismatch at dimension ' + (index + 1) + ': source ' + sourceValue + ', target expects ' + targetValue,
      }
    }
  }

  return { valid: true }
}

// Original and custom-added nodes are valid rewire endpoints. Known target
// tensor contracts are preserved, while incomplete metadata remains permissive.
export function validateRewire(
  graph: Pick<OnnxGraph, 'nodes' | 'edges'>,
  sourceNodeId: string,
  targetNodeId: string,
  inputPosition: number,
): RewireValidation {
  if (sourceNodeId === targetNodeId) {
    return { valid: false, reason: 'Cannot connect a node to itself' }
  }
  if (!isStructuralEditTarget(sourceNodeId) || !isStructuralEditTarget(targetNodeId)) {
    return { valid: false, reason: 'Only original or custom-added nodes can be rewired' }
  }
  const source = graph.nodes.find((n) => n.id === sourceNodeId)
  const target = graph.nodes.find((n) => n.id === targetNodeId)
  if (!source || !target) return { valid: false, reason: 'Node not found' }
  const sourceOutputPosition = source.outputs.findIndex((output) => output !== '')
  const sourceTensor = source.outputs[sourceOutputPosition]
  if (!sourceTensor) return { valid: false, reason: 'Source has no output' }
  if (!target.inputs[inputPosition]) return { valid: false, reason: 'Invalid input position' }
  if (getDescendants(graph, targetNodeId).has(sourceNodeId)) {
    return { valid: false, reason: 'Would create a cycle' }
  }
  const compatibility = validateTensorCompatibility(
    source.outputMetadata?.[sourceOutputPosition],
    target.inputMetadata?.[inputPosition],
  )
  if (!compatibility.valid) return compatibility
  return { valid: true }
}

// inputPosition indexes targetNode.inputs. Replaces whatever currently feeds that
// position with sourceNode's (first real) output -- same position-based addressing
// as insertPassthroughNode, and same single-primary-output assumption already used
// by deleteNodeWithReconnect. Caller is expected to have checked validateRewire first.
export function rewireEdge(
  graph: SelectableGraph,
  sourceNodeId: string,
  targetNodeId: string,
  inputPosition: number,
): SelectableGraph {
  const source = graph.nodes.find((n) => n.id === sourceNodeId)
  const target = graph.nodes.find((n) => n.id === targetNodeId)
  if (!source || !target) return graph
  const sourceOutputPosition = source.outputs.findIndex((output) => output !== '')
  const newTensorName = source.outputs[sourceOutputPosition]
  const oldTensorName = target.inputs[inputPosition]
  if (!newTensorName || !oldTensorName || newTensorName === oldTensorName) return graph

  const rewiredTarget = { ...target, inputs: target.inputs.map((inp, i) => (i === inputPosition ? newTensorName : inp)) }
  const newEdge: OnnxGraph['edges'][number] = {
    id: `${sourceNodeId}->${targetNodeId}@${newTensorName}`,
    source: sourceNodeId,
    target: targetNodeId,
    label: newTensorName,
    shape: source.outputMetadata?.[sourceOutputPosition]?.shape,
  }

  return {
    ...graph,
    nodes: graph.nodes.map((n) => (n.id === targetNodeId ? rewiredTarget : n)),
    edges: graph.edges.filter((e) => !(e.target === targetNodeId && e.label === oldTensorName)).concat([newEdge]),
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
