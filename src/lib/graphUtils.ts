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
      const matches = n.opType.toLowerCase().includes(q) || n.id.toLowerCase().includes(q)
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
