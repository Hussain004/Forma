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

export function computeOpCounts(nodes: OnnxNode[]): Record<string, number> {
  const counts: Record<string, number> = {}
  for (const n of nodes) {
    if (n.opType === 'Input' || n.opType === 'Output') continue
    counts[n.opType] = (counts[n.opType] ?? 0) + 1
  }
  return counts
}
