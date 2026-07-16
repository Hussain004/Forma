// Dagre layout in a worker, used by GraphCanvas for graphs above its
// large-graph threshold. dagre's cost grows superlinearly with node count --
// a ~1300-node BERT takes seconds, and running that synchronously (as the
// small-graph path still does, where it costs single-digit milliseconds)
// freezes the tab on every structural edit.

import dagre from 'dagre'

export interface LayoutRequest {
  requestId: number
  rankdir: 'TB' | 'LR'
  nodes: { id: string; width: number; height: number }[]
  edges: { source: string; target: string }[]
}

export interface LayoutResponse {
  requestId: number
  positions: Record<string, { x: number; y: number }>
  // Set when dagre threw. Known case: extremely deep graphs (a chain ~1000+
  // nodes long) overflow the call stack in a worker, whose stack is smaller
  // than the main thread's -- the same layout succeeds on the main thread,
  // so the caller falls back to its synchronous path.
  error?: string
}

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<LayoutRequest>) => void) | null
  postMessage: (message: LayoutResponse) => void
}

ctx.onmessage = (event: MessageEvent<LayoutRequest>) => {
  const { requestId, rankdir, nodes, edges } = event.data
  try {
    const g = new dagre.graphlib.Graph()
    g.setGraph({ rankdir, ranksep: 48, nodesep: 24 })
    g.setDefaultEdgeLabel(() => ({}))
    nodes.forEach((n) => g.setNode(n.id, { width: n.width, height: n.height }))
    edges.forEach((e) => g.setEdge(e.source, e.target))
    dagre.layout(g)

    const positions: Record<string, { x: number; y: number }> = {}
    nodes.forEach((n) => {
      const pos = g.node(n.id)
      positions[n.id] = { x: pos.x - n.width / 2, y: pos.y - n.height / 2 }
    })
    ctx.postMessage({ requestId, positions })
  } catch (err) {
    ctx.postMessage({ requestId, positions: {}, error: (err as Error).message })
  }
}
