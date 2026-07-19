import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
  useReactFlow,
  type Node,
  type Edge,
  type NodeProps,
  type ReactFlowInstance,
} from '@xyflow/react'
import dagre from 'dagre'
import '@xyflow/react/dist/style.css'
import type { OnnxNode, OnnxEdge } from '../lib/onnxTypes'
import type { LayoutRequest, LayoutResponse } from '../workers/layoutWorker'
import { formatShape } from '../lib/onnxProtoParser'
import { validateEdges, opCategoryColor, type SelectableNode } from '../lib/graphUtils'

const NODE_WIDTH = 180
const NODE_HEIGHT = 64

// Above this node count, dagre moves off the main thread (see layoutWorker.ts)
// and the large-model banner shows. Below it, layout stays synchronous -- at
// small scale dagre costs single-digit milliseconds and the async round-trip
// would only add flicker.
const LARGE_GRAPH_THRESHOLD = 500

type TraceRole = 'ancestor' | 'descendant' | null

function traceAccent(role: TraceRole): string | null {
  if (role === 'ancestor') return '#3498DB'
  if (role === 'descendant') return '#52C57A'
  return null
}

// manualPositions holds nodes the user placed by hand (addCustomNode) -- they're
// excluded from dagre's graph entirely (not registered as a node, and any edge
// touching one is skipped) so auto-layout for the rest of the graph proceeds
// exactly as if the manually-placed node didn't exist, and the node itself keeps
// the position it was dropped at instead of being reshuffled on every re-layout.
// React Flow still draws edges to/from it correctly regardless, since edge
// rendering only depends on each node's final resolved position.
function applyDagreLayout(
  nodes: Node[],
  edges: Edge[],
  layoutDir: 'TB' | 'LR' = 'TB',
  manualPositions: Map<string, { x: number; y: number }>,
): Node[] {
  const g = new dagre.graphlib.Graph()
  g.setGraph({ rankdir: layoutDir ?? 'TB', ranksep: 48, nodesep: 24 })
  g.setDefaultEdgeLabel(() => ({}))
  nodes.forEach((n) => { if (!manualPositions.has(n.id)) g.setNode(n.id, { width: NODE_WIDTH, height: NODE_HEIGHT }) })
  edges.forEach((e) => { if (!manualPositions.has(e.source) && !manualPositions.has(e.target)) g.setEdge(e.source, e.target) })
  dagre.layout(g)
  return nodes.map((n) => {
    const manual = manualPositions.get(n.id)
    if (manual) return { ...n, position: manual }
    const pos = g.node(n.id)
    return { ...n, position: { x: pos.x - NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 } }
  })
}

const handleStyle = {
  width: 6,
  height: 6,
  borderRadius: 0,
  background: 'var(--color-amber)',
  border: 'none',
}

type OperatorData = { opType: string; paramCount: number; shapeLabel: string; dimmed: boolean; excluded: boolean; traceRole: TraceRole; traceActive: boolean; isModified: boolean; isSynthetic: boolean; isDeleted: boolean; inputCount: number }
type IOData = { label: string; shapeLabel: string; dimmed: boolean; excluded: boolean; traceRole: TraceRole; traceActive: boolean; isDeleted: boolean }

const strikeStyle: React.CSSProperties = {
  position: 'absolute',
  left: 0,
  right: 0,
  top: '50%',
  height: 1,
  background: '#FFFFFF',
  pointerEvents: 'none',
}

function OperatorNode({ data, selected }: NodeProps<Node<OperatorData>>) {
  const opacity = data.isDeleted
    ? 0.35
    : data.excluded
    ? 0.5
    : data.dimmed
      ? 0.25
      : data.traceActive && !selected && data.traceRole === null
        ? 0.4
        : 1
  const border = data.isDeleted
    ? '1px dashed rgba(192,57,43,0.7)'
    : data.excluded
    ? '1px solid rgba(255,255,255,0.08)'
    : '1px solid rgba(255,255,255,0.12)'
  const accent = data.isDeleted
    ? 'var(--color-error)'
    : data.excluded
    ? 'rgba(255,255,255,0.08)'
    : traceAccent(data.traceRole) ?? (selected ? 'var(--color-amber)' : opCategoryColor(data.opType))
  // Badges are absolutely positioned in the top corners; without reserved
  // headroom they sit directly on the op-type title (centered content leaves
  // only ~8px of slack, less than a badge's height).
  const hasBadge = data.isModified || data.isSynthetic || data.isDeleted
  return (
    <div
      style={{
        position: 'relative',
        width: NODE_WIDTH,
        minHeight: NODE_HEIGHT,
        background: data.isDeleted ? 'transparent' : 'var(--bg-surface)',
        border,
        borderRadius: 2,
        padding: hasBadge ? '24px 12px 8px' : '8px 12px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        gap: 2,
        fontFamily: 'var(--font-mono)',
        opacity,
        transition: 'opacity 150ms ease, border-color 150ms ease',
      }}
    >
      <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 4, background: accent, borderRadius: '2px 0 0 2px', transition: 'background-color 150ms ease' }} />
      {data.isModified && (
        <div style={{ position: 'absolute', top: 5, right: 5, fontSize: 9, letterSpacing: '0.08em', color: 'var(--color-amber)', background: 'rgba(255,176,0,0.12)', padding: '1px 4px', borderRadius: 1 }}>
          MOD
        </div>
      )}
      {data.isSynthetic && (
        <div style={{ position: 'absolute', top: 5, left: 7, fontSize: 9, letterSpacing: '0.08em', color: 'var(--color-green)', background: 'rgba(82,197,122,0.12)', padding: '1px 4px', borderRadius: 1 }}>
          NEW
        </div>
      )}
      {data.isDeleted && (
        <div data-testid="deleted-node-badge" style={{ position: 'absolute', top: 5, right: 5, fontSize: 9, letterSpacing: '0.08em', color: 'var(--color-error)', background: 'rgba(192,57,43,0.12)', padding: '1px 4px', borderRadius: 1 }}>
          DEL
        </div>
      )}
      {data.excluded && <div style={strikeStyle} />}
      {/* One target handle per input position (not one shared handle) -- lets a
          drag-to-rewire connection specify exactly which input slot it replaces,
          the same way an existing edge already identifies its slot by tensor name. */}
      {Array.from({ length: Math.max(data.inputCount, 1) }, (_, i) => (
        <Handle
          key={i}
          id={`in-${i}`}
          type="target"
          position={Position.Top}
          style={{ ...handleStyle, left: `${((i + 1) / (Math.max(data.inputCount, 1) + 1)) * 100}%` }}
        />
      ))}
      <div style={{ color: 'var(--text-primary)', fontSize: 13, fontWeight: 500, letterSpacing: '0.04em', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {data.opType}
      </div>
      <div style={{ color: 'var(--text-secondary)', fontSize: 10, letterSpacing: '0.06em' }}>
        {data.paramCount > 0 ? data.paramCount.toLocaleString() + ' PARAMS' : 'NO PARAMS'}
      </div>
      {data.shapeLabel && (
        <div style={{ color: 'var(--text-dim)', fontSize: 10, letterSpacing: '0.04em', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {data.shapeLabel}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} style={handleStyle} />
    </div>
  )
}

function IONode({ data, selected }: NodeProps<Node<IOData>>) {
  const opacity = data.isDeleted
    ? 0.35
    : data.excluded
    ? 0.5
    : data.dimmed
      ? 0.25
      : data.traceActive && !selected && data.traceRole === null
        ? 0.4
        : 1
  const ioAccent = data.excluded
    ? 'rgba(255,255,255,0.08)'
    : traceAccent(data.traceRole) ?? 'var(--color-amber)'
  return (
    <div
      style={{
        position: 'relative',
        width: NODE_WIDTH,
        minHeight: NODE_HEIGHT,
        background: data.isDeleted ? 'transparent' : 'var(--bg-raised)',
        border: data.excluded
          ? '1px solid rgba(255,255,255,0.08)'
          : selected ? '1px solid var(--color-amber)' : '1px solid rgba(255,255,255,0.15)',
        borderLeft: data.excluded ? '1px solid rgba(255,255,255,0.08)' : `2px solid ${ioAccent}`,
        borderRadius: 2,
        padding: '8px 12px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        gap: 2,
        fontFamily: 'var(--font-mono)',
        opacity,
        transition: 'opacity 150ms ease, border-color 150ms ease',
      }}
    >
      {data.excluded && <div style={strikeStyle} />}
      <Handle type="target" position={Position.Top} style={handleStyle} />
      <div style={{ color: 'var(--text-primary)', fontSize: 13, fontWeight: 500, letterSpacing: '0.04em', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {data.label}
      </div>
      {data.shapeLabel && (
        <div style={{ color: 'var(--text-dim)', fontSize: 10, letterSpacing: '0.04em' }}>
          {data.shapeLabel}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} style={handleStyle} />
    </div>
  )
}

const nodeTypes = { operator: OperatorNode, io: IONode }
const IO_OPS = new Set(['Input', 'Output'])

interface GraphCanvasProps {
  onnxNodes: OnnxNode[]
  onnxEdges: OnnxEdge[]
  selectedNodeId: string | null
  onNodeSelect: (nodeId: string) => void
  onNodeCtrlClick?: (nodeId: string) => void
  onBoxSelect?: (nodeIds: string[]) => void
  onEdgeClick?: (targetNodeId: string, tensorName: string) => void
  onRewire?: (sourceNodeId: string, targetNodeId: string, inputPosition: number) => void
  pendingNodeType?: { opType: string; inputCount: number } | null
  onPlaceNode?: (position: { x: number; y: number }) => void
  jumpToNodeId?: string | null
  traceAncestors?: Set<string>
  traceDescendants?: Set<string>
  layoutDir?: 'TB' | 'LR'
  diffActive?: boolean
}

// Nodes parsed directly from the loaded model use the `node_<idx>_<opType>` id
// scheme; a passthrough Identity node inserted via structural editing does not,
// which is what marks it as generated rather than part of the original file.
const ORIGINAL_NODE_ID_RE = /^node_\d+_/

function toFlowGraph(
  onnxNodes: OnnxNode[],
  onnxEdges: OnnxEdge[],
  selectedNodeId: string | null,
  traceAncestors: Set<string>,
  traceDescendants: Set<string>,
  layoutDir: 'TB' | 'LR' = 'TB',
  // 'deferred' skips the synchronous dagre pass -- positions arrive later
  // from layoutWorker.ts. Manual positions still apply immediately.
  layoutMode: 'sync' | 'deferred' = 'sync',
) {
  const traceActive = selectedNodeId !== null && (traceAncestors.size > 0 || traceDescendants.size > 0)
  const rawNodes: Node[] = onnxNodes.map((n) => {
    const isIO = IO_OPS.has(n.opType)
    const shapeLabel = isIO
      ? formatShape(n.outputShapes?.[0] ?? n.inputShapes?.[0])
      : formatShape(n.outputShapes?.[0])
    const traceRole: TraceRole = traceAncestors.has(n.id)
      ? 'ancestor'
      : traceDescendants.has(n.id)
        ? 'descendant'
        : null
    const label = isIO ? (n.outputs[0] ?? n.inputs[0] ?? n.opType) : n.opType
    const ariaLabel = isIO
      ? `${n.opType === 'Input' ? 'Input' : 'Output'} ${label}${shapeLabel ? `, shape ${shapeLabel}` : ''}`
      : `${n.opType}${n.paramCount > 0 ? `, ${n.paramCount.toLocaleString()} params` : ', no params'}${n.excluded ? ', excluded from stats' : ''}`
    return {
      id: n.id,
      type: isIO ? 'io' : 'operator',
      draggable: n.diffStatus !== 'deleted',
      connectable: n.diffStatus !== 'deleted',
      selectable: n.diffStatus !== 'deleted',
      focusable: n.diffStatus !== 'deleted',
      position: { x: 0, y: 0 },
      selected: (n as SelectableNode).selected ?? (n.id === selectedNodeId),
      ariaLabel,
      data: isIO
        ? { label, shapeLabel, dimmed: n.dimmed ?? false, excluded: n.excluded ?? false, traceRole, traceActive, isDeleted: n.diffStatus === 'deleted' }
        : { opType: n.opType, paramCount: n.paramCount, shapeLabel, dimmed: n.dimmed ?? false, excluded: n.excluded ?? false, traceRole, traceActive, isModified: n.isModified ?? false, isSynthetic: !ORIGINAL_NODE_ID_RE.test(n.id), isDeleted: n.diffStatus === 'deleted', inputCount: n.inputs.length },
    }
  })

  const nodeById = new Map(onnxNodes.map(n => [n.id, n]))
  const nodeIdSet = new Set(onnxNodes.map(n => n.id))
  const invalidIds = new Set(
    validateEdges({ nodes: onnxNodes, edges: onnxEdges, modelName: '', totalParams: 0, totalSizeMB: 0 }).map(e => e.id),
  )
  const edges: Edge[] = onnxEdges
    .filter(e => !invalidIds.has(e.id) && nodeIdSet.has(e.source) && nodeIdSet.has(e.target))
    .map(e => {
      const adjacent = selectedNodeId && (e.source === selectedNodeId || e.target === selectedNodeId)
      const shapeLabel = adjacent && e.shape ? formatShape(e.shape) : undefined
      const inputPosition = e.label ? nodeById.get(e.target)?.inputs.indexOf(e.label) ?? -1 : -1
      const stroke = e.diffStatus === 'changed'
        ? '#3498DB'
        : e.diffStatus === 'removed' ? 'rgba(192,57,43,0.6)' : 'var(--color-amber)'
      return {
        id: e.id,
        source: e.source,
        target: e.target,
        targetHandle: inputPosition >= 0 ? `in-${inputPosition}` : undefined,
        label: shapeLabel,
        labelStyle: { fontSize: 10, fontFamily: 'JetBrains Mono, monospace', fill: 'var(--text-dim)' },
        labelBgStyle: { fill: 'var(--bg-base)', fillOpacity: 0.85 },
        labelBgPadding: [3, 4] as [number, number],
        style: { stroke, strokeWidth: e.diffStatus === 'changed' ? 2 : 1, strokeDasharray: e.diffStatus === 'removed' ? '5 4' : undefined, cursor: e.diffStatus === 'removed' ? 'default' : 'pointer' },
        zIndex: e.diffStatus === 'removed' ? 0 : 1,
        data: { diffStatus: e.diffStatus },
      }
    })

  const manualPositions = new Map<string, { x: number; y: number }>()
  for (const n of onnxNodes) {
    if (n.position) manualPositions.set(n.id, n.position)
  }

  if (layoutMode === 'deferred') {
    const nodes = rawNodes.map((n) => {
      const manual = manualPositions.get(n.id)
      return manual ? { ...n, position: manual } : n
    })
    return { nodes, edges, manualPositions }
  }
  return { nodes: applyDagreLayout(rawNodes, edges, layoutDir, manualPositions), edges, manualPositions }
}

function JumpController({ jumpToNodeId }: { jumpToNodeId?: string | null }) {
  const { fitView } = useReactFlow()
  useEffect(() => {
    if (!jumpToNodeId) return
    fitView({ nodes: [{ id: jumpToNodeId }], duration: 400, padding: 0.5 })
  }, [jumpToNodeId, fitView])
  return null
}

const EMPTY_TRACE: Set<string> = new Set()

export function GraphCanvas({ onnxNodes, onnxEdges, selectedNodeId, onNodeSelect, onNodeCtrlClick, onBoxSelect, onEdgeClick, onRewire, pendingNodeType, onPlaceNode, jumpToNodeId, traceAncestors = EMPTY_TRACE, traceDescendants = EMPTY_TRACE, layoutDir = 'TB', diffActive = false }: GraphCanvasProps) {
  const isLargeGraph = onnxNodes.length > LARGE_GRAPH_THRESHOLD
  const computed = useMemo(
    () => toFlowGraph(onnxNodes, onnxEdges, selectedNodeId, traceAncestors, traceDescendants, layoutDir, isLargeGraph ? 'deferred' : 'sync'),
    [onnxNodes, onnxEdges, selectedNodeId, traceAncestors, traceDescendants, layoutDir, isLargeGraph],
  )

  const [nodes, setNodes, onNodesChange] = useNodesState(computed.nodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(computed.edges)
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null)
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null)
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null)
  // Tracks canvas zoom so the placement ghost can match the size the node will
  // actually render at -- without this it stays a fixed screen size while the
  // placed node scales with the viewport, so the two visibly disagree.
  const [zoom, setZoom] = useState(1)
  // A plain click on an edge used to insert a passthrough node immediately --
  // no confirmation, no indication edges were even clickable. Now a click
  // opens this popover; only its own button commits the insert.
  const [edgePopover, setEdgePopover] = useState<{ edgeId: string; targetNodeId: string; tensorName: string; x: number; y: number } | null>(null)
  // ReactFlowInstance.screenToFlowPosition is only reachable via useReactFlow(),
  // which requires being a descendant of <ReactFlow>'s own provider (see
  // JumpController above) -- GraphCanvas itself is the ancestor that renders
  // <ReactFlow>, so onInit's instance callback is the way to reach it from here.
  const reactFlowInstanceRef = useRef<ReactFlowInstance | null>(null)
  // Lazily created on the first large graph, never for small ones -- besides
  // being pointless there, tests stub the global Worker with a single mock
  // shared with the ONNX worker, so an unconditional second Worker would
  // clobber that mock's message handler.
  const layoutWorkerRef = useRef<Worker | null>(null)
  const layoutRequestIdRef = useRef(0)
  const [layoutPending, setLayoutPending] = useState(false)

  useEffect(() => () => { layoutWorkerRef.current?.terminate() }, [])

  useEffect(() => {
    setNodes(computed.nodes)
    setEdges(computed.edges)
    if (!isLargeGraph) return

    if (!layoutWorkerRef.current) {
      layoutWorkerRef.current = new Worker(new URL('../workers/layoutWorker.ts', import.meta.url), { type: 'module' })
    }
    const worker = layoutWorkerRef.current
    const requestId = ++layoutRequestIdRef.current
    setLayoutPending(true)
    const manual = computed.manualPositions
    // Extremely deep graphs (chains ~1000+ long) overflow dagre's recursion
    // in a worker, whose stack is smaller than the main thread's -- the same
    // layout succeeds synchronously. One freeze is the correct trade against
    // an unpositioned pile of nodes.
    const syncFallback = () => {
      if (requestId !== layoutRequestIdRef.current) return
      setNodes(applyDagreLayout(computed.nodes, computed.edges, layoutDir, manual))
      setLayoutPending(false)
      requestAnimationFrame(() => reactFlowInstanceRef.current?.fitView({ padding: 0.1 }))
    }
    worker.onerror = syncFallback
    worker.onmessage = (event: MessageEvent<LayoutResponse>) => {
      // A newer request may have been posted while this one computed
      // (structural edit, layout toggle) -- only the latest response counts.
      if (event.data.requestId !== layoutRequestIdRef.current) return
      if (event.data.error) {
        syncFallback()
        return
      }
      const { positions } = event.data
      setNodes((nds) => nds.map((n) => (positions[n.id] ? { ...n, position: positions[n.id] } : n)))
      setLayoutPending(false)
      requestAnimationFrame(() => reactFlowInstanceRef.current?.fitView({ padding: 0.1 }))
    }
    worker.postMessage({
      requestId,
      rankdir: layoutDir,
      nodes: computed.nodes.filter((n) => !manual.has(n.id)).map((n) => ({ id: n.id, width: NODE_WIDTH, height: NODE_HEIGHT })),
      edges: computed.edges.filter((e) => !manual.has(e.source) && !manual.has(e.target)).map((e) => ({ source: e.source, target: e.target })),
    } satisfies LayoutRequest)
  }, [computed, setNodes, setEdges, isLargeGraph, layoutDir])

  const hoveredNode = hoveredNodeId ? onnxNodes.find((n) => n.id === hoveredNodeId) : null

  useEffect(() => {
    if (!edgePopover) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setEdgePopover(null)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [edgePopover])

  // Thickens the edge the popover is currently open for so the selection
  // reads as deliberate rather than just another same-colored line.
  const displayEdges = edgePopover
    ? edges.map((e) => (e.id === edgePopover.edgeId ? { ...e, style: { ...e.style, strokeWidth: 2.5 } } : e))
    : edges

  return (
    <div
      style={{ position: 'relative', width: '100%', height: '100%', cursor: pendingNodeType ? 'crosshair' : undefined }}
      data-testid="graph-canvas"
      onMouseMove={pendingNodeType ? (event) => setCursorPos({ x: event.clientX, y: event.clientY }) : undefined}
    >
      {isLargeGraph && (
        <div
          data-testid="large-graph-banner"
          style={{
            position: 'absolute',
            top: 8,
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 1200,
            padding: '3px 12px',
            background: 'var(--bg-raised)',
            border: '1px solid rgba(255,255,255,0.12)',
            borderRadius: 2,
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            whiteSpace: 'nowrap',
            color: layoutPending ? 'var(--color-amber)' : 'var(--text-dim)',
            pointerEvents: 'none',
          }}
        >
          {layoutPending ? 'Computing layout...' : `Large model: ${onnxNodes.length.toLocaleString()} nodes -- layout runs in the background`}
        </div>
      )}
      <ReactFlow
        nodes={nodes}
        edges={displayEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        onInit={(instance) => { reactFlowInstanceRef.current = instance }}
        onViewportChange={(viewport) => setZoom(viewport.zoom)}
        // Shift+drag box-select works already (react-flow's own default:
        // holding selectionKeyCode disables panOnDrag and turns the drag into
        // a selection box) -- this just syncs the result into App's selection
        // state. onSelectionEnd (not onSelectionChange) deliberately: it only
        // fires for the drag-box gesture, not for ordinary clicks, so it can't
        // fight the existing onNodeClick-driven select/ctrl-select path.
        onSelectionEnd={() => {
          const current = reactFlowInstanceRef.current?.getNodes() ?? []
          onBoxSelect?.(current.filter((n) => n.selected).map((n) => n.id))
        }}
        onNodeClick={(event, node) => {
          setEdgePopover(null)
          if (event.ctrlKey || event.metaKey) {
            onNodeCtrlClick?.(node.id)
          } else {
            onNodeSelect(node.id)
          }
        }}
        onEdgeClick={(event, edge) => {
          // The React Flow edge's `label` field is repurposed to show the tensor
          // shape (only when adjacent to the selection), not the tensor name --
          // resolve the real name from the original OnnxEdge with the same id.
          const original = onnxEdges.find((e) => e.id === edge.id)
          if (original?.diffStatus === 'removed') return
          if (!original?.label) return
          setEdgePopover({ edgeId: edge.id, targetNodeId: edge.target, tensorName: original.label, x: event.clientX, y: event.clientY })
        }}
        onConnect={(connection) => {
          const match = connection.targetHandle ? /^in-(\d+)$/.exec(connection.targetHandle) : null
          if (!match || !connection.source || !connection.target) return
          onRewire?.(connection.source, connection.target, Number(match[1]))
        }}
        onPaneClick={(event) => {
          setEdgePopover(null)
          if (!pendingNodeType) return
          const flowPos = reactFlowInstanceRef.current?.screenToFlowPosition({ x: event.clientX, y: event.clientY })
          if (!flowPos) return
          onPlaceNode?.({ x: flowPos.x - NODE_WIDTH / 2, y: flowPos.y - NODE_HEIGHT / 2 })
        }}
        connectionLineStyle={{ stroke: 'var(--color-amber)', strokeWidth: 1 }}
        onNodeMouseEnter={(event, node) => {
          setHoveredNodeId(node.id)
          setTooltipPos({ x: event.clientX, y: event.clientY })
        }}
        onNodeMouseLeave={() => {
          setHoveredNodeId(null)
          setTooltipPos(null)
        }}
        fitView
        proOptions={{ hideAttribution: true }}
        minZoom={0.1}
        maxZoom={3}
        // Skip DOM nodes for elements outside the viewport once a graph gets big
        // enough for it to matter -- gated by count (not always-on) because
        // jsdom's zero-size bounding rects in tests would otherwise cull every
        // node in every small test fixture.
        onlyRenderVisibleElements={onnxNodes.length > 300}
      >
        <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="#2A2F38" />
        <Controls showInteractive={false} />
        <MiniMap
          nodeColor={(node) => (node.type === 'operator' ? opCategoryColor((node.data as OperatorData).opType) : 'var(--color-amber)')}
          maskColor="rgba(18,22,26,0.8)"
          style={{
            background: 'var(--bg-base)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 2,
          }}
          pannable
          zoomable
        />
        <JumpController jumpToNodeId={jumpToNodeId} />
      </ReactFlow>
      {diffActive && (
        <div data-testid="diff-legend" style={{ position: 'absolute', top: 8, left: 8, zIndex: 1200, display: 'flex', alignItems: 'center', gap: 12, padding: '4px 8px', background: 'var(--bg-raised)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 2, fontFamily: 'var(--font-mono)', fontSize: 9, letterSpacing: '0.06em', textTransform: 'uppercase', pointerEvents: 'none' }}>
          <span style={{ color: 'var(--color-error)' }}>DEL</span>
          <span style={{ color: 'var(--color-amber)' }}>MOD</span>
          <span style={{ color: 'var(--color-green)' }}>NEW</span>
          <span style={{ color: '#3498DB' }}>Changed edge</span>
          <span style={{ color: 'rgba(192,57,43,0.8)' }}>Original edge</span>
        </div>
      )}
      {edgePopover && (
        <div
          data-testid="edge-insert-popover"
          style={{
            position: 'fixed',
            left: edgePopover.x + 10,
            top: edgePopover.y + 10,
            background: 'var(--bg-raised)',
            border: '1px solid rgba(255,176,0,0.3)',
            borderRadius: 2,
            padding: 10,
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            fontFamily: 'var(--font-mono)',
            zIndex: 1500,
          }}
        >
          <span style={{ color: 'var(--text-secondary)', fontSize: 11, letterSpacing: '0.02em', maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {edgePopover.tensorName}
          </span>
          <button
            data-testid="edge-insert-confirm"
            onClick={() => {
              onEdgeClick?.(edgePopover.targetNodeId, edgePopover.tensorName)
              setEdgePopover(null)
            }}
            className="btn-bar btn-primary"
            style={{ fontSize: 10, padding: '3px 8px', whiteSpace: 'nowrap' }}
          >
            Insert passthrough
          </button>
        </div>
      )}
      {pendingNodeType && cursorPos && (
        <div
          data-testid="add-node-ghost"
          style={{
            position: 'fixed',
            left: cursorPos.x - NODE_WIDTH / 2,
            top: cursorPos.y - NODE_HEIGHT / 2,
            width: NODE_WIDTH,
            minHeight: NODE_HEIGHT,
            transform: `scale(${zoom})`,
            transformOrigin: 'center center',
            background: 'var(--bg-surface)',
            border: '1px dashed rgba(255,176,0,0.6)',
            borderRadius: 2,
            padding: '8px 12px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            gap: 2,
            opacity: 0.6,
            pointerEvents: 'none',
            zIndex: 9999,
            fontFamily: 'var(--font-mono)',
          }}
        >
          <div style={{ color: 'var(--color-amber)', fontSize: 13, fontWeight: 500, letterSpacing: '0.04em' }}>
            {pendingNodeType.opType}
          </div>
          <div style={{ color: 'var(--text-dim)', fontSize: 9, letterSpacing: '0.06em' }}>
            CLICK TO PLACE / ESC TO CANCEL
          </div>
        </div>
      )}
      {hoveredNode && tooltipPos && (
        <div
          style={{
            position: 'fixed',
            left: tooltipPos.x + 16,
            top: tooltipPos.y + 8,
            background: 'var(--bg-raised)',
            border: '1px solid rgba(255,255,255,0.15)',
            borderRadius: 2,
            padding: '8px 12px',
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--text-primary)',
            pointerEvents: 'none',
            zIndex: 9999,
            minWidth: 160,
            letterSpacing: '0.04em',
          }}
        >
          <div style={{ color: 'var(--color-amber)', marginBottom: 2 }}>{hoveredNode.opType}</div>
          {hoveredNode.name && (
            <div style={{ color: 'var(--text-dim)', fontSize: 9, letterSpacing: '0.04em', marginBottom: 4, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {hoveredNode.name}
            </div>
          )}
          {hoveredNode.paramCount > 0 && (
            <div style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
              {hoveredNode.paramCount.toLocaleString()} params
            </div>
          )}
          {hoveredNode.outputShapes?.[0] && (
            <div style={{ color: 'var(--text-dim)', fontSize: 10, marginTop: 2 }}>
              {formatShape(hoveredNode.outputShapes[0])}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
