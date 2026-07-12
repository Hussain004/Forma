import { useEffect, useMemo, useState } from 'react'
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
} from '@xyflow/react'
import dagre from 'dagre'
import '@xyflow/react/dist/style.css'
import type { OnnxNode, OnnxEdge } from '../lib/onnxTypes'
import { formatShape } from '../lib/onnxProtoParser'
import { validateEdges, opCategoryColor, type SelectableNode } from '../lib/graphUtils'

const NODE_WIDTH = 180
const NODE_HEIGHT = 64

type TraceRole = 'ancestor' | 'descendant' | null

function traceAccent(role: TraceRole): string | null {
  if (role === 'ancestor') return '#3498DB'
  if (role === 'descendant') return '#52C57A'
  return null
}

function applyDagreLayout(nodes: Node[], edges: Edge[], layoutDir: 'TB' | 'LR' = 'TB'): Node[] {
  const g = new dagre.graphlib.Graph()
  g.setGraph({ rankdir: layoutDir ?? 'TB', ranksep: 48, nodesep: 24 })
  g.setDefaultEdgeLabel(() => ({}))
  nodes.forEach((n) => g.setNode(n.id, { width: NODE_WIDTH, height: NODE_HEIGHT }))
  edges.forEach((e) => g.setEdge(e.source, e.target))
  dagre.layout(g)
  return nodes.map((n) => {
    const pos = g.node(n.id)
    return { ...n, position: { x: pos.x - NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 } }
  })
}

const handleStyle = {
  width: 6,
  height: 6,
  borderRadius: 0,
  background: '#FFB000',
  border: 'none',
}

type OperatorData = { opType: string; paramCount: number; shapeLabel: string; dimmed: boolean; excluded: boolean; traceRole: TraceRole; traceActive: boolean; isModified: boolean; isSynthetic: boolean }
type IOData = { label: string; shapeLabel: string; dimmed: boolean; excluded: boolean; traceRole: TraceRole; traceActive: boolean }

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
  const opacity = data.excluded
    ? 0.5
    : data.dimmed
      ? 0.25
      : data.traceActive && !selected && data.traceRole === null
        ? 0.4
        : 1
  const border = data.excluded
    ? '1px solid rgba(255,255,255,0.08)'
    : '1px solid rgba(255,255,255,0.12)'
  const accent = data.excluded
    ? 'rgba(255,255,255,0.08)'
    : traceAccent(data.traceRole) ?? (selected ? '#FFB000' : opCategoryColor(data.opType))
  return (
    <div
      style={{
        position: 'relative',
        width: NODE_WIDTH,
        minHeight: NODE_HEIGHT,
        background: '#16191C',
        border,
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
      <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 4, background: accent, borderRadius: '2px 0 0 2px', transition: 'background-color 150ms ease' }} />
      {data.isModified && (
        <div style={{ position: 'absolute', top: 3, right: 3, fontSize: 7, letterSpacing: '0.08em', color: '#FFB000', background: 'rgba(255,176,0,0.12)', padding: '1px 4px', borderRadius: 1 }}>
          MOD
        </div>
      )}
      {data.isSynthetic && (
        <div style={{ position: 'absolute', top: 3, left: 7, fontSize: 7, letterSpacing: '0.08em', color: 'var(--color-green)', background: 'rgba(82,197,122,0.12)', padding: '1px 4px', borderRadius: 1 }}>
          NEW
        </div>
      )}
      {data.excluded && <div style={strikeStyle} />}
      <Handle type="target" position={Position.Top} style={handleStyle} />
      <div style={{ color: '#E8EAF0', fontSize: 13, fontWeight: 500, letterSpacing: '0.04em', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {data.opType}
      </div>
      <div style={{ color: '#8A8F9E', fontSize: 10, letterSpacing: '0.06em' }}>
        {data.paramCount > 0 ? data.paramCount.toLocaleString() + ' PARAMS' : 'NO PARAMS'}
      </div>
      {data.shapeLabel && (
        <div style={{ color: '#5A6070', fontSize: 9, letterSpacing: '0.04em', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {data.shapeLabel}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} style={handleStyle} />
    </div>
  )
}

function IONode({ data, selected }: NodeProps<Node<IOData>>) {
  const opacity = data.excluded
    ? 0.5
    : data.dimmed
      ? 0.25
      : data.traceActive && !selected && data.traceRole === null
        ? 0.4
        : 1
  const ioAccent = data.excluded
    ? 'rgba(255,255,255,0.08)'
    : traceAccent(data.traceRole) ?? '#FFB000'
  return (
    <div
      style={{
        position: 'relative',
        width: NODE_WIDTH,
        minHeight: NODE_HEIGHT,
        background: '#1C2128',
        border: data.excluded
          ? '1px solid rgba(255,255,255,0.08)'
          : selected ? '1px solid #FFB000' : '1px solid rgba(255,255,255,0.15)',
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
      <div style={{ color: '#E8EAF0', fontSize: 13, fontWeight: 500, letterSpacing: '0.04em', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {data.label}
      </div>
      {data.shapeLabel && (
        <div style={{ color: '#5A6070', fontSize: 9, letterSpacing: '0.04em' }}>
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
  onEdgeClick?: (targetNodeId: string, tensorName: string) => void
  jumpToNodeId?: string | null
  traceAncestors?: Set<string>
  traceDescendants?: Set<string>
  layoutDir?: 'TB' | 'LR'
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
    return {
      id: n.id,
      type: isIO ? 'io' : 'operator',
      position: { x: 0, y: 0 },
      selected: (n as SelectableNode).selected ?? (n.id === selectedNodeId),
      data: isIO
        ? { label: n.outputs[0] ?? n.inputs[0] ?? n.opType, shapeLabel, dimmed: n.dimmed ?? false, excluded: n.excluded ?? false, traceRole, traceActive }
        : { opType: n.opType, paramCount: n.paramCount, shapeLabel, dimmed: n.dimmed ?? false, excluded: n.excluded ?? false, traceRole, traceActive, isModified: n.isModified ?? false, isSynthetic: !ORIGINAL_NODE_ID_RE.test(n.id) },
    }
  })

  const nodeIdSet = new Set(onnxNodes.map(n => n.id))
  const invalidIds = new Set(
    validateEdges({ nodes: onnxNodes, edges: onnxEdges, modelName: '', totalParams: 0, totalSizeMB: 0 }).map(e => e.id),
  )
  const edges: Edge[] = onnxEdges
    .filter(e => !invalidIds.has(e.id) && nodeIdSet.has(e.source) && nodeIdSet.has(e.target))
    .map(e => {
      const adjacent = selectedNodeId && (e.source === selectedNodeId || e.target === selectedNodeId)
      const shapeLabel = adjacent && e.shape ? formatShape(e.shape) : undefined
      return {
        id: e.id,
        source: e.source,
        target: e.target,
        label: shapeLabel,
        labelStyle: { fontSize: 9, fontFamily: 'JetBrains Mono, monospace', fill: '#5A6070' },
        labelBgStyle: { fill: '#12161A', fillOpacity: 0.85 },
        labelBgPadding: [3, 4] as [number, number],
        style: { stroke: '#FFB000', strokeWidth: 1, cursor: 'pointer' },
      }
    })

  return { nodes: applyDagreLayout(rawNodes, edges, layoutDir), edges }
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

export function GraphCanvas({ onnxNodes, onnxEdges, selectedNodeId, onNodeSelect, onNodeCtrlClick, onEdgeClick, jumpToNodeId, traceAncestors = EMPTY_TRACE, traceDescendants = EMPTY_TRACE, layoutDir = 'TB' }: GraphCanvasProps) {
  const computed = useMemo(
    () => toFlowGraph(onnxNodes, onnxEdges, selectedNodeId, traceAncestors, traceDescendants, layoutDir),
    [onnxNodes, onnxEdges, selectedNodeId, traceAncestors, traceDescendants, layoutDir],
  )

  const [nodes, setNodes, onNodesChange] = useNodesState(computed.nodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(computed.edges)
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null)
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null)

  useEffect(() => {
    setNodes(computed.nodes)
    setEdges(computed.edges)
  }, [computed, setNodes, setEdges])

  const hoveredNode = hoveredNodeId ? onnxNodes.find((n) => n.id === hoveredNodeId) : null

  return (
    <div style={{ width: '100%', height: '100%' }} data-testid="graph-canvas">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        onNodeClick={(event, node) => {
          if (event.ctrlKey || event.metaKey) {
            onNodeCtrlClick?.(node.id)
          } else {
            onNodeSelect(node.id)
          }
        }}
        onEdgeClick={(_event, edge) => {
          // The React Flow edge's `label` field is repurposed to show the tensor
          // shape (only when adjacent to the selection), not the tensor name --
          // resolve the real name from the original OnnxEdge with the same id.
          const original = onnxEdges.find((e) => e.id === edge.id)
          if (original?.label) onEdgeClick?.(edge.target, original.label)
        }}
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
      >
        <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="#2A2F38" />
        <Controls
          showInteractive={false}
          style={{
            background: '#16191C',
            border: '1px solid rgba(255,255,255,0.12)',
            borderRadius: 2,
            boxShadow: 'none',
          }}
        />
        <MiniMap
          nodeColor={() => '#FFB000'}
          maskColor="rgba(18,22,26,0.8)"
          style={{
            background: '#12161A',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 2,
          }}
          pannable
          zoomable
        />
        <JumpController jumpToNodeId={jumpToNodeId} />
      </ReactFlow>
      {hoveredNode && tooltipPos && (
        <div
          style={{
            position: 'fixed',
            left: tooltipPos.x + 16,
            top: tooltipPos.y + 8,
            background: '#1C2128',
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
          <div style={{ color: '#FFB000', marginBottom: 2 }}>{hoveredNode.opType}</div>
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
