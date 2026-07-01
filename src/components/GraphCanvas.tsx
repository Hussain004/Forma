import { useEffect, useMemo } from 'react'
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
import { validateEdges, opCategoryColor } from '../lib/graphUtils'

const NODE_WIDTH = 180
const NODE_HEIGHT = 64

type TraceRole = 'ancestor' | 'descendant' | null

function traceAccent(role: TraceRole): string | null {
  if (role === 'ancestor') return '#3498DB'
  if (role === 'descendant') return '#52C57A'
  return null
}

function applyDagreLayout(nodes: Node[], edges: Edge[]): Node[] {
  const g = new dagre.graphlib.Graph()
  g.setGraph({ rankdir: 'TB', ranksep: 48, nodesep: 24 })
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

type OperatorData = { opType: string; paramCount: number; shapeLabel: string; dimmed: boolean; excluded: boolean; traceRole: TraceRole; traceActive: boolean }
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
      }}
    >
      <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 4, background: accent, borderRadius: '2px 0 0 2px' }} />
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
  jumpToNodeId?: string | null
  traceAncestors?: Set<string>
  traceDescendants?: Set<string>
}

function toFlowGraph(
  onnxNodes: OnnxNode[],
  onnxEdges: OnnxEdge[],
  selectedNodeId: string | null,
  traceAncestors: Set<string>,
  traceDescendants: Set<string>,
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
      selected: n.id === selectedNodeId,
      data: isIO
        ? { label: n.outputs[0] ?? n.inputs[0] ?? n.opType, shapeLabel, dimmed: n.dimmed ?? false, excluded: n.excluded ?? false, traceRole, traceActive }
        : { opType: n.opType, paramCount: n.paramCount, shapeLabel, dimmed: n.dimmed ?? false, excluded: n.excluded ?? false, traceRole, traceActive },
    }
  })

  const nodeIdSet = new Set(onnxNodes.map(n => n.id))
  const invalidIds = new Set(
    validateEdges({ nodes: onnxNodes, edges: onnxEdges, modelName: '', totalParams: 0, totalSizeMB: 0 }).map(e => e.id),
  )
  const edges: Edge[] = onnxEdges
    .filter(e => !invalidIds.has(e.id) && nodeIdSet.has(e.source) && nodeIdSet.has(e.target))
    .map(e => ({
      id: e.id,
      source: e.source,
      target: e.target,
      style: { stroke: '#FFB000', strokeWidth: 1 },
    }))

  return { nodes: applyDagreLayout(rawNodes, edges), edges }
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

export function GraphCanvas({ onnxNodes, onnxEdges, selectedNodeId, onNodeSelect, jumpToNodeId, traceAncestors = EMPTY_TRACE, traceDescendants = EMPTY_TRACE }: GraphCanvasProps) {
  const computed = useMemo(
    () => toFlowGraph(onnxNodes, onnxEdges, selectedNodeId, traceAncestors, traceDescendants),
    [onnxNodes, onnxEdges, selectedNodeId, traceAncestors, traceDescendants],
  )

  const [nodes, setNodes, onNodesChange] = useNodesState(computed.nodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(computed.edges)

  useEffect(() => {
    setNodes(computed.nodes)
    setEdges(computed.edges)
  }, [computed, setNodes, setEdges])

  return (
    <div style={{ width: '100%', height: '100%' }} data-testid="graph-canvas">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) => onNodeSelect(node.id)}
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
    </div>
  )
}
