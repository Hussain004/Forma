import { useEffect, useMemo } from 'react'
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeProps,
} from '@xyflow/react'
import dagre from 'dagre'
import '@xyflow/react/dist/style.css'
import type { OnnxNode, OnnxEdge } from '../lib/onnxTypes'
import { validateEdges } from '../lib/graphUtils'

const NODE_WIDTH = 160
const NODE_HEIGHT = 64

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

type OperatorData = { opType: string; paramCount: number }
type IOData = { label: string }

function OperatorNode({ data, selected }: NodeProps<Node<OperatorData>>) {
  return (
    <div
      style={{
        width: NODE_WIDTH,
        minHeight: NODE_HEIGHT,
        background: '#16191C',
        border: selected ? '1px solid #FFB000' : '1px solid rgba(255,255,255,0.15)',
        borderRadius: 2,
        padding: '8px 12px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        gap: 4,
        fontFamily: 'var(--font-mono)',
      }}
    >
      <Handle type="target" position={Position.Top} style={handleStyle} />
      <div
        style={{
          color: '#E8EAF0',
          fontSize: 13,
          fontWeight: 500,
          letterSpacing: '0.04em',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      >
        {data.opType}
      </div>
      <div style={{ color: '#8A8F9E', fontSize: 11, letterSpacing: '0.06em' }}>
        {data.paramCount.toLocaleString()} PARAMS
      </div>
      <Handle type="source" position={Position.Bottom} style={handleStyle} />
    </div>
  )
}

function IONode({ data, selected }: NodeProps<Node<IOData>>) {
  return (
    <div
      style={{
        width: NODE_WIDTH,
        minHeight: NODE_HEIGHT,
        background: '#1C2128',
        border: selected ? '1px solid #FFB000' : '1px solid rgba(255,255,255,0.15)',
        borderLeft: '2px solid #FFB000',
        borderRadius: 2,
        padding: '8px 12px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        fontFamily: 'var(--font-mono)',
      }}
    >
      <Handle type="target" position={Position.Top} style={handleStyle} />
      <div
        style={{
          color: '#E8EAF0',
          fontSize: 13,
          fontWeight: 500,
          letterSpacing: '0.04em',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      >
        {data.label}
      </div>
      <Handle type="source" position={Position.Bottom} style={handleStyle} />
    </div>
  )
}

const nodeTypes = { operator: OperatorNode, io: IONode }

const IO_OPS = new Set(['Input', 'Output'])

interface GraphCanvasProps {
  onnxNodes: OnnxNode[]
  onnxEdges: OnnxEdge[]
  onNodeSelect: (nodeId: string) => void
}

function toFlowGraph(onnxNodes: OnnxNode[], onnxEdges: OnnxEdge[]): {
  nodes: Node[]
  edges: Edge[]
} {
  const rawNodes: Node[] = onnxNodes.map((n) => {
    const isIO = IO_OPS.has(n.opType)
    return {
      id: n.id,
      type: isIO ? 'io' : 'operator',
      position: { x: 0, y: 0 },
      data: isIO
        ? { label: n.outputs[0] ?? n.inputs[0] ?? n.opType }
        : { opType: n.opType, paramCount: n.paramCount },
    }
  })

  // Drop edges referencing nodes that don't exist so dagre/React Flow never see dangling refs.
  const invalidIds = new Set(
    validateEdges({
      nodes: onnxNodes,
      edges: onnxEdges,
      modelName: '',
      totalParams: 0,
      totalSizeMB: 0,
    }).map((e) => e.id),
  )

  const edges: Edge[] = onnxEdges
    .filter((e) => !invalidIds.has(e.id))
    .map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    label: e.label,
    style: { stroke: '#FFB000', strokeWidth: 1 },
  }))

  return { nodes: applyDagreLayout(rawNodes, edges), edges }
}

export function GraphCanvas({ onnxNodes, onnxEdges, onNodeSelect }: GraphCanvasProps) {
  const computed = useMemo(
    () => toFlowGraph(onnxNodes, onnxEdges),
    [onnxNodes, onnxEdges],
  )

  const [nodes, setNodes, onNodesChange] = useNodesState(computed.nodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(computed.edges)

  useEffect(() => {
    setNodes(computed.nodes)
    setEdges(computed.edges)
  }, [computed, setNodes, setEdges])

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) => onNodeSelect(node.id)}
        fitView
        proOptions={{ hideAttribution: true }}
        minZoom={0.2}
        maxZoom={2}
      >
        <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="#2A2F38" />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}
