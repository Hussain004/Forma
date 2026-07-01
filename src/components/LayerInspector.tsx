import type { OnnxNode } from '../lib/onnxTypes'
import { formatShape } from '../lib/onnxProtoParser'
import { opCategoryColor } from '../lib/graphUtils'

interface LayerInspectorProps {
  node: OnnxNode | null
  onToggleExclude?: (nodeId: string) => void
  quantizeEstimate?: { ratio: number } | null
  modelStats?: { opCounts: Record<string, number>; totalNodes: number; graphDepth?: number } | null
  multiSelection?: {
    nodes: OnnxNode[]
    totalParams: number
    totalSizeMB: number
  }
  onBulkExclude?: () => void
  onBulkInclude?: () => void
}

const bulkButtonStyle: React.CSSProperties = {
  background: 'none',
  border: '1px solid rgba(255,255,255,0.15)',
  borderRadius: 2,
  color: 'var(--text-dim)',
  fontFamily: 'var(--font-mono)',
  fontSize: 10,
  letterSpacing: '0.06em',
  padding: '2px 10px',
  cursor: 'pointer',
  textTransform: 'uppercase',
}

const CATEGORY_LEGEND: { name: string; color: string }[] = [
  { name: 'Convolution', color: '#C0392B' },
  { name: 'Activation', color: '#52C57A' },
  { name: 'Normalization', color: '#3498DB' },
  { name: 'Linear/MatMul', color: '#E67E22' },
  { name: 'Pooling', color: '#9B59B6' },
  { name: 'Reshape/Transpose', color: '#1ABC9C' },
  { name: 'Other', color: 'rgba(255,255,255,0.15)' },
]

const swatchStyle: React.CSSProperties = {
  width: 8,
  height: 8,
  borderRadius: 1,
  flexShrink: 0,
  display: 'inline-block',
}

const labelStyle: React.CSSProperties = {
  color: 'var(--text-secondary)',
  textTransform: 'uppercase',
  letterSpacing: '0.08em',
  fontSize: 11,
  minWidth: 96,
  flexShrink: 0,
}

const valueStyle: React.CSSProperties = {
  color: 'var(--text-primary)',
  fontSize: 12,
  wordBreak: 'break-word',
}

const rowStyle: React.CSSProperties = {
  display: 'flex',
  gap: 12,
  padding: '5px 0',
  borderBottom: '1px solid rgba(255,255,255,0.05)',
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div style={rowStyle}>
      <span style={labelStyle}>{label}</span>
      <span style={valueStyle}>{value}</span>
    </div>
  )
}

function sectionHeader(label: string) {
  return (
    <div
      style={{
        color: 'var(--text-dim)',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        fontSize: 10,
        margin: '16px 0 4px',
      }}
    >
      {label}
    </div>
  )
}

function sensitivityLabel(params: number): string {
  if (params > 10_000_000) return 'HIGH (>10M params)'
  if (params > 1_000_000)  return 'MEDIUM (>1M params)'
  if (params > 100_000)    return 'LOW (>100K params)'
  return 'MINIMAL'
}

function sensitivityColor(params: number): string {
  if (params > 10_000_000) return '#C0392B'
  if (params > 1_000_000)  return '#E67E22'
  if (params > 100_000)    return '#FFB000'
  return '#52C57A'
}

export function LayerInspector({ node, onToggleExclude, quantizeEstimate, modelStats, multiSelection, onBulkExclude, onBulkInclude }: LayerInspectorProps) {
  if (multiSelection && multiSelection.nodes.length > 1) {
    const opCounts: Record<string, number> = {}
    for (const n of multiSelection.nodes) {
      opCounts[n.opType] = (opCounts[n.opType] ?? 0) + 1
    }
    const sortedOps = Object.entries(opCounts).sort((a, b) => b[1] - a[1])
    return (
      <div
        style={{
          background: 'var(--bg-surface)',
          borderLeft: '2px solid #FFB000',
          padding: 16,
          height: '100%',
          minWidth: 260,
          overflowY: 'auto',
          boxSizing: 'border-box',
        }}
      >
        <div style={{ color: '#FFB000', fontFamily: 'var(--font-mono)', fontSize: 12, letterSpacing: '0.08em', marginBottom: 12, textTransform: 'uppercase' }}>
          {multiSelection.nodes.length} Nodes Selected
        </div>
        <div style={rowStyle}>
          <span style={labelStyle}>TOTAL PARAMS</span>
          <span style={valueStyle}>{multiSelection.totalParams.toLocaleString()}</span>
        </div>
        <div style={rowStyle}>
          <span style={labelStyle}>TOTAL SIZE</span>
          <span style={valueStyle}>{multiSelection.totalSizeMB.toFixed(2)} MB</span>
        </div>
        {sectionHeader('Op Types')}
        {sortedOps.map(([opType, count]) => (
          <div key={opType} style={rowStyle}>
            <span style={{ ...labelStyle, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ ...swatchStyle, background: opCategoryColor(opType) }} />
              {opType}
            </span>
            <span style={valueStyle}>{count}</span>
          </div>
        ))}
        <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
          <button onClick={onBulkExclude} style={bulkButtonStyle}>EXCLUDE ALL</button>
          <button onClick={onBulkInclude} style={bulkButtonStyle}>INCLUDE ALL</button>
        </div>
      </div>
    )
  }
  if (!node) {
    if (!modelStats) {
      return (
        <div
          style={{
            background: 'var(--bg-surface)',
            borderLeft: '2px solid rgba(255,255,255,0.1)',
            padding: 16,
            height: '100%',
            minWidth: 260,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <span
            style={{
              color: 'var(--text-secondary)',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
              fontSize: 12,
            }}
          >
            Select a node
          </span>
        </div>
      )
    }
    const sorted = Object.entries(modelStats.opCounts).sort((a, b) => b[1] - a[1])
    const presentColors = new Set(sorted.map(([opType]) => opCategoryColor(opType)))
    const legend = CATEGORY_LEGEND.filter((c) => presentColors.has(c.color))
    return (
      <div
        style={{
          background: 'var(--bg-surface)',
          borderLeft: '2px solid rgba(255,255,255,0.1)',
          padding: 16,
          height: '100%',
          minWidth: 260,
          overflowY: 'auto',
          boxSizing: 'border-box',
        }}
      >
        <div style={{ color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.1em', fontSize: 10, marginBottom: 12 }}>
          Model Summary
        </div>
        <div style={rowStyle}>
          <span style={labelStyle}>TOTAL NODES</span>
          <span style={valueStyle}>{modelStats.totalNodes.toLocaleString()}</span>
        </div>
        {modelStats.graphDepth !== undefined && (
          <div style={rowStyle}>
            <span style={labelStyle}>DEPTH</span>
            <span style={valueStyle}>{modelStats.graphDepth.toLocaleString()}</span>
          </div>
        )}
        <div style={{ color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.1em', fontSize: 10, margin: '16px 0 4px' }}>
          Op Types
        </div>
        {sorted.map(([opType, count]) => (
          <div key={opType} style={rowStyle}>
            <span style={{ ...labelStyle, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ ...swatchStyle, background: opCategoryColor(opType) }} />
              {opType}
            </span>
            <span style={valueStyle}>{count}</span>
          </div>
        ))}
        {legend.length > 0 && (
          <>
            <div style={{ color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.1em', fontSize: 10, margin: '16px 0 4px' }}>
              Categories
            </div>
            {legend.map((c) => (
              <div key={c.name} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0' }}>
                <span style={{ ...swatchStyle, background: c.color }} />
                <span style={{ color: 'var(--text-secondary)', fontSize: 11, letterSpacing: '0.04em' }}>{c.name}</span>
              </div>
            ))}
          </>
        )}
      </div>
    )
  }

  const isCompute = node.opType !== 'Input' && node.opType !== 'Output'

  const handleCopy = () => {
    const lines = [
      `Op Type:    ${node.opType}`,
      `Parameters: ${node.paramCount.toLocaleString()}`,
      `Size:       ${node.estimatedSizeMB.toFixed(3)} MB`,
    ]
    if (node.inputs.length > 0) lines.push(`Inputs:     ${node.inputs.join(', ')}`)
    if (node.outputs.length > 0) lines.push(`Outputs:    ${node.outputs.join(', ')}`)
    navigator.clipboard.writeText(lines.join('\n')).catch(() => {})
  }

  return (
    <div
      style={{
        background: 'var(--bg-surface)',
        borderLeft: '2px solid #FFB000',
        padding: 16,
        height: '100%',
        minWidth: 260,
        overflowY: 'auto',
        boxSizing: 'border-box',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
        <button
          onClick={handleCopy}
          style={{
            background: 'none',
            border: '1px solid rgba(255,176,0,0.4)',
            borderRadius: 2,
            color: '#FFB000',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            letterSpacing: '0.06em',
            padding: '2px 10px',
            cursor: 'pointer',
            textTransform: 'uppercase',
          }}
        >
          COPY
        </button>
      </div>
      <Row label="OP TYPE" value={node.opType} />
      <Row label="PARAMETERS" value={node.paramCount.toLocaleString()} />
      <Row label="EST. SIZE" value={`${node.estimatedSizeMB.toFixed(3)} MB`} />
      {node.estimatedSizeMB > 0 && quantizeEstimate && quantizeEstimate.ratio > 0 && (
        <div style={{ color: 'var(--text-dim)', fontSize: 10, padding: '2px 0 0 108px', letterSpacing: '0.04em' }}>
          {`INT8: ${(node.estimatedSizeMB / quantizeEstimate.ratio).toFixed(3)} MB`}
        </div>
      )}

      {isCompute && node.paramCount > 0 && (
        <div style={{ ...rowStyle, alignItems: 'center' }}>
          <span style={labelStyle}>SENSITIVITY</span>
          <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: sensitivityColor(node.paramCount), letterSpacing: '0.04em' }}>
            {sensitivityLabel(node.paramCount)}
          </span>
        </div>
      )}

      {isCompute && (
        <div style={{ ...rowStyle, alignItems: 'center' }}>
          <span style={labelStyle}>EXCLUDED</span>
          <button
            onClick={() => node && onToggleExclude?.(node.id)}
            style={{
              background: 'none',
              border: node.excluded ? '1px solid #FFB000' : '1px solid rgba(255,255,255,0.15)',
              borderRadius: 2,
              color: node.excluded ? '#FFB000' : 'var(--text-dim)',
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              letterSpacing: '0.06em',
              padding: '2px 10px',
              cursor: 'pointer',
              textTransform: 'uppercase',
            }}
          >
            {node.excluded ? 'YES' : 'NO'}
          </button>
        </div>
      )}

      {node.inputShapes && node.inputShapes.length > 0 && (
        <>
          {sectionHeader('Input shapes')}
          {node.inputShapes.map((shape, i) => (
            <Row key={i} label={node.inputs[i] ?? `input_${i}`} value={formatShape(shape) || 'unknown'} />
          ))}
        </>
      )}

      {node.outputShapes && node.outputShapes.length > 0 && (
        <>
          {sectionHeader('Output shapes')}
          {node.outputShapes.map((shape, i) => (
            <Row key={i} label={node.outputs[i] ?? `output_${i}`} value={formatShape(shape) || 'unknown'} />
          ))}
        </>
      )}

      {node.inputs.length > 0 && (
        <>
          {sectionHeader('Inputs')}
          {node.inputs.map((inp) => (
            <div key={inp} style={{ ...rowStyle, gap: 0 }}>
              <span style={{ ...valueStyle, fontSize: 11, color: 'var(--text-secondary)' }}>{inp}</span>
            </div>
          ))}
        </>
      )}

      {node.outputs.length > 0 && (
        <>
          {sectionHeader('Outputs')}
          {node.outputs.map((out) => (
            <div key={out} style={{ ...rowStyle, gap: 0 }}>
              <span style={{ ...valueStyle, fontSize: 11, color: 'var(--text-secondary)' }}>{out}</span>
            </div>
          ))}
        </>
      )}
    </div>
  )
}
