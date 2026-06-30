import type { OnnxNode } from '../lib/onnxTypes'
import { formatShape } from '../lib/onnxProtoParser'

interface LayerInspectorProps {
  node: OnnxNode | null
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

export function LayerInspector({ node }: LayerInspectorProps) {
  if (!node) {
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

  const isCompute = node.opType !== 'Input' && node.opType !== 'Output'

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
      <Row label="OP TYPE" value={node.opType} />
      <Row label="PARAMETERS" value={node.paramCount.toLocaleString()} />
      <Row label="EST. SIZE" value={`${node.estimatedSizeMB.toFixed(3)} MB`} />

      {isCompute && node.paramCount > 0 && (
        <div style={{ ...rowStyle, alignItems: 'center' }}>
          <span style={labelStyle}>SENSITIVITY</span>
          <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: sensitivityColor(node.paramCount), letterSpacing: '0.04em' }}>
            {sensitivityLabel(node.paramCount)}
          </span>
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
