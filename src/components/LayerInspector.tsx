import type { OnnxNode } from '../lib/onnxTypes'

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
  padding: '4px 0',
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

export function LayerInspector({ node }: LayerInspectorProps) {
  if (!node) {
    return (
      <div
        style={{
          background: 'var(--bg-surface)',
          borderLeft: '2px solid rgba(255,255,255,0.1)',
          padding: 16,
          height: '100%',
          minWidth: 240,
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

  const attrEntries = Object.entries(node.attributes)

  return (
    <div
      style={{
        background: 'var(--bg-surface)',
        borderLeft: '2px solid #FFB000',
        padding: 16,
        height: '100%',
        minWidth: 240,
        overflowY: 'auto',
      }}
    >
      <Row label="OP TYPE" value={node.opType} />
      <Row label="INPUTS" value={node.inputs.length ? node.inputs.join(', ') : '--'} />
      <Row label="OUTPUTS" value={node.outputs.length ? node.outputs.join(', ') : '--'} />
      <Row label="PARAMETERS" value={node.paramCount.toLocaleString()} />
      <Row label="EST. SIZE" value={`${node.estimatedSizeMB.toFixed(3)} MB`} />
      {attrEntries.length > 0 && (
        <>
          <div
            style={{
              color: 'var(--text-dim)',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
              fontSize: 10,
              margin: '16px 0 4px',
            }}
          >
            Attributes
          </div>
          {attrEntries.map(([key, val]) => (
            <Row key={key} label={key} value={String(val)} />
          ))}
        </>
      )}
    </div>
  )
}
