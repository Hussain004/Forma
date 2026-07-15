import { useState, useEffect, useRef, type CSSProperties } from 'react'
import type { OnnxNode, ModelMetadata } from '../lib/onnxTypes'
import { formatShape } from '../lib/onnxProtoParser'
import { opCategoryColor, type DeleteEligibility } from '../lib/graphUtils'
import { parseAttrEdit } from '../lib/attrUtils'

function PencilIcon({ color }: { color: string }) {
  const style = { transition: 'fill 140ms ease' }
  return (
    <svg width="9" height="9" viewBox="0 0 9 9" aria-hidden="true" style={{ flexShrink: 0, display: 'block' }}>
      <polygon points="1.5,7 6,2.5 7,3.5 2.5,8" fill={color} style={style} />
      <polygon points="1.5,7 2,8.5 2.5,8" fill={color} style={style} />
      <polygon points="6,2.5 7,1.5 8,2.5 7,3.5" fill={color} style={style} />
    </svg>
  )
}

interface LayerInspectorProps {
  node: OnnxNode | null
  onToggleExclude?: (nodeId: string) => void
  quantizeEstimate?: { ratio: number } | null
  modelStats?: { opCounts: Record<string, number>; totalNodes: number; graphDepth?: number; metadata?: ModelMetadata } | null
  multiSelection?: {
    nodes: OnnxNode[]
    totalParams: number
    totalSizeMB: number
  }
  onBulkExclude?: () => void
  onBulkInclude?: () => void
  onBulkDelete?: () => void
  onAttrEdit?: (nodeId: string, attrName: string, value: string | number) => void
  onDeleteNode?: (nodeId: string, keepInputPosition: number | null) => void
  deleteEligibility?: DeleteEligibility
  onCopy?: () => void
}

const bulkButtonStyle: React.CSSProperties = {
  fontSize: 10,
  padding: '2px 10px',
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

// Tensor name + (when known) its shape, one row -- replaces what used to be
// two separate sections ("Input shapes" and "Inputs") that both listed every
// tensor name, once with a shape and once without. Tensor names aren't
// semantic labels like "OP TYPE" (they're arbitrary, sometimes long
// identifiers), so this deliberately doesn't reuse Row's uppercase labelStyle.
function IORow({ name, shape }: { name: string; shape?: string }) {
  return (
    <div style={{ ...rowStyle, gap: 8 }}>
      <span style={{ color: 'var(--text-secondary)', fontSize: 11, wordBreak: 'break-word', flex: 1, minWidth: 0 }}>{name}</span>
      {shape && <span style={{ color: 'var(--text-dim)', fontSize: 10, whiteSpace: 'nowrap' }}>{shape}</span>}
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

export function LayerInspector({ node, onToggleExclude, quantizeEstimate, modelStats, multiSelection, onBulkExclude, onBulkInclude, onBulkDelete, onAttrEdit, onDeleteNode, deleteEligibility, onCopy }: LayerInspectorProps) {
  const [editingAttr, setEditingAttr] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const [hoveredAttr, setHoveredAttr] = useState<string | null>(null)
  const [showDeletePicker, setShowDeletePicker] = useState(false)
  const cancelEditRef = useRef(false)

  useEffect(() => {
    setEditingAttr(null)
    setEditValue('')
    setShowDeletePicker(false)
  }, [node?.id])

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
          borderLeft: '2px solid var(--color-amber)',
          padding: 16,
          height: '100%',
          minWidth: 260,
          overflowY: 'auto',
          boxSizing: 'border-box',
        }}
      >
        <div style={{ color: 'var(--color-amber)', fontFamily: 'var(--font-mono)', fontSize: 12, letterSpacing: '0.08em', marginBottom: 12, textTransform: 'uppercase' }}>
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
          <button onClick={onBulkExclude} className="btn-ghost" style={bulkButtonStyle}>EXCLUDE ALL</button>
          <button onClick={onBulkInclude} className="btn-ghost" style={bulkButtonStyle}>INCLUDE ALL</button>
          {onBulkDelete && (
            <button data-testid="bulk-delete-button" onClick={onBulkDelete} className="btn-danger" style={bulkButtonStyle}>DELETE ALL</button>
          )}
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
            flexDirection: 'column',
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
          <span style={{ color: 'var(--text-dim)', letterSpacing: '0.06em', fontSize: 10, marginTop: 8 }}>
            Press ? for shortcuts
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
        {modelStats.metadata?.producerName && (
          <div style={rowStyle}>
            <span style={labelStyle}>PRODUCER</span>
            <span style={valueStyle}>{modelStats.metadata.producerName}{modelStats.metadata.producerVersion ? ` ${modelStats.metadata.producerVersion}` : ''}</span>
          </div>
        )}
        {modelStats.metadata?.opsetVersion ? (
          <div style={rowStyle}>
            <span style={labelStyle}>OPSET</span>
            <span style={valueStyle}>{modelStats.metadata.opsetVersion}</span>
          </div>
        ) : null}
        {modelStats.metadata?.irVersion ? (
          <div style={rowStyle}>
            <span style={labelStyle}>IR VER</span>
            <span style={valueStyle}>{modelStats.metadata.irVersion}</span>
          </div>
        ) : null}
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
        <div style={{ marginTop: 20, paddingTop: 12, borderTop: '1px solid rgba(255,255,255,0.06)', color: 'var(--text-dim)', fontSize: 10, letterSpacing: '0.04em' }}>
          Click a node to inspect. Press ? for shortcuts.
        </div>
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
    const attrs = Object.entries(node.attributes ?? {})
    if (attrs.length > 0) attrs.forEach(([k, v]) => lines.push(`${k}: ${v}`))
    if (node.inputs.length > 0) lines.push(`Inputs:     ${node.inputs.join(', ')}`)
    if (node.outputs.length > 0) lines.push(`Outputs:    ${node.outputs.join(', ')}`)
    navigator.clipboard.writeText(lines.join('\n')).then(() => onCopy?.()).catch(() => {})
  }

  function startEdit(attrName: string, current: string | number) {
    setEditingAttr(attrName)
    setEditValue(String(current))
  }

  function commitEdit(attrName: string, original: string | number) {
    if (cancelEditRef.current) {
      cancelEditRef.current = false
      return
    }
    const parsed = parseAttrEdit(editValue, original)
    if (parsed !== original && node) {
      onAttrEdit?.(node.id, attrName, parsed)
    }
    setEditingAttr(null)
  }

  function cancelEdit() {
    cancelEditRef.current = true
    setEditingAttr(null)
  }

  return (
    <div
      style={{
        background: 'var(--bg-surface)',
        borderLeft: '2px solid var(--color-amber)',
        padding: 16,
        height: '100%',
        minWidth: 260,
        overflowY: 'auto',
        boxSizing: 'border-box',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        {node.isModified && (
          <span style={{
            fontSize: 9,
            letterSpacing: '0.08em',
            color: 'var(--color-amber)',
            background: 'rgba(255,176,0,0.12)',
            padding: '2px 6px',
            borderRadius: 1,
            textTransform: 'uppercase',
          }}>
            Modified
          </span>
        )}
        <button
          onClick={handleCopy}
          className="btn-primary"
          style={{ marginLeft: 'auto', fontSize: 10, padding: '2px 10px' }}
        >
          COPY
        </button>
      </div>
      <Row label="OP TYPE" value={node.opType} />
      {node.name && <Row label="NODE NAME" value={node.name} />}

      {Object.keys(node.attributes ?? {}).length > 0 && (
        <>
          {sectionHeader('Attributes')}
          {Object.entries(node.attributes).map(([k, v]) => {
            const original = v as string | number
            const isEditing = editingAttr === k
            const isHovered = hoveredAttr === k
            const pencilColor = isHovered ? 'var(--color-amber)' : 'var(--text-dim)'
            const hoverRowStyle: CSSProperties = {
              ...rowStyle,
              background: isHovered && !isEditing ? 'rgba(255,176,0,0.04)' : 'transparent',
              transition: 'background 140ms ease',
            }
            return (
              <div key={k} style={hoverRowStyle}>
                <span style={labelStyle}>{k}</span>
                {isEditing ? (
                  <input
                    data-testid={`attr-input-${k}`}
                    autoFocus
                    value={editValue}
                    onChange={e => setEditValue(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') { e.preventDefault(); commitEdit(k, original) }
                      if (e.key === 'Escape') { e.stopPropagation(); cancelEdit() }
                    }}
                    onBlur={() => commitEdit(k, original)}
                    className="input-mono"
                    style={{
                      flex: 1,
                      minWidth: 0,
                      background: 'rgba(255,176,0,0.06)',
                      borderRadius: 1,
                      color: 'var(--text-primary)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: 12,
                      padding: '0 4px',
                    }}
                  />
                ) : (
                  <div
                    data-testid={`attr-value-${k}`}
                    style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6, cursor: 'text', minWidth: 0 }}
                    onClick={() => startEdit(k, original)}
                    onMouseEnter={() => setHoveredAttr(k)}
                    onMouseLeave={() => setHoveredAttr(null)}
                  >
                    <span style={{ ...valueStyle, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {String(v)}
                    </span>
                    <PencilIcon color={pencilColor} />
                  </div>
                )}
              </div>
            )
          })}
        </>
      )}

      {node.inputs.length > 0 && (
        <>
          {sectionHeader('Inputs')}
          {node.inputs.map((inp, i) => (
            <IORow key={inp || i} name={inp || `input_${i}`} shape={node.inputShapes?.[i] !== undefined ? (formatShape(node.inputShapes[i]) || 'unknown') : undefined} />
          ))}
        </>
      )}

      {node.outputs.length > 0 && (
        <>
          {sectionHeader('Outputs')}
          {node.outputs.map((out, i) => (
            <IORow key={out || i} name={out || `output_${i}`} shape={node.outputShapes?.[i] !== undefined ? (formatShape(node.outputShapes[i]) || 'unknown') : undefined} />
          ))}
        </>
      )}

      {sectionHeader('Stats')}
      <Row label="PARAMETERS" value={node.paramCount.toLocaleString()} />
      <Row label="EST. SIZE" value={`${node.estimatedSizeMB.toFixed(3)} MB`} />
      {node.estimatedSizeMB > 0 && quantizeEstimate && quantizeEstimate.ratio > 0 && (
        <div style={{ color: 'var(--text-dim)', fontSize: 10, padding: '2px 0 0 108px', letterSpacing: '0.04em' }}>
          {`INT8: ${(node.estimatedSizeMB / quantizeEstimate.ratio).toFixed(3)} MB`}
        </div>
      )}
      {isCompute && node.paramCount > 0 && (
        <div style={{ ...rowStyle, alignItems: 'center' }}>
          <span style={labelStyle} title="A heuristic based on parameter count only -- not a profiling measurement">
            SENSITIVITY
          </span>
          <span
            title="A heuristic based on parameter count only -- not a profiling measurement"
            style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: sensitivityColor(node.paramCount), letterSpacing: '0.04em' }}
          >
            {sensitivityLabel(node.paramCount)}
          </span>
        </div>
      )}

      {isCompute && (
        <div style={{ marginTop: 20, paddingTop: 12, borderTop: '1px solid rgba(255,255,255,0.06)' }}>
          <div style={{ ...rowStyle, alignItems: 'center', border: 'none' }}>
            <button
              onClick={() => node && onToggleExclude?.(node.id)}
              title="Excludes this node from the model summary's param/size rollups and dims it on the canvas. Never affects export -- the node stays in the model either way."
              style={{
                border: node.excluded ? '1px solid var(--color-amber)' : '1px solid rgba(255,255,255,0.15)',
                color: node.excluded ? 'var(--color-amber)' : 'var(--text-secondary)',
                fontSize: 10,
                padding: '3px 10px',
              }}
            >
              {node.excluded ? 'Include in stats' : 'Exclude from stats'}
            </button>
          </div>

          {onDeleteNode && deleteEligibility && (
            <div style={{ ...rowStyle, flexDirection: 'column', alignItems: 'stretch', gap: 6, border: 'none' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={labelStyle}>DELETE NODE</span>
                <button
                  data-testid="delete-node-button"
                  disabled={!deleteEligibility.eligible}
                  title={deleteEligibility.reason}
                  onClick={() => {
                    if (!deleteEligibility.eligible) return
                    if (deleteEligibility.candidateInputs.length > 1) {
                      setShowDeletePicker((v) => !v)
                      return
                    }
                    onDeleteNode(node.id, deleteEligibility.candidateInputs[0]?.position ?? null)
                  }}
                  className="btn-danger"
                  style={{ fontSize: 10, padding: '2px 10px' }}
                >
                  {deleteEligibility.candidateInputs.length > 1 ? 'Choose source' : 'Delete'}
                </button>
              </div>
              {!deleteEligibility.eligible && deleteEligibility.reason && (
                <div style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: '0.02em', paddingLeft: 108 }}>
                  {deleteEligibility.reason}
                </div>
              )}
              {showDeletePicker && deleteEligibility.eligible && deleteEligibility.candidateInputs.length > 1 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2, paddingLeft: 108 }}>
                  <div style={{ fontSize: 9, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                    Reconnect using
                  </div>
                  {deleteEligibility.candidateInputs.map((c) => (
                    <div
                      key={c.position}
                      data-testid={`delete-picker-option-${c.position}`}
                      onClick={() => { onDeleteNode(node.id, c.position); setShowDeletePicker(false) }}
                      style={{
                        cursor: 'pointer',
                        padding: '4px 8px',
                        fontSize: 11,
                        fontFamily: 'var(--font-mono)',
                        color: 'var(--text-primary)',
                        background: 'rgba(255,176,0,0.06)',
                        border: '1px solid rgba(255,176,0,0.2)',
                        borderRadius: 1,
                      }}
                    >
                      {c.tensorName}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
