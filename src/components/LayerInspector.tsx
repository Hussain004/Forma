import { useState, useEffect, useRef, type CSSProperties } from 'react'
import type { OnnxDim, OnnxNode, ModelMetadata } from '../lib/onnxTypes'
import { formatShape } from '../lib/onnxProtoParser'
import { opCategoryColor, type DeleteEligibility } from '../lib/graphUtils'
import { parseAttrEdit } from '../lib/attrUtils'
import { PIPELINE_RECIPES, type PipelineRecipe } from '../lib/pipelineRecipes'

// Bare comma list ("1, 3, 8, 8"), not formatShape's bracketed display form --
// this one has to round-trip through parseShapeEdit below. A purely numeric
// token is a concrete dim, anything else is a symbolic one (batch, N, ...).
function formatShapeForEdit(dims?: OnnxDim[]): string {
  return (dims ?? []).map((d) => ('value' in d ? String(d.value) : d.param)).join(', ')
}

// Empty input means unranked (shape omitted entirely), not rank-0/scalar --
// there's no text-box syntax here for "explicitly empty shape", the same
// simplification onnxProtoWriter.ts's encodeValueInfo documents.
function parseShapeEdit(value: string): OnnxDim[] | null {
  const trimmed = value.trim()
  if (!trimmed) return null
  return trimmed.split(',').map((token) => {
    const t = token.trim()
    return /^-?\d+$/.test(t) ? { value: Number(t) } : { param: t }
  })
}

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
  onExtractRepro?: () => void
  onAttrEdit?: (nodeId: string, attrName: string, value: string | number) => void
  onDeleteNode?: (nodeId: string, keepInputPosition: number | null) => void
  deleteEligibility?: DeleteEligibility
  onCopy?: () => void
  onRenameNode?: (nodeId: string, name: string) => void
  onRenameTensor?: (oldName: string, newName: string) => void
  onSetGraphIO?: (nodeId: string, elemType: number, dims: OnnxDim[] | null) => void
  onPromoteOutput?: (tensorName: string) => void
  onReplaceConstant?: (initializerName: string, values: number[]) => void
  onInsertRecipe?: (recipe: PipelineRecipe) => void
}

const bulkButtonStyle: React.CSSProperties = {
  fontSize: 12,
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
  fontSize: 13,
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

// One inline-editable field, shared by attribute values, node/tensor
// renames, graph I/O type/shape, and constant values -- click to edit, Enter
// or blur commits, Escape cancels. `fieldKey` distinguishes which field is
// being edited (see LayerInspector's editingField state); only one is ever
// open at a time.
interface FieldEdit {
  editingField: string | null
  editValue: string
  onStart: (key: string, current: string) => void
  onChange: (value: string) => void
  onCommit: (key: string) => void
  onCancel: () => void
}

function EditableText({ fieldKey, value, edit, testIdPrefix, testIdSuffix, title, textStyle }: {
  fieldKey: string
  value: string
  edit: FieldEdit
  testIdPrefix: string
  // Defaults to fieldKey; overridden where fieldKey carries a prefix (e.g.
  // `attr:${name}`) that pre-v2.3 tests don't expect in the testid itself.
  testIdSuffix?: string
  title?: string
  textStyle?: CSSProperties
}) {
  const testId = testIdSuffix ?? fieldKey
  if (edit.editingField === fieldKey) {
    return (
      <input
        data-testid={`${testIdPrefix}-input-${testId}`}
        autoFocus
        value={edit.editValue}
        onChange={(e) => edit.onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') { e.preventDefault(); edit.onCommit(fieldKey) }
          if (e.key === 'Escape') { e.stopPropagation(); edit.onCancel() }
        }}
        onBlur={() => edit.onCommit(fieldKey)}
        className="input-mono"
        style={{ flex: 1, minWidth: 0, background: 'rgba(255,176,0,0.06)', borderRadius: 1, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontSize: 12, padding: '0 4px' }}
      />
    )
  }
  return (
    <button
      type="button"
      data-testid={`${testIdPrefix}-value-${testId}`}
      className="attr-value-button"
      title={title}
      onClick={() => edit.onStart(fieldKey, value)}
      onKeyDown={(e) => { if (e.key === 'Enter') edit.onStart(fieldKey, value) }}
    >
      <span style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', ...textStyle }}>{value}</span>
      <PencilIcon color="var(--text-dim)" />
    </button>
  )
}

// Tensor name + (when known) its shape, one row -- replaces what used to be
// two separate sections ("Input shapes" and "Inputs") that both listed every
// tensor name, once with a shape and once without. Tensor names aren't
// semantic labels like "OP TYPE" (they're arbitrary, sometimes long
// identifiers), so this deliberately doesn't reuse Row's uppercase labelStyle.
// fieldKey+edit make the name itself renamable; onPromote adds a button to
// promote this tensor to a graph output (outputs of a compute node only).
function IORow({ name, shape, fieldKey, edit, onPromote }: {
  name: string
  shape?: string
  fieldKey?: string
  edit?: FieldEdit
  onPromote?: () => void
}) {
  return (
    <div style={{ ...rowStyle, gap: 8, alignItems: 'center' }}>
      {fieldKey && edit ? (
        <EditableText fieldKey={fieldKey} value={name} edit={edit} testIdPrefix="tensor" title="Rename this tensor" textStyle={{ fontSize: 13, color: 'var(--text-secondary)' }} />
      ) : (
        <span style={{ color: 'var(--text-secondary)', fontSize: 13, wordBreak: 'break-word', flex: 1, minWidth: 0 }}>{name}</span>
      )}
      {shape && <span style={{ color: 'var(--text-dim)', fontSize: 12, whiteSpace: 'nowrap' }}>{shape}</span>}
      {onPromote && (
        <button type="button" data-testid={`promote-output-${name}`} onClick={onPromote} title="Promote this tensor to an additional graph output" className="btn-ghost" style={{ fontSize: 11, padding: '1px 6px', flexShrink: 0 }}>
          &rarr; OUT
        </button>
      )}
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
        fontSize: 12,
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

export function LayerInspector({ node, onToggleExclude, quantizeEstimate, modelStats, multiSelection, onBulkExclude, onBulkInclude, onBulkDelete, onExtractRepro, onAttrEdit, onDeleteNode, deleteEligibility, onCopy, onRenameNode, onRenameTensor, onSetGraphIO, onPromoteOutput, onReplaceConstant, onInsertRecipe }: LayerInspectorProps) {
  // One editingField/editValue pair drives every inline-editable field on this
  // panel (attributes, node name, tensor names, graph I/O type/shape, constant
  // values) -- fieldKey (see commitEditField) says which. Only one can be open
  // at a time, which is also the desired UX: committing or canceling one
  // closes it before another can open.
  const [editingField, setEditingField] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const [showDeletePicker, setShowDeletePicker] = useState(false)
  const cancelEditRef = useRef(false)

  useEffect(() => {
    setEditingField(null)
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
          {onExtractRepro && (
            <button
              data-testid="extract-repro-button"
              onClick={onExtractRepro}
              title="Export just this selection as a standalone ONNX file: boundary tensors become new graph inputs/outputs, required weights are preserved"
              className="btn-bar"
              style={bulkButtonStyle}
            >
              EXTRACT REPRO
            </button>
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
          <span style={{ color: 'var(--text-dim)', letterSpacing: '0.06em', fontSize: 12, marginTop: 8 }}>
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
        <div style={{ color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.1em', fontSize: 12, marginBottom: 12 }}>
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
        <div style={{ color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.1em', fontSize: 12, margin: '16px 0 4px' }}>
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
            <div style={{ color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.1em', fontSize: 12, margin: '16px 0 4px' }}>
              Categories
            </div>
            {legend.map((c) => (
              <div key={c.name} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0' }}>
                <span style={{ ...swatchStyle, background: c.color }} />
                <span style={{ color: 'var(--text-secondary)', fontSize: 13, letterSpacing: '0.04em' }}>{c.name}</span>
              </div>
            ))}
          </>
        )}
        <div style={{ marginTop: 20, paddingTop: 12, borderTop: '1px solid rgba(255,255,255,0.06)', color: 'var(--text-dim)', fontSize: 12, letterSpacing: '0.04em' }}>
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

  function startEditField(key: string, current: string) {
    setEditingField(key)
    setEditValue(current)
  }

  function cancelEditField() {
    cancelEditRef.current = true
    setEditingField(null)
  }

  // Dispatches on fieldKey's prefix -- see the field's own EditableText call
  // site for the exact key shape (e.g. `attr:${name}`, `input:${position}`).
  function commitEditField(key: string) {
    if (cancelEditRef.current) {
      cancelEditRef.current = false
      setEditingField(null)
      return
    }
    if (node) {
      if (key === 'nodename') {
        const name = editValue.trim()
        if (name && name !== (node.name ?? '')) onRenameNode?.(node.id, name)
      } else if (key.startsWith('attr:')) {
        const attrName = key.slice(5)
        const original = node.attributes[attrName] as string | number
        const parsed = parseAttrEdit(editValue, original)
        if (parsed !== original) onAttrEdit?.(node.id, attrName, parsed)
      } else if (key.startsWith('input:') || key.startsWith('output:')) {
        const [kind, positionText] = key.split(':')
        const position = Number(positionText)
        const oldName = kind === 'input' ? node.inputs[position] : node.outputs[position]
        const newName = editValue.trim()
        if (oldName && newName && newName !== oldName) onRenameTensor?.(oldName, newName)
      } else if (key === 'iotype') {
        const elemType = Number(editValue.trim())
        if (Number.isInteger(elemType) && elemType > 0) {
          const currentShape = node.opType === 'Input' ? node.outputShapes?.[0] : node.inputShapes?.[0]
          onSetGraphIO?.(node.id, elemType, currentShape ?? null)
        }
      } else if (key === 'ioshape') {
        const elemType = node.opType === 'Input' ? node.outputMetadata?.[0]?.elemType : node.inputMetadata?.[0]?.elemType
        onSetGraphIO?.(node.id, elemType ?? 1, parseShapeEdit(editValue))
      } else if (key.startsWith('const:')) {
        const initializerName = key.slice(6)
        const values = editValue.split(',').map((token) => Number(token.trim()))
        if (values.every((v) => Number.isFinite(v))) onReplaceConstant?.(initializerName, values)
      }
    }
    setEditingField(null)
  }

  const fieldEdit: FieldEdit = {
    editingField,
    editValue,
    onStart: startEditField,
    onChange: setEditValue,
    onCommit: commitEditField,
    onCancel: cancelEditField,
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
            fontSize: 11,
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
          style={{ marginLeft: 'auto', fontSize: 12, padding: '2px 10px' }}
        >
          COPY
        </button>
      </div>
      <Row label="OP TYPE" value={node.opType} />
      {isCompute && onRenameNode && (
        <div style={rowStyle}>
          <span style={labelStyle}>NODE NAME</span>
          <EditableText fieldKey="nodename" value={node.name ?? ''} edit={fieldEdit} testIdPrefix="nodename" title="Rename this node" />
        </div>
      )}
      {(!isCompute || !onRenameNode) && node.name && <Row label="NODE NAME" value={node.name} />}

      {Object.keys(node.attributes ?? {}).length > 0 && (
        <>
          {sectionHeader('Attributes')}
          {Object.entries(node.attributes).map(([k, v]) => (
            <div key={k} style={rowStyle}>
              <span style={labelStyle}>{k}</span>
              <EditableText fieldKey={`attr:${k}`} testIdSuffix={k} value={String(v)} edit={fieldEdit} testIdPrefix="attr" title={`Edit ${k}`} textStyle={valueStyle} />
            </div>
          ))}
        </>
      )}

      {node.inputs.length > 0 && (
        <>
          {sectionHeader('Inputs')}
          {node.inputs.map((inp, i) => {
            const name = inp || `input_${i}`
            const constValues = node.inputMetadata?.[i]?.values
            return (
              <div key={name + i}>
                <IORow
                  name={name}
                  shape={node.inputShapes?.[i] !== undefined ? (formatShape(node.inputShapes[i]) || 'unknown') : undefined}
                  fieldKey={onRenameTensor ? `input:${i}` : undefined}
                  edit={onRenameTensor ? fieldEdit : undefined}
                />
                {constValues && (
                  <div style={{ ...rowStyle, gap: 8, paddingLeft: 12 }}>
                    <span style={{ color: 'var(--text-dim)', fontSize: 12, flexShrink: 0 }}>= </span>
                    {onReplaceConstant ? (
                      <EditableText fieldKey={`const:${name}`} value={constValues.join(', ')} edit={fieldEdit} testIdPrefix="const" title={`Edit constant ${name}`} textStyle={{ fontSize: 12 }} />
                    ) : (
                      <span style={{ color: 'var(--text-secondary)', fontSize: 12, wordBreak: 'break-word' }}>{constValues.join(', ')}</span>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </>
      )}

      {node.outputs.length > 0 && (
        <>
          {sectionHeader('Outputs')}
          {node.outputs.map((out, i) => {
            const name = out || `output_${i}`
            return (
              <IORow
                key={name + i}
                name={name}
                shape={node.outputShapes?.[i] !== undefined ? (formatShape(node.outputShapes[i]) || 'unknown') : undefined}
                fieldKey={onRenameTensor ? `output:${i}` : undefined}
                edit={onRenameTensor ? fieldEdit : undefined}
                onPromote={isCompute && onPromoteOutput ? () => onPromoteOutput(name) : undefined}
              />
            )
          })}
        </>
      )}

      {!isCompute && onSetGraphIO && (
        <>
          {sectionHeader('Declared Type')}
          <div style={rowStyle}>
            <span style={labelStyle} title="ONNX elem_type: 1=float32, 2=uint8, 3=int8, 6=int32, 7=int64, 9=bool, 10=float16, 11=double">ELEM TYPE</span>
            <EditableText
              fieldKey="iotype"
              value={String((node.opType === 'Input' ? node.outputMetadata?.[0]?.elemType : node.inputMetadata?.[0]?.elemType) ?? '')}
              edit={fieldEdit}
              testIdPrefix="io"
              title="Edit the declared element type (ONNX elem_type enum)"
            />
          </div>
          <div style={rowStyle}>
            <span style={labelStyle} title="Comma-separated dims; a non-numeric token is a symbolic dimension (batch, N, ...); empty means unranked">SHAPE</span>
            <EditableText
              fieldKey="ioshape"
              value={formatShapeForEdit(node.opType === 'Input' ? node.outputShapes?.[0] : node.inputShapes?.[0])}
              edit={fieldEdit}
              testIdPrefix="io"
              title="Edit the declared shape"
            />
          </div>
        </>
      )}

      {!isCompute && onInsertRecipe && (
        <>
          {sectionHeader(node.opType === 'Input' ? 'Preprocessing Recipes' : 'Postprocessing Recipes')}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {PIPELINE_RECIPES.filter((r) =>
              r.category === (node.opType === 'Input' ? 'preprocess' : 'postprocess') &&
              (!r.minOpset || (modelStats?.metadata?.opsetVersion ?? Infinity) >= r.minOpset),
            ).map((recipe) => (
              <button
                key={recipe.id}
                type="button"
                data-testid={`insert-recipe-${recipe.id}`}
                onClick={() => onInsertRecipe(recipe)}
                title={recipe.description}
                className="btn-ghost"
                style={{ textAlign: 'left', fontSize: 13, padding: '4px 8px' }}
              >
                {recipe.label}
              </button>
            ))}
          </div>
        </>
      )}

      {sectionHeader('Stats')}
      <Row label="PARAMETERS" value={node.paramCount.toLocaleString()} />
      <Row label="EST. SIZE" value={`${node.estimatedSizeMB.toFixed(3)} MB`} />
      {node.estimatedSizeMB > 0 && quantizeEstimate && quantizeEstimate.ratio > 0 && (
        <div style={{ color: 'var(--text-dim)', fontSize: 12, padding: '2px 0 0 108px', letterSpacing: '0.04em' }}>
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
            style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: sensitivityColor(node.paramCount), letterSpacing: '0.04em' }}
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
                fontSize: 12,
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
                  style={{ fontSize: 12, padding: '2px 10px' }}
                >
                  {deleteEligibility.candidateInputs.length > 1 ? 'Choose source' : 'Delete'}
                </button>
              </div>
              {!deleteEligibility.eligible && deleteEligibility.reason && (
                <div style={{ fontSize: 12, color: 'var(--text-dim)', letterSpacing: '0.02em', paddingLeft: 108 }}>
                  {deleteEligibility.reason}
                </div>
              )}
              {showDeletePicker && deleteEligibility.eligible && deleteEligibility.candidateInputs.length > 1 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2, paddingLeft: 108 }}>
                  <div style={{ fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                    Reconnect using
                  </div>
                  {deleteEligibility.candidateInputs.map((c) => (
                    <button
                      key={c.position}
                      type="button"
                      data-testid={`delete-picker-option-${c.position}`}
                      onClick={() => { onDeleteNode(node.id, c.position); setShowDeletePicker(false) }}
                      style={{
                        textAlign: 'left',
                        textTransform: 'none',
                        letterSpacing: 'normal',
                        cursor: 'pointer',
                        padding: '4px 8px',
                        fontSize: 13,
                        fontFamily: 'var(--font-mono)',
                        color: 'var(--text-primary)',
                        background: 'rgba(255,176,0,0.06)',
                        border: '1px solid rgba(255,176,0,0.2)',
                        borderRadius: 1,
                      }}
                    >
                      {c.tensorName}
                    </button>
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
