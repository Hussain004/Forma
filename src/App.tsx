import { useEffect, useMemo, useRef, useState } from 'react'
import { parseAttrEdit } from './lib/attrUtils'
import { ModelDropzone } from './components/ModelDropzone'
import { GraphCanvas } from './components/GraphCanvas'
import { LayerInspector } from './components/LayerInspector'
import { useOnnxWorker } from './hooks/useOnnxWorker'
import { toSelectableGraph, deselectAll, filterGraph, excludeNode, includeNode, setMultiSelection, bulkExclude, bulkInclude, computeOpCounts, computeGraphDepth, getAncestors, getDescendants, getDeleteEligibility, deleteNodeWithReconnect, insertPassthroughNode, validateRewire, rewireEdge, addCustomNode, structuralNodeIndex, CURATED_NODE_TYPES, type SelectableGraph } from './lib/graphUtils'
import { formatQuantizeEstimate } from './lib/quantize'
import type { OnnxNode } from './lib/onnxTypes'
import type { QuantizeEstimate } from './hooks/useOnnxWorker'
import type { StructuralOp } from './lib/onnxProtoWriter'
import './index.css'

// UI-layer structural edit record: carries both the live graph ids (for the
// visualization-layer reducers in graphUtils.ts) and the original 0-based node
// indices (for translating to the protobuf writer's index-only StructuralOp at
// export time). See onnxProtoWriter.ts for why positions, not tensor-name values,
// are what make ops replay correctly in sequence.
type GraphEdit =
  | { type: 'delete'; nodeId: string; nodeIndex: number; keepInputPosition: number | null }
  | { type: 'insertPassthrough'; targetNodeId: string; targetNodeIndex: number; inputPosition: number; newNodeId: string }
  | { type: 'rewire'; sourceNodeId: string; sourceNodeIndex: number; targetNodeId: string; targetNodeIndex: number; inputPosition: number }
  | { type: 'addNode'; newNodeId: string; newNodeIndex: number; opType: string; inputCount: number; position: { x: number; y: number } }

type UndoEntry =
  | { kind: 'attr'; nodeId: string; attrName: string; prevValue: string | number }
  | { kind: 'structural' }

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`
  return String(n)
}

// Renders both a full ("2.4M PARAMS") and compact ("2.4M") variant as two
// complete, separate text nodes rather than nesting the unit word inside a
// conditionally-hidden child span -- the latter reads fine visually but
// breaks plain-string test queries, since testing-library's default text
// matcher doesn't concatenate text split across element boundaries.
function StatValue({ value, unit, title }: { value: string; unit: string; title: string }) {
  return (
    <span style={{ flexShrink: 0 }} title={title}>
      <span className="stat-full">{value} {unit}</span>
      <span className="stat-compact" aria-hidden="true">{value}</span>
    </span>
  )
}

const SHORTCUTS: [string, string][] = [
  ['/', 'Focus the node filter'],
  ['Click', 'Select a node'],
  ['Ctrl+Click', 'Multi-select nodes'],
  ['Shift+Drag', 'Box-select multiple nodes'],
  ['Drag handle to handle', 'Rewire a connection'],
  ['Delete', 'Delete the selected node(s)'],
  ['Ctrl+Z', 'Undo the last edit'],
  ['Esc', 'Cancel / clear selection'],
  ['?', 'Toggle this panel'],
]

function ShortcutsOverlay({ onClose }: { onClose: () => void }) {
  return (
    <div
      role="dialog"
      aria-label="Keyboard shortcuts"
      data-testid="shortcuts-overlay"
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 2000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(0,0,0,0.5)',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--bg-surface)',
          borderLeft: '2px solid var(--color-amber)',
          border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: 2,
          padding: 20,
          width: 360,
          fontFamily: 'var(--font-mono)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <span style={{ color: 'var(--color-amber)', fontSize: 12, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            Keyboard shortcuts
          </span>
          <button onClick={onClose} className="btn-ghost" style={{ fontSize: 10, padding: '2px 8px' }}>
            Close
          </button>
        </div>
        {SHORTCUTS.map(([key, desc]) => (
          <div key={key} style={{ display: 'flex', justifyContent: 'space-between', gap: 16, padding: '5px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <span style={{ color: 'var(--text-primary)', fontSize: 11, letterSpacing: '0.02em' }}>{desc}</span>
            <span style={{ color: 'var(--text-dim)', fontSize: 11, letterSpacing: '0.02em', whiteSpace: 'nowrap' }}>{key}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

interface StatsBarProps {
  modelName: string
  totalParams: number
  totalSizeMB: number
  nodeCount: number
  quantizeEstimate: QuantizeEstimate | null
  filterQuery: string
  onFilterChange: (value: string) => void
  filterInputRef: React.RefObject<HTMLInputElement | null>
  onFilterKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void
  onFilterFocus: () => void
  onFilterBlur: () => void
  dropdownResults: Array<{ id: string; opType: string; paramCount: number }>
  showDropdown: boolean
  dropdownIndex: number
  onDropdownSelect: (id: string) => void
  layoutDir: 'TB' | 'LR'
  onLayoutToggle: () => void
  onBenchmark: () => void
  benchmarkLabel: string | null
  isBenchmarking: boolean
  onDownload: () => void
  canDownload: boolean
  onDownloadModified: () => void
  canDownloadModified: boolean
  onReset: () => void
  isReadOnly: boolean
  onAddNode: (opType: string, inputCount: number) => void
  editCount: number
}

function StatsBar({ modelName, totalParams, totalSizeMB, nodeCount, quantizeEstimate, filterQuery, onFilterChange, filterInputRef, onFilterKeyDown, onFilterFocus, onFilterBlur, dropdownResults, showDropdown, dropdownIndex, onDropdownSelect, layoutDir, onLayoutToggle, onBenchmark, benchmarkLabel, isBenchmarking, onDownload, canDownload, onDownloadModified, canDownloadModified, onReset, isReadOnly, onAddNode, editCount }: StatsBarProps) {
  const quantizeLabel = formatQuantizeEstimate(quantizeEstimate)
  const [showAddNode, setShowAddNode] = useState(false)
  const [addNodeQuery, setAddNodeQuery] = useState('')
  const [addNodeInputCount, setAddNodeInputCount] = useState(1)
  const [showOverflowMenu, setShowOverflowMenu] = useState(false)

  const commitAddNode = (opType: string, inputCount: number) => {
    if (!opType.trim()) return
    onAddNode(opType.trim(), inputCount)
    setShowAddNode(false)
    setAddNodeQuery('')
    setAddNodeInputCount(1)
  }
  return (
    <div className="stats-bar" style={{
      height: 52,
      background: '#0E1114',
      borderBottom: '1px solid rgba(255,255,255,0.1)',
      display: 'flex',
      alignItems: 'center',
      padding: '0 24px',
      gap: 32,
      fontFamily: 'var(--font-mono)',
      fontSize: 13,
      color: 'var(--text-secondary)',
      flexShrink: 0,
      letterSpacing: '0.06em',
    }}>
      <span style={{ color: 'var(--text-primary)', fontWeight: 500, flex: '1 1 120px', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={modelName}>
        {modelName}
      </span>
      {isReadOnly && (
        <span style={{ flexShrink: 0, fontSize: 10, letterSpacing: '0.08em', color: 'var(--text-dim)', background: 'rgba(255,255,255,0.06)', padding: '2px 8px', borderRadius: 2, textTransform: 'uppercase', whiteSpace: 'nowrap' }}>
          TFLite read-only
        </span>
      )}
      <StatValue value={formatNumber(totalParams)} unit="PARAMS" title="Total parameters" />
      <StatValue value={totalSizeMB.toFixed(1)} unit="MB" title="Estimated size" />
      {quantizeLabel && (
        <span style={{ flexShrink: 0, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: 13, letterSpacing: '0.06em' }}>
          {quantizeLabel}
        </span>
      )}
      <StatValue value={String(nodeCount)} unit={nodeCount === 1 ? 'NODE' : 'NODES'} title="Node count" />
      <div style={{ position: 'relative', flexShrink: 0 }}>
        <input
          ref={filterInputRef}
          type="text"
          role="combobox"
          aria-expanded={showDropdown && dropdownResults.length > 0}
          aria-controls="search-dropdown"
          aria-autocomplete="list"
          aria-activedescendant={dropdownIndex >= 0 ? `search-result-${dropdownIndex}` : undefined}
          value={filterQuery}
          onChange={(e) => onFilterChange(e.target.value)}
          onKeyDown={onFilterKeyDown}
          onFocus={onFilterFocus}
          onBlur={onFilterBlur}
          placeholder="FILTER NODES  /"
          title="Focus with / -- press ? for all shortcuts"
          className="input-mono"
          style={{
            background: 'transparent',
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-mono)',
            fontSize: 13,
            letterSpacing: '0.06em',
            padding: '4px 12px',
            width: 200,
            borderRadius: 2,
          }}
        />
        {showDropdown && dropdownResults.length > 0 && (
          <div id="search-dropdown" data-testid="search-dropdown" role="listbox" style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            width: 320,
            background: 'var(--bg-raised)',
            border: '1px solid rgba(255,255,255,0.15)',
            borderRadius: 2,
            zIndex: 1000,
            marginTop: 4,
            maxHeight: 240,
            overflowY: 'auto',
          }}>
            {dropdownResults.map((node, i) => (
              <div
                key={node.id}
                id={`search-result-${i}`}
                role="option"
                aria-selected={i === dropdownIndex}
                data-testid="search-result"
                onMouseDown={() => onDropdownSelect(node.id)}
                style={{
                  padding: '8px 16px',
                  cursor: 'pointer',
                  background: i === dropdownIndex ? 'rgba(255,176,0,0.12)' : 'transparent',
                  borderLeft: i === dropdownIndex ? '2px solid var(--color-amber)' : '2px solid transparent',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontFamily: 'var(--font-mono)',
                  transition: 'background 120ms ease, border-color 120ms ease',
                }}
              >
                <span style={{ color: 'var(--text-primary)', fontSize: 13 }}>{node.opType}</span>
                {node.paramCount > 0 && (
                  <span style={{ color: 'var(--text-dim)', fontSize: 11 }}>
                    {node.paramCount.toLocaleString()}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
      <div style={{ marginLeft: 'auto', flexShrink: 0, display: 'flex', alignItems: 'center', gap: 12, paddingLeft: 24, borderLeft: '1px solid rgba(255,255,255,0.08)' }}>
        <button
          onClick={onLayoutToggle}
          title="Toggle top-to-bottom / left-to-right layout"
          className="btn-bar btn-ghost"
        >
          LAYOUT {layoutDir}
        </button>
        {!isReadOnly && (
          <div style={{ position: 'relative' }}>
            <button onClick={() => setShowAddNode(v => !v)} className="btn-bar btn-ghost">
              Add Node
            </button>
            {showAddNode && (
              <div data-testid="add-node-dropdown" style={{
                position: 'absolute',
                top: '100%',
                right: 0,
                width: 220,
                background: 'var(--bg-raised)',
                border: '1px solid rgba(255,255,255,0.15)',
                borderRadius: 2,
                zIndex: 1000,
                marginTop: 4,
                maxHeight: 320,
                overflowY: 'auto',
              }}>
                <div style={{ padding: 8 }}>
                  <input
                    data-testid="add-node-query"
                    autoFocus
                    type="text"
                    value={addNodeQuery}
                    onChange={(e) => setAddNodeQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') { e.preventDefault(); commitAddNode(addNodeQuery, addNodeInputCount) }
                      if (e.key === 'Escape') { setShowAddNode(false); setAddNodeQuery(''); setAddNodeInputCount(1) }
                    }}
                    onBlur={() => setTimeout(() => setShowAddNode(false), 150)}
                    placeholder="Op type..."
                    className="input-mono"
                    style={{
                      width: '100%',
                      boxSizing: 'border-box',
                      background: 'transparent',
                      color: 'var(--text-primary)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: 12,
                      letterSpacing: '0.04em',
                      padding: '4px 8px',
                      borderRadius: 2,
                    }}
                  />
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 6 }}>
                    <span style={{ fontSize: 9, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                      Inputs
                    </span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <button
                        type="button"
                        data-testid="add-node-input-count-dec"
                        onMouseDown={(e) => { e.preventDefault(); setAddNodeInputCount((n) => Math.max(1, n - 1)) }}
                        className="btn-ghost"
                        style={{ padding: '0 6px', fontSize: 11, lineHeight: '16px' }}
                      >
                        -
                      </button>
                      <span data-testid="add-node-input-count" style={{ fontSize: 11, color: 'var(--text-primary)', minWidth: 10, textAlign: 'center' }}>
                        {addNodeInputCount}
                      </span>
                      <button
                        type="button"
                        data-testid="add-node-input-count-inc"
                        onMouseDown={(e) => { e.preventDefault(); setAddNodeInputCount((n) => Math.min(8, n + 1)) }}
                        className="btn-ghost"
                        style={{ padding: '0 6px', fontSize: 11, lineHeight: '16px' }}
                      >
                        +
                      </button>
                    </div>
                  </div>
                </div>
                <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)' }}>
                  {CURATED_NODE_TYPES.map(({ opType, inputCount }) => (
                    <button
                      key={opType}
                      data-testid={`add-node-option-${opType}`}
                      onMouseDown={() => commitAddNode(opType, inputCount)}
                      className="btn-ghost"
                      style={{ display: 'block', width: '100%', textAlign: 'left', padding: '6px 12px' }}
                    >
                      {opType}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        {canDownloadModified && (
          <button
            onClick={onDownloadModified}
            title="Export the model with your edits applied"
            className="btn-bar btn-primary"
          >
            Export Modified ({editCount})
          </button>
        )}

        {/* Below ~1100px (container width, accounts for the resizable inspector
            panel) this group collapses into a single overflow menu -- Export
            Modified above stays put either way, it's the primary action. */}
        <div className="statsbar-full-actions" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {!isReadOnly && benchmarkLabel && (
            <span style={{ color: 'var(--color-green)' }}>{benchmarkLabel}</span>
          )}
          {!isReadOnly && (
            <button onClick={onBenchmark} disabled={isBenchmarking} className="btn-bar">
              {isBenchmarking ? 'Running' : 'Benchmark'}
            </button>
          )}
          {canDownload && (
            <button onClick={onDownload} title="Download the unmodified original file" className="btn-bar">
              Download Original
            </button>
          )}
          <button onClick={onReset} className="btn-bar btn-ghost">
            Load new
          </button>
        </div>
        <div className="statsbar-collapsed-actions" style={{ position: 'relative' }}>
          <button
            onClick={() => setShowOverflowMenu(v => !v)}
            onBlur={() => setTimeout(() => setShowOverflowMenu(false), 150)}
            title="More actions"
            className="btn-bar btn-ghost"
          >
            ...
          </button>
          {showOverflowMenu && (
            <div style={{
              position: 'absolute',
              top: '100%',
              right: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'stretch',
              width: 180,
              background: 'var(--bg-raised)',
              border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: 2,
              zIndex: 1000,
              marginTop: 4,
              padding: 4,
              gap: 2,
            }}>
              {!isReadOnly && benchmarkLabel && (
                <span style={{ color: 'var(--color-green)', fontSize: 10, padding: '4px 8px' }}>{benchmarkLabel}</span>
              )}
              {!isReadOnly && (
                <button
                  onClick={() => { onBenchmark(); setShowOverflowMenu(false) }}
                  disabled={isBenchmarking}
                  className="btn-ghost"
                  style={{ textAlign: 'left', padding: '6px 8px' }}
                >
                  {isBenchmarking ? 'Running' : 'Benchmark'}
                </button>
              )}
              {canDownload && (
                <button
                  onClick={() => { onDownload(); setShowOverflowMenu(false) }}
                  className="btn-ghost"
                  style={{ textAlign: 'left', padding: '6px 8px' }}
                >
                  Download Original
                </button>
              )}
              <button
                onClick={() => { onReset(); setShowOverflowMenu(false) }}
                className="btn-ghost"
                style={{ textAlign: 'left', padding: '6px 8px' }}
              >
                Load new
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function App() {
  const { loadModel, runBenchmark, exportModel, exportModifiedModel, graph, status, error, operationError, progress, benchmarkResult, quantizeEstimate } = useOnnxWorker()
  // TFLite support is read-only: no inference session ever exists for it (no TFLite
  // runtime in this project), so attribute/structural editing, Benchmark, and Export
  // Modified are all withheld for it -- see tfliteParser.ts and onnxWorker.ts.
  const isReadOnly = graph?.format === 'tflite'
  const [selectableGraph, setSelectableGraph] = useState<SelectableGraph | null>(null)
  const [filterQuery, setFilterQuery] = useState('')
  const [layoutDir, setLayoutDir] = useState<'TB' | 'LR'>('TB')
  const [showDropdown, setShowDropdown] = useState(false)
  const [dropdownIndex, setDropdownIndex] = useState(-1)
  const [excludedNodeIds, setExcludedNodeIds] = useState<Set<string>>(new Set())
  const [selectedNodeIds, setSelectedNodeIds] = useState<Set<string>>(new Set())
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [showDropzone, setShowDropzone] = useState(true)
  const [panelWidth, setPanelWidth] = useState(280)
  const [jumpToNodeId, setJumpToNodeId] = useState<string | null>(null)
  const [attrOverrides, setAttrOverrides] = useState<Map<string, Record<string, string | number>>>(new Map())
  const [structuralOps, setStructuralOps] = useState<GraphEdit[]>([])
  const [undoStack, setUndoStack] = useState<UndoEntry[]>([])
  const [pendingNodeType, setPendingNodeType] = useState<{ opType: string; inputCount: number } | null>(null)
  const [showShortcuts, setShowShortcuts] = useState(false)
  const [announcement, setAnnouncement] = useState<{ text: string; tone: 'reject' | 'info' } | null>(null)
  const announceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [dragOverlay, setDragOverlay] = useState(false)
  // dragenter/dragleave fire for every descendant element a drag passes over,
  // not just the root -- a depth counter is the standard way to tell "left
  // the root entirely" from "moved between two children of the root".
  const dragDepthRef = useRef(0)
  const filterInputRef = useRef<HTMLInputElement>(null)
  const undoStackRef = useRef(undoStack)
  undoStackRef.current = undoStack
  const selectedNodeIdsRef = useRef(selectedNodeIds)
  selectedNodeIdsRef.current = selectedNodeIds
  const structuralGraphRef = useRef<SelectableGraph | null>(null)
  const isReadOnlyRef = useRef(isReadOnly)
  isReadOnlyRef.current = isReadOnly
  const passthroughCounterRef = useRef(0)
  const customNodeCounterRef = useRef(0)
  const pendingNodeTypeRef = useRef(pendingNodeType)
  pendingNodeTypeRef.current = pendingNodeType
  const showShortcutsRef = useRef(showShortcuts)
  showShortcutsRef.current = showShortcuts
  const isResizing = useRef(false)
  const resizeStartX = useRef(0)
  const resizeStartWidth = useRef(0)

  const handleResizeStart = (e: React.MouseEvent) => {
    e.preventDefault()
    isResizing.current = true
    resizeStartX.current = e.clientX
    resizeStartWidth.current = panelWidth

    const onMove = (ev: MouseEvent) => {
      if (!isResizing.current) return
      const delta = resizeStartX.current - ev.clientX
      setPanelWidth(Math.max(180, Math.min(600, resizeStartWidth.current + delta)))
    }
    const onUp = () => {
      isResizing.current = false
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  useEffect(() => {
    setSelectableGraph(graph ? toSelectableGraph(graph) : null)
    setFilterQuery('')
    setExcludedNodeIds(new Set())
    setSelectedNodeIds(new Set())
    setSelectedNodeId(null)
    setAttrOverrides(new Map())
    setStructuralOps([])
    setUndoStack([])
    passthroughCounterRef.current = 0
    customNodeCounterRef.current = 0
    setPendingNodeType(null)
    if (graph) setShowDropzone(false)
  }, [graph])

  // Avionics-annunciator-style status line: a single line of feedback for
  // actions that were previously silent (a rejected rewire, nodes skipped
  // during bulk delete, a completed undo/copy) or destructive-looking
  // (a failed benchmark/export that no longer tears down the workspace,
  // see operationError below), auto-clearing after a few seconds.
  const announce = (text: string, tone: 'reject' | 'info' = 'info') => {
    if (announceTimeoutRef.current) clearTimeout(announceTimeoutRef.current)
    setAnnouncement({ text, tone })
    announceTimeoutRef.current = setTimeout(() => setAnnouncement(null), 4000)
  }

  useEffect(() => {
    if (operationError) announce(operationError.message, 'reject')
  }, [operationError])

  // Shared by the Delete keyboard shortcut and the "Delete all" bulk button --
  // covers a single selected node too, since selectedNodeIds always contains the
  // primary selection (see applySelection). Only unambiguous nodes (dead-end or a
  // single reconnect candidate) are deleted; anything with multiple candidate
  // inputs is silently skipped, same as the pre-v1.4 single-node keyboard path --
  // that ambiguity needs the picker in the Layer Inspector, which needs a single
  // click target to choose from.
  const applyBulkDelete = (ids: Set<string>) => {
    const g = structuralGraphRef.current
    if (!g || ids.size === 0) return
    const newOps: GraphEdit[] = []
    const newUndo: UndoEntry[] = []
    for (const nodeId of ids) {
      const eligibility = getDeleteEligibility(g, nodeId)
      if (!eligibility.eligible || eligibility.candidateInputs.length > 1) continue
      const nodeIndex = structuralNodeIndex(nodeId)
      if (nodeIndex === null) continue
      const keepInputPosition = eligibility.candidateInputs.length === 1 ? eligibility.candidateInputs[0].position : null
      newOps.push({ type: 'delete', nodeId, nodeIndex, keepInputPosition })
      newUndo.push({ kind: 'structural' })
    }
    const skipped = ids.size - newOps.length
    if (newOps.length === 0) {
      announce(`${skipped} of ${ids.size} node${ids.size === 1 ? '' : 's'} skipped: ambiguous or ineligible`, 'reject')
      return
    }
    if (skipped > 0) {
      announce(`Deleted ${newOps.length} of ${ids.size} -- ${skipped} skipped: ambiguous or ineligible`, 'reject')
    }
    setStructuralOps(prev => [...prev, ...newOps])
    setUndoStack(prev => [...prev, ...newUndo])
    setSelectedNodeId(null)
    setSelectedNodeIds(new Set())
  }

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (e.key === 'z' && (e.ctrlKey || e.metaKey) && tag !== 'INPUT' && tag !== 'TEXTAREA') {
        e.preventDefault()
        const stack = undoStackRef.current
        if (stack.length === 0) return
        const last = stack[stack.length - 1]
        setUndoStack(prev => prev.slice(0, -1))
        if (last.kind === 'attr') {
          setAttrOverrides(prev => {
            const next = new Map(prev)
            const existing = { ...(next.get(last.nodeId) ?? {}) }
            existing[last.attrName] = last.prevValue
            next.set(last.nodeId, existing)
            return next
          })
          announce(`Undid ${last.attrName} edit`)
        } else {
          // Structural ops are strict append-order, so popping the last undo
          // entry and the last op always correspond, regardless of delete/insert.
          setStructuralOps(prev => prev.slice(0, -1))
          announce('Undid last structural edit')
        }
        return
      }
      if (e.key === 'Delete' && tag !== 'INPUT' && tag !== 'TEXTAREA') {
        if (isReadOnlyRef.current) return
        applyBulkDelete(selectedNodeIdsRef.current)
        return
      }
      if (e.key === 'Escape') {
        if (showShortcutsRef.current) {
          setShowShortcuts(false)
          return
        }
        if (pendingNodeTypeRef.current) {
          setPendingNodeType(null)
          return
        }
        setFilterQuery('')
        setShowDropdown(false)
        setSelectedNodeIds(new Set())
        setSelectedNodeId(null)
        setSelectableGraph((sg) => (sg ? deselectAll(sg) : sg))
        return
      }
      if (e.key === '/' && tag !== 'INPUT' && tag !== 'TEXTAREA') {
        e.preventDefault()
        filterInputRef.current?.focus()
        return
      }
      if (e.key === '?' && tag !== 'INPUT' && tag !== 'TEXTAREA') {
        e.preventDefault()
        setShowShortcuts(v => !v)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const graphWithOverrides = useMemo((): SelectableGraph | null => {
    if (!selectableGraph || attrOverrides.size === 0) return selectableGraph
    return {
      ...selectableGraph,
      nodes: selectableGraph.nodes.map(n => {
        const overrides = attrOverrides.get(n.id)
        if (!overrides) return n
        const isModified = Object.entries(overrides).some(([k, v]) => v !== n.attributes[k])
        return { ...n, attributes: { ...n.attributes, ...overrides }, isModified }
      }),
    }
  }, [selectableGraph, attrOverrides])

  const graphWithStructuralEdits = useMemo((): SelectableGraph | null => {
    if (!graphWithOverrides || structuralOps.length === 0) return graphWithOverrides
    return structuralOps.reduce<SelectableGraph>(
      (g, op) =>
        op.type === 'delete'
          ? deleteNodeWithReconnect(g, op.nodeId, op.keepInputPosition)
          : op.type === 'insertPassthrough'
            ? insertPassthroughNode(g, op.targetNodeId, op.inputPosition, op.newNodeId)
            : op.type === 'rewire'
              ? rewireEdge(g, op.sourceNodeId, op.targetNodeId, op.inputPosition)
              : addCustomNode(g, op.newNodeId, op.opType, op.inputCount, op.position),
      graphWithOverrides,
    )
  }, [graphWithOverrides, structuralOps])
  structuralGraphRef.current = graphWithStructuralEdits

  const filteredGraph = useMemo(
    () => (graphWithStructuralEdits ? filterGraph(graphWithStructuralEdits, filterQuery) : null),
    [graphWithStructuralEdits, filterQuery],
  )

  const dropdownResults = useMemo(() => {
    if (!filterQuery.trim() || !filteredGraph) return []
    return filteredGraph.nodes
      .filter(n => !n.dimmed && n.opType !== 'Input' && n.opType !== 'Output')
      .slice(0, 8)
  }, [filterQuery, filteredGraph])

  const modelStats = useMemo(() => {
    if (!graphWithStructuralEdits) return null
    const opCounts = computeOpCounts(graphWithStructuralEdits.nodes)
    return {
      opCounts,
      totalNodes: graphWithStructuralEdits.nodes.filter(n => n.opType !== 'Input' && n.opType !== 'Output').length,
      graphDepth: computeGraphDepth(graphWithStructuralEdits),
      metadata: graph?.metadata,
    }
  }, [graphWithStructuralEdits, graph])

  const selectedNode: OnnxNode | null = selectedNodeId
    ? filteredGraph?.nodes.find((n) => n.id === selectedNodeId) ?? null
    : null

  const deleteEligibility = useMemo(() => {
    if (!selectedNodeId || !graphWithStructuralEdits) return undefined
    return getDeleteEligibility(graphWithStructuralEdits, selectedNodeId)
  }, [selectedNodeId, graphWithStructuralEdits])

  const multiSelection = useMemo(() => {
    if (selectedNodeIds.size <= 1 || !graphWithStructuralEdits) return undefined
    const nodes = graphWithStructuralEdits.nodes.filter((n) => selectedNodeIds.has(n.id))
    return {
      nodes,
      totalParams: nodes.reduce((s, n) => s + n.paramCount, 0),
      totalSizeMB: nodes.reduce((s, n) => s + n.estimatedSizeMB, 0),
    }
  }, [selectedNodeIds, graphWithStructuralEdits])

  const { ancestors, descendants } = useMemo(() => {
    if (!selectedNodeId || !graphWithStructuralEdits) return { ancestors: new Set<string>(), descendants: new Set<string>() }
    return {
      ancestors: getAncestors(graphWithStructuralEdits, selectedNodeId),
      descendants: getDescendants(graphWithStructuralEdits, selectedNodeId),
    }
  }, [selectedNodeId, graphWithStructuralEdits])

  const handleFilterChange = (value: string) => {
    setFilterQuery(value)
    setJumpToNodeId(null)
    setDropdownIndex(-1)
    setShowDropdown(value.trim().length > 0)
  }

  const handleDropdownSelect = (id: string) => {
    applySelection(new Set([id]), id)
    setJumpToNodeId(id)
    setShowDropdown(false)
  }

  const handleFilterKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setDropdownIndex(i => Math.min(i + 1, dropdownResults.length - 1))
      setShowDropdown(true)
      return
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      setDropdownIndex(i => Math.max(0, i - 1))
      return
    }
    if (e.key === 'Enter') {
      const target = dropdownResults[dropdownIndex] ?? dropdownResults[0]
      if (target) handleDropdownSelect(target.id)
      return
    }
    if (e.key === 'Escape') {
      setShowDropdown(false)
      setFilterQuery('')
      setSelectedNodeId(null)
    }
  }

  const handleFilterFocus = () => {
    setJumpToNodeId(null)
    if (filterQuery.trim()) setShowDropdown(true)
  }

  const handleFilterBlur = () => {
    setTimeout(() => setShowDropdown(false), 150)
  }

  const handleModelLoaded = (buffer: ArrayBuffer, filename: string) => {
    setSelectableGraph(null)
    loadModel(buffer, filename)
  }

  const applySelection = (ids: Set<string>, primary: string | null) => {
    setSelectedNodeIds(ids)
    setSelectedNodeId(primary)
    setSelectableGraph((sg) => (sg ? setMultiSelection(sg, ids) : sg))
  }

  const handleNodeSelect = (nodeId: string) => {
    applySelection(new Set([nodeId]), nodeId)
  }

  const handleNodeCtrlClick = (nodeId: string) => {
    const next = new Set(selectedNodeIds)
    let primary: string | null
    if (next.has(nodeId)) {
      next.delete(nodeId)
      primary = selectedNodeId === nodeId ? (next.values().next().value ?? null) : selectedNodeId
    } else {
      next.add(nodeId)
      primary = nodeId
    }
    applySelection(next, primary)
  }

  // Fired once when a Shift+drag selection box completes (see GraphCanvas's
  // onSelectionEnd) -- an empty box (nothing under it) clears the selection,
  // matching the usual click-on-empty-space-to-deselect convention.
  const handleBoxSelect = (nodeIds: string[]) => {
    applySelection(new Set(nodeIds), nodeIds[0] ?? null)
  }

  const handleBulkExclude = () => {
    setExcludedNodeIds((prev) => new Set([...prev, ...selectedNodeIds]))
    setSelectableGraph((sg) => (sg ? bulkExclude(sg, selectedNodeIds) : sg))
  }

  const handleBulkInclude = () => {
    setExcludedNodeIds((prev) => {
      const nextSet = new Set(prev)
      for (const id of selectedNodeIds) nextSet.delete(id)
      return nextSet
    })
    setSelectableGraph((sg) => (sg ? bulkInclude(sg, selectedNodeIds) : sg))
  }

  const handleToggleExclude = (nodeId: string) => {
    const willExclude = !excludedNodeIds.has(nodeId)
    setExcludedNodeIds((prev) => {
      const next = new Set(prev)
      if (willExclude) next.add(nodeId)
      else next.delete(nodeId)
      return next
    })
    setSelectableGraph((sg) => sg ? (willExclude ? excludeNode(sg, nodeId) : includeNode(sg, nodeId)) : sg)
  }

  const handleAttrEdit = (nodeId: string, attrName: string, newValue: string | number) => {
    const overrides = attrOverrides.get(nodeId)
    const originalNode = selectableGraph?.nodes.find(n => n.id === nodeId)
    const prevValue = overrides?.[attrName] ?? originalNode?.attributes[attrName]
    const resolved = prevValue !== undefined && typeof prevValue !== 'boolean' ? prevValue : ''
    const parsed = parseAttrEdit(String(newValue), resolved)
    if (parsed === resolved) return
    setUndoStack(prev => [...prev, { kind: 'attr', nodeId, attrName, prevValue: resolved }])
    setAttrOverrides(prev => {
      const next = new Map(prev)
      const existing = { ...(next.get(nodeId) ?? {}) }
      existing[attrName] = newValue
      next.set(nodeId, existing)
      return next
    })
  }

  const handleDeleteNode = (nodeId: string, keepInputPosition: number | null) => {
    const nodeIndex = structuralNodeIndex(nodeId)
    if (nodeIndex === null) return
    setStructuralOps(prev => [...prev, { type: 'delete', nodeId, nodeIndex, keepInputPosition }])
    setUndoStack(prev => [...prev, { kind: 'structural' }])
    if (selectedNodeId === nodeId || selectedNodeIds.has(nodeId)) {
      setSelectedNodeId(null)
      setSelectedNodeIds(new Set())
    }
  }

  const handleBulkDelete = () => applyBulkDelete(selectedNodeIds)

  // Fired by GraphCanvas when a drag-to-connect lands on a specific input handle.
  // validateRewire covers the self-connection/original-node/cycle checks; an
  // invalid drop is silently ignored, same convention as handleEdgeClick below.
  const handleRewire = (sourceNodeId: string, targetNodeId: string, inputPosition: number) => {
    if (!graphWithStructuralEdits) return
    const validation = validateRewire(graphWithStructuralEdits, sourceNodeId, targetNodeId, inputPosition)
    if (!validation.valid) {
      announce(`Rewire rejected: ${validation.reason ?? 'invalid connection'}`, 'reject')
      return
    }
    const sourceNodeIndex = structuralNodeIndex(sourceNodeId)
    const targetNodeIndex = structuralNodeIndex(targetNodeId)
    if (sourceNodeIndex === null || targetNodeIndex === null) return
    setStructuralOps(prev => [
      ...prev,
      { type: 'rewire', sourceNodeId, sourceNodeIndex, targetNodeId, targetNodeIndex, inputPosition },
    ])
    setUndoStack(prev => [...prev, { kind: 'structural' }])
  }

  const handleInsertPassthrough = (targetNodeId: string, inputPosition: number) => {
    const targetNodeIndex = structuralNodeIndex(targetNodeId)
    if (targetNodeIndex === null) return
    passthroughCounterRef.current += 1
    const newNodeId = `passthrough_${passthroughCounterRef.current}`
    setStructuralOps(prev => [
      ...prev,
      { type: 'insertPassthrough', targetNodeId, targetNodeIndex, inputPosition, newNodeId },
    ])
    setUndoStack(prev => [...prev, { kind: 'structural' }])
  }

  // Places a new, initially unconnected node on the canvas -- wiring its inputs and
  // outputs to the rest of the graph happens afterward via handleRewire (drag a
  // connection onto one of its input handles, or drag its own output onto some
  // other node's input handle). Attributes start empty; the curated op list is
  // chosen to need none, since there's no UI yet to add a new attribute key.
  // Picking an op type from the StatsBar doesn't place the node immediately --
  // it enters placement mode (a translucent preview follows the cursor in
  // GraphCanvas) and handlePlaceNode below commits it wherever the user clicks.
  const handleAddNode = (opType: string, inputCount: number) => {
    setPendingNodeType({ opType, inputCount })
  }

  const handlePlaceNode = (position: { x: number; y: number }) => {
    if (!pendingNodeType) return
    customNodeCounterRef.current += 1
    const newNodeIndex = customNodeCounterRef.current
    setStructuralOps(prev => [
      ...prev,
      { type: 'addNode', newNodeId: `custom_${newNodeIndex}`, newNodeIndex, opType: pendingNodeType.opType, inputCount: pendingNodeType.inputCount, position },
    ])
    setUndoStack(prev => [...prev, { kind: 'structural' }])
    setPendingNodeType(null)
  }

  // Edges are addressed by (target node, tensor name) rather than an edge id, since
  // React Flow's edge label is repurposed to show the tensor shape, not the tensor
  // name. Only edges targeting an original or custom-added node are insertable: a
  // synthetic passthrough source/target is a deliberate scope boundary -- chaining
  // edits onto a generated passthrough would require resolving ops against ops
  // rather than always against the original model.
  // Called from the confirm button in GraphCanvas's edge-click popover, not
  // from the raw click itself -- inserting a node used to happen on a single
  // unlabeled click, which both mutated the model with zero warning and was
  // undiscoverable as a feature in the first place.
  const handleEdgeClick = (targetNodeId: string, tensorName: string) => {
    if (!graphWithStructuralEdits) return
    if (structuralNodeIndex(targetNodeId) === null) {
      announce('Cannot insert here: not an editable target', 'reject')
      return
    }
    const targetNode = graphWithStructuralEdits.nodes.find(n => n.id === targetNodeId)
    if (!targetNode) return
    const inputPosition = targetNode.inputs.indexOf(tensorName)
    if (inputPosition === -1) return
    const incoming = graphWithStructuralEdits.edges.find(e => e.target === targetNodeId && e.label === tensorName)
    if (!incoming || incoming.source.startsWith('passthrough_')) {
      announce('Cannot insert onto a generated connection', 'reject')
      return
    }
    handleInsertPassthrough(targetNodeId, inputPosition)
    announce(`Inserted passthrough on ${tensorName} (Ctrl+Z to undo)`)
  }

  const handleReset = () => {
    setShowDropzone(true)
    setSelectableGraph(null)
  }

  const handleDownload = () => {
    exportModel().then((buf) => {
      const blob = new Blob([buf], { type: 'application/octet-stream' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const baseName = (graph?.modelName ?? 'model').replace(/\.[^.]+$/, '')
      a.download = baseName + '_original.onnx'
      a.click()
      URL.revokeObjectURL(url)
    })
  }

  const handleDownloadModified = () => {
    const overridesByIndex = new Map<number, Record<string, string | number>>()
    for (const [nodeId, overrides] of attrOverrides) {
      const nodeIndex = structuralNodeIndex(nodeId)
      if (nodeIndex === null) continue
      overridesByIndex.set(nodeIndex, overrides)
    }
    const writerOps: StructuralOp[] = structuralOps.map(op =>
      op.type === 'delete'
        ? { type: 'delete', nodeIndex: op.nodeIndex, keepInputPosition: op.keepInputPosition }
        : op.type === 'insertPassthrough'
          ? { type: 'insertPassthrough', targetNodeIndex: op.targetNodeIndex, inputPosition: op.inputPosition, newNodeName: op.newNodeId }
          : op.type === 'rewire'
            ? { type: 'rewire', targetNodeIndex: op.targetNodeIndex, inputPosition: op.inputPosition, sourceNodeIndex: op.sourceNodeIndex }
            : { type: 'addNode', newNodeIndex: op.newNodeIndex, opType: op.opType, inputCount: op.inputCount },
    )
    exportModifiedModel(overridesByIndex, writerOps).then((buf) => {
      const blob = new Blob([buf], { type: 'application/octet-stream' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const baseName = (graph?.modelName ?? 'model').replace(/\.[^.]+$/, '')
      a.download = baseName + '_modified.onnx'
      a.click()
      URL.revokeObjectURL(url)
    })
  }

  const benchmarkLabel = benchmarkResult
    ? `avg ${benchmarkResult.avgMs.toFixed(1)} ms / median ${benchmarkResult.medianMs.toFixed(1)} ms / min ${benchmarkResult.minMs.toFixed(1)} ms / max ${benchmarkResult.maxMs.toFixed(1)} ms (${benchmarkResult.runs} runs, batch 1, zeroed inputs)`
    : status === 'benchmarking' ? 'Running warmup + benchmark...' : null

  const dropzoneStatus =
    status === 'running' || status === 'benchmarking' || status === 'exporting' ? 'loading' :
    status === 'ready' ? 'ready' :
    status

  const isReady = status === 'ready' || status === 'benchmarking' || status === 'exporting'
  const workspaceShown = isReady && !!filteredGraph && !showDropzone

  // Lets a model be dropped at any time to replace the one currently open --
  // previously the only way back to a file picker was the "Load new" button.
  // Only active once a graph is already shown; ModelDropzone owns its own
  // drop target for the empty/loading/error states.
  const handleRootDragEnter = (e: React.DragEvent) => {
    if (!workspaceShown) return
    e.preventDefault()
    dragDepthRef.current += 1
    if (dragDepthRef.current === 1) setDragOverlay(true)
  }
  const handleRootDragOver = (e: React.DragEvent) => {
    if (!workspaceShown) return
    e.preventDefault()
  }
  const handleRootDragLeave = (e: React.DragEvent) => {
    if (!workspaceShown) return
    e.preventDefault()
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)
    if (dragDepthRef.current === 0) setDragOverlay(false)
  }
  const handleRootDrop = (e: React.DragEvent) => {
    if (!workspaceShown) return
    e.preventDefault()
    dragDepthRef.current = 0
    setDragOverlay(false)
    const file = e.dataTransfer.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) handleModelLoaded(reader.result, file.name)
    }
    reader.readAsArrayBuffer(file)
  }

  return (
    <div
      style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', background: 'var(--bg-base)', overflow: 'hidden' }}
      onDragEnter={handleRootDragEnter}
      onDragOver={handleRootDragOver}
      onDragLeave={handleRootDragLeave}
      onDrop={handleRootDrop}
    >
      {(showDropzone || !isReady) && (
        <div style={{ position: 'absolute', inset: 0, zIndex: 10 }}>
          <ModelDropzone
            onModelLoaded={handleModelLoaded}
            status={dropzoneStatus}
            error={error}
            progressLabel={progress?.stage ?? null}
            progressPercent={progress?.percent ?? null}
          />
        </div>
      )}
      {isReady && filteredGraph && !showDropzone && (
        <>
          <StatsBar
            modelName={graph?.modelName ?? ''}
            totalParams={graph?.totalParams ?? 0}
            totalSizeMB={graph?.totalSizeMB ?? 0}
            nodeCount={filteredGraph.nodes.filter(n => n.opType !== 'Input' && n.opType !== 'Output').length}
            quantizeEstimate={quantizeEstimate}
            filterQuery={filterQuery}
            onFilterChange={handleFilterChange}
            filterInputRef={filterInputRef}
            onFilterKeyDown={handleFilterKeyDown}
            onFilterFocus={handleFilterFocus}
            onFilterBlur={handleFilterBlur}
            dropdownResults={dropdownResults}
            showDropdown={showDropdown}
            dropdownIndex={dropdownIndex}
            onDropdownSelect={handleDropdownSelect}
            layoutDir={layoutDir}
            onLayoutToggle={() => setLayoutDir(d => d === 'TB' ? 'LR' : 'TB')}
            onBenchmark={() => runBenchmark(10)}
            benchmarkLabel={benchmarkLabel}
            isBenchmarking={status === 'benchmarking'}
            onDownload={handleDownload}
            canDownload={status === 'ready'}
            onDownloadModified={handleDownloadModified}
            canDownloadModified={status === 'ready' && !isReadOnly && (attrOverrides.size > 0 || structuralOps.length > 0)}
            onReset={handleReset}
            onAddNode={handleAddNode}
            isReadOnly={isReadOnly}
            editCount={attrOverrides.size + structuralOps.length}
          />
          <div
            data-testid="announcement"
            role="status"
            aria-live="polite"
            style={{
              height: 22,
              flexShrink: 0,
              display: 'flex',
              alignItems: 'center',
              padding: '0 24px',
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              letterSpacing: '0.04em',
              background: '#0E1114',
              borderBottom: announcement ? '1px solid rgba(255,255,255,0.08)' : '1px solid transparent',
              color: announcement?.tone === 'reject' ? 'var(--color-error)' : 'var(--color-amber)',
              transition: 'color 140ms ease, border-color 140ms ease',
            }}
          >
            {announcement?.text ?? ''}
          </div>
          <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
            <div style={{ flex: 1, height: '100%' }}>
              <GraphCanvas
                onnxNodes={filteredGraph.nodes}
                onnxEdges={filteredGraph.edges}
                selectedNodeId={selectedNodeId}
                onNodeSelect={handleNodeSelect}
                onNodeCtrlClick={handleNodeCtrlClick}
                onBoxSelect={handleBoxSelect}
                onEdgeClick={isReadOnly ? undefined : handleEdgeClick}
                onRewire={isReadOnly ? undefined : handleRewire}
                pendingNodeType={isReadOnly ? null : pendingNodeType}
                onPlaceNode={isReadOnly ? undefined : handlePlaceNode}
                jumpToNodeId={jumpToNodeId}
                traceAncestors={ancestors}
                traceDescendants={descendants}
                layoutDir={layoutDir}
              />
            </div>
            <div style={{ width: panelWidth, flexShrink: 0, borderLeft: '1px solid rgba(255,255,255,0.1)', height: '100%', position: 'relative', display: 'flex' }}>
              <div
                onMouseDown={handleResizeStart}
                style={{
                  position: 'absolute',
                  left: 0,
                  top: 0,
                  bottom: 0,
                  width: 4,
                  cursor: 'col-resize',
                  zIndex: 1,
                  background: 'transparent',
                  transition: 'background 140ms ease',
                }}
                onMouseEnter={e => { (e.currentTarget as HTMLDivElement).style.background = 'rgba(255,176,0,0.25)' }}
                onMouseLeave={e => { (e.currentTarget as HTMLDivElement).style.background = 'transparent' }}
              />
              <div style={{ flex: 1, overflow: 'hidden' }}>
                <LayerInspector node={selectedNode} onToggleExclude={handleToggleExclude} quantizeEstimate={quantizeEstimate} modelStats={modelStats} multiSelection={multiSelection} onBulkExclude={handleBulkExclude} onBulkInclude={handleBulkInclude} onBulkDelete={isReadOnly ? undefined : handleBulkDelete} onAttrEdit={isReadOnly ? undefined : handleAttrEdit} onDeleteNode={isReadOnly ? undefined : handleDeleteNode} deleteEligibility={isReadOnly ? undefined : deleteEligibility} onCopy={() => announce('Copied to clipboard')} />
              </div>
            </div>
          </div>
        </>
      )}
      {showShortcuts && <ShortcutsOverlay onClose={() => setShowShortcuts(false)} />}
      {dragOverlay && (
        <div
          data-testid="drag-replace-overlay"
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 3000,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            background: 'rgba(18,22,26,0.92)',
            border: '2px dashed var(--color-amber)',
            pointerEvents: 'none',
          }}
        >
          <span style={{ color: 'var(--color-amber)', textTransform: 'uppercase', letterSpacing: '0.16em', fontSize: 15, fontFamily: 'var(--font-mono)' }}>
            Drop to replace current model
          </span>
          {attrOverrides.size + structuralOps.length > 0 && (
            <span style={{ color: 'var(--color-error)', fontSize: 11, letterSpacing: '0.06em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
              Unsaved edits will be lost
            </span>
          )}
        </div>
      )}
    </div>
  )
}

export default App
