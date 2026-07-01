import { useEffect, useMemo, useRef, useState } from 'react'
import { ModelDropzone } from './components/ModelDropzone'
import { GraphCanvas } from './components/GraphCanvas'
import { LayerInspector } from './components/LayerInspector'
import { useOnnxWorker } from './hooks/useOnnxWorker'
import { toSelectableGraph, deselectAll, filterGraph, excludeNode, includeNode, setMultiSelection, bulkExclude, bulkInclude, computeOpCounts, computeGraphDepth, getAncestors, getDescendants, type SelectableGraph } from './lib/graphUtils'
import { formatQuantizeEstimate } from './lib/quantize'
import type { OnnxNode } from './lib/onnxTypes'
import type { QuantizeEstimate } from './hooks/useOnnxWorker'
import './index.css'

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`
  return String(n)
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
  onDownload: () => void
  canDownload: boolean
  onReset: () => void
}

function StatsBar({ modelName, totalParams, totalSizeMB, nodeCount, quantizeEstimate, filterQuery, onFilterChange, filterInputRef, onFilterKeyDown, onFilterFocus, onFilterBlur, dropdownResults, showDropdown, dropdownIndex, onDropdownSelect, layoutDir, onLayoutToggle, onBenchmark, benchmarkLabel, onDownload, canDownload, onReset }: StatsBarProps) {
  const quantizeLabel = formatQuantizeEstimate(quantizeEstimate)
  return (
    <div style={{
      height: 36,
      background: '#0E1114',
      borderBottom: '1px solid rgba(255,255,255,0.1)',
      display: 'flex',
      alignItems: 'center',
      padding: '0 16px',
      gap: 24,
      fontFamily: 'var(--font-mono)',
      fontSize: 11,
      color: 'var(--text-secondary)',
      flexShrink: 0,
      letterSpacing: '0.06em',
      overflow: 'hidden',
    }}>
      <span style={{ color: 'var(--text-primary)', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {modelName}
      </span>
      <span>{formatNumber(totalParams)} PARAMS</span>
      <span>{totalSizeMB.toFixed(1)} MB</span>
      {quantizeLabel && (
        <span style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '0.06em' }}>
          {quantizeLabel}
        </span>
      )}
      <span>{nodeCount} NODES</span>
      <div style={{ position: 'relative' }}>
        <input
          ref={filterInputRef}
          type="text"
          value={filterQuery}
          onChange={(e) => onFilterChange(e.target.value)}
          onKeyDown={onFilterKeyDown}
          onFocus={onFilterFocus}
          onBlur={onFilterBlur}
          placeholder="FILTER NODES"
          style={{
            background: 'transparent',
            border: '1px solid rgba(255,255,255,0.15)',
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            letterSpacing: '0.06em',
            padding: '2px 8px',
            width: 160,
            borderRadius: 2,
          }}
        />
        {showDropdown && dropdownResults.length > 0 && (
          <div data-testid="search-dropdown" role="listbox" style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            width: 280,
            background: '#1C2128',
            border: '1px solid rgba(255,255,255,0.15)',
            borderRadius: 2,
            zIndex: 1000,
            marginTop: 2,
            maxHeight: 240,
            overflowY: 'auto',
          }}>
            {dropdownResults.map((node, i) => (
              <div
                key={node.id}
                role="option"
                aria-selected={i === dropdownIndex}
                data-testid="search-result"
                onMouseDown={() => onDropdownSelect(node.id)}
                style={{
                  padding: '6px 12px',
                  cursor: 'pointer',
                  background: i === dropdownIndex ? 'rgba(255,176,0,0.12)' : 'transparent',
                  borderLeft: i === dropdownIndex ? '2px solid #FFB000' : '2px solid transparent',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                <span style={{ color: 'var(--text-primary)', fontSize: 11 }}>{node.opType}</span>
                {node.paramCount > 0 && (
                  <span style={{ color: 'var(--text-dim)', fontSize: 10 }}>
                    {node.paramCount.toLocaleString()}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
        <button
          onClick={onLayoutToggle}
          style={{
            background: 'none',
            border: '1px solid rgba(255,255,255,0.2)',
            borderRadius: 2,
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            letterSpacing: '0.06em',
            padding: '4px 16px',
            cursor: 'pointer',
            textTransform: 'uppercase',
          }}
        >
          {layoutDir}
        </button>
        {benchmarkLabel && (
          <span style={{ color: 'var(--color-green)' }}>{benchmarkLabel}</span>
        )}
        <button
          onClick={onBenchmark}
          style={{
            background: 'none',
            border: '1px solid rgba(255,255,255,0.2)',
            borderRadius: 2,
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            letterSpacing: '0.06em',
            padding: '4px 16px',
            cursor: 'pointer',
            textTransform: 'uppercase',
          }}
        >
          Benchmark
        </button>
        {canDownload && (
          <button
            onClick={onDownload}
            style={{
              background: 'none',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: 2,
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              letterSpacing: '0.06em',
              padding: '4px 16px',
              cursor: 'pointer',
              textTransform: 'uppercase',
            }}
          >
            Download
          </button>
        )}
        <button
          onClick={onReset}
          style={{
            background: 'none',
            border: '1px solid rgba(255,255,255,0.2)',
            borderRadius: 2,
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            letterSpacing: '0.06em',
            padding: '4px 16px',
            cursor: 'pointer',
            textTransform: 'uppercase',
          }}
        >
          Load new
        </button>
      </div>
    </div>
  )
}

function App() {
  const { loadModel, runBenchmark, exportModel, graph, status, error, progress, benchmarkResult, quantizeEstimate } = useOnnxWorker()
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
  const filterInputRef = useRef<HTMLInputElement>(null)
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
    if (graph) setShowDropzone(false)
  }, [graph])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (e.key === 'Escape') {
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
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const filteredGraph = useMemo(
    () => (selectableGraph ? filterGraph(selectableGraph, filterQuery) : null),
    [selectableGraph, filterQuery],
  )

  const dropdownResults = useMemo(() => {
    if (!filterQuery.trim() || !filteredGraph) return []
    return filteredGraph.nodes
      .filter(n => !n.dimmed && n.opType !== 'Input' && n.opType !== 'Output')
      .slice(0, 8)
  }, [filterQuery, filteredGraph])

  const modelStats = useMemo(() => {
    if (!selectableGraph) return null
    const opCounts = computeOpCounts(selectableGraph.nodes)
    return {
      opCounts,
      totalNodes: selectableGraph.nodes.filter(n => n.opType !== 'Input' && n.opType !== 'Output').length,
      graphDepth: computeGraphDepth(selectableGraph),
    }
  }, [selectableGraph])

  const selectedNode: OnnxNode | null = selectedNodeId
    ? filteredGraph?.nodes.find((n) => n.id === selectedNodeId) ?? null
    : null

  const multiSelection = useMemo(() => {
    if (selectedNodeIds.size <= 1 || !selectableGraph) return undefined
    const nodes = selectableGraph.nodes.filter((n) => selectedNodeIds.has(n.id))
    return {
      nodes,
      totalParams: nodes.reduce((s, n) => s + n.paramCount, 0),
      totalSizeMB: nodes.reduce((s, n) => s + n.estimatedSizeMB, 0),
    }
  }, [selectedNodeIds, selectableGraph])

  const { ancestors, descendants } = useMemo(() => {
    if (!selectedNodeId || !selectableGraph) return { ancestors: new Set<string>(), descendants: new Set<string>() }
    return {
      ancestors: getAncestors(selectableGraph, selectedNodeId),
      descendants: getDescendants(selectableGraph, selectedNodeId),
    }
  }, [selectedNodeId, selectableGraph])

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
      a.download = baseName + '_export.onnx'
      a.click()
      URL.revokeObjectURL(url)
    })
  }

  const benchmarkLabel = benchmarkResult
    ? `avg ${benchmarkResult.avgMs.toFixed(1)} ms / min ${benchmarkResult.minMs.toFixed(1)} ms / max ${benchmarkResult.maxMs.toFixed(1)} ms (${benchmarkResult.runs} runs)`
    : status === 'benchmarking' ? 'Benchmarking...' : null

  const dropzoneStatus =
    status === 'running' || status === 'benchmarking' || status === 'exporting' ? 'loading' :
    status === 'ready' ? 'ready' :
    status

  const isReady = status === 'ready' || status === 'benchmarking' || status === 'exporting'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', background: 'var(--bg-base)', overflow: 'hidden' }}>
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
            onDownload={handleDownload}
            canDownload={status === 'ready'}
            onReset={handleReset}
          />
          <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
            <div style={{ flex: 1, height: '100%' }}>
              <GraphCanvas
                onnxNodes={filteredGraph.nodes}
                onnxEdges={filteredGraph.edges}
                selectedNodeId={selectedNodeId}
                onNodeSelect={handleNodeSelect}
                onNodeCtrlClick={handleNodeCtrlClick}
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
                }}
                onMouseEnter={e => { (e.currentTarget as HTMLDivElement).style.background = 'rgba(255,176,0,0.25)' }}
                onMouseLeave={e => { (e.currentTarget as HTMLDivElement).style.background = 'transparent' }}
              />
              <div style={{ flex: 1, overflow: 'hidden' }}>
                <LayerInspector node={selectedNode} onToggleExclude={handleToggleExclude} quantizeEstimate={quantizeEstimate} modelStats={modelStats} multiSelection={multiSelection} onBulkExclude={handleBulkExclude} onBulkInclude={handleBulkInclude} />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default App
