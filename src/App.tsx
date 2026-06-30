import { useEffect, useMemo, useRef, useState } from 'react'
import { ModelDropzone } from './components/ModelDropzone'
import { GraphCanvas } from './components/GraphCanvas'
import { LayerInspector } from './components/LayerInspector'
import { useOnnxWorker } from './hooks/useOnnxWorker'
import { toSelectableGraph, selectNode, deselectAll, filterGraph, excludeNode, includeNode, type SelectableGraph } from './lib/graphUtils'
import type { OnnxNode } from './lib/onnxTypes'
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
  filterQuery: string
  onFilterChange: (value: string) => void
  onBenchmark: () => void
  benchmarkLabel: string | null
  onReset: () => void
}

function StatsBar({ modelName, totalParams, totalSizeMB, nodeCount, filterQuery, onFilterChange, onBenchmark, benchmarkLabel, onReset }: StatsBarProps) {
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
      <span>{nodeCount} NODES</span>
      <input
        type="text"
        value={filterQuery}
        onChange={(e) => onFilterChange(e.target.value)}
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
      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
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
  const { loadModel, runBenchmark, graph, status, error, progress, benchmarkResult } = useOnnxWorker()
  const [selectableGraph, setSelectableGraph] = useState<SelectableGraph | null>(null)
  const [filterQuery, setFilterQuery] = useState('')
  const [excludedNodeIds, setExcludedNodeIds] = useState<Set<string>>(new Set())
  const [showDropzone, setShowDropzone] = useState(true)
  const [panelWidth, setPanelWidth] = useState(280)
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
    if (graph) setShowDropzone(false)
  }, [graph])

  const filteredGraph = useMemo(
    () => (selectableGraph ? filterGraph(selectableGraph, filterQuery) : null),
    [selectableGraph, filterQuery],
  )

  const selectedNode: OnnxNode | null = filteredGraph?.nodes.find((n) => n.selected) ?? null
  const selectedNodeId: string | null = selectedNode?.id ?? null

  const handleModelLoaded = (buffer: ArrayBuffer, filename: string) => {
    setSelectableGraph(null)
    loadModel(buffer, filename)
  }

  const handleNodeSelect = (nodeId: string) => {
    setSelectableGraph((sg) => sg ? selectNode(deselectAll(sg), nodeId) : sg)
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
            filterQuery={filterQuery}
            onFilterChange={setFilterQuery}
            onBenchmark={() => runBenchmark(10)}
            benchmarkLabel={benchmarkLabel}
            onReset={handleReset}
          />
          <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
            <div style={{ flex: 1, height: '100%' }}>
              <GraphCanvas
                onnxNodes={filteredGraph.nodes}
                onnxEdges={filteredGraph.edges}
                selectedNodeId={selectedNodeId}
                onNodeSelect={handleNodeSelect}
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
                <LayerInspector node={selectedNode} onToggleExclude={handleToggleExclude} />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default App
