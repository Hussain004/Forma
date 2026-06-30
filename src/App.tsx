import { useEffect, useState } from 'react'
import { ModelDropzone } from './components/ModelDropzone'
import { GraphCanvas } from './components/GraphCanvas'
import { LayerInspector } from './components/LayerInspector'
import { useOnnxWorker } from './hooks/useOnnxWorker'
import { toSelectableGraph, selectNode, deselectAll, type SelectableGraph } from './lib/graphUtils'
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
  onBenchmark: () => void
  benchmarkLabel: string | null
  onReset: () => void
}

function StatsBar({ modelName, totalParams, totalSizeMB, nodeCount, onBenchmark, benchmarkLabel, onReset }: StatsBarProps) {
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
      <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
        {benchmarkLabel && (
          <span style={{ color: '#4A5D23' }}>{benchmarkLabel}</span>
        )}
        <button
          onClick={onBenchmark}
          style={{
            background: 'none',
            border: '1px solid rgba(255,255,255,0.15)',
            borderRadius: 2,
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            letterSpacing: '0.06em',
            padding: '2px 10px',
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
            border: '1px solid rgba(255,255,255,0.15)',
            borderRadius: 2,
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            letterSpacing: '0.06em',
            padding: '2px 10px',
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
  const [showDropzone, setShowDropzone] = useState(true)

  useEffect(() => {
    setSelectableGraph(graph ? toSelectableGraph(graph) : null)
    if (graph) setShowDropzone(false)
  }, [graph])

  const selectedNode: OnnxNode | null = selectableGraph?.nodes.find((n) => n.selected) ?? null
  const selectedNodeId: string | null = selectedNode?.id ?? null

  const handleModelLoaded = (buffer: ArrayBuffer, filename: string) => {
    setSelectableGraph(null)
    loadModel(buffer, filename)
  }

  const handleNodeSelect = (nodeId: string) => {
    setSelectableGraph((sg) => sg ? selectNode(deselectAll(sg), nodeId) : sg)
  }

  const handleReset = () => {
    setShowDropzone(true)
    setSelectableGraph(null)
  }

  const benchmarkLabel = benchmarkResult
    ? `avg ${benchmarkResult.avgMs.toFixed(1)} ms / min ${benchmarkResult.minMs.toFixed(1)} ms / max ${benchmarkResult.maxMs.toFixed(1)} ms (${benchmarkResult.runs} runs)`
    : status === 'benchmarking' ? 'Benchmarking...' : null

  const dropzoneStatus =
    status === 'running' || status === 'benchmarking' ? 'loading' :
    status === 'ready' ? 'ready' :
    status

  const isReady = status === 'ready' || status === 'benchmarking'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', background: 'var(--bg-base)', overflow: 'hidden' }}>
      {(showDropzone || !isReady) && (
        <div style={{ position: 'absolute', inset: 0, zIndex: 10 }}>
          <ModelDropzone
            onModelLoaded={handleModelLoaded}
            status={dropzoneStatus}
            error={error}
            progressLabel={progress?.stage ?? null}
          />
        </div>
      )}
      {isReady && selectableGraph && !showDropzone && (
        <>
          <StatsBar
            modelName={graph?.modelName ?? ''}
            totalParams={graph?.totalParams ?? 0}
            totalSizeMB={graph?.totalSizeMB ?? 0}
            nodeCount={selectableGraph.nodes.filter(n => n.opType !== 'Input' && n.opType !== 'Output').length}
            onBenchmark={() => runBenchmark(10)}
            benchmarkLabel={benchmarkLabel}
            onReset={handleReset}
          />
          <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
            <div style={{ flex: 1, height: '100%' }}>
              <GraphCanvas
                onnxNodes={selectableGraph.nodes}
                onnxEdges={selectableGraph.edges}
                selectedNodeId={selectedNodeId}
                onNodeSelect={handleNodeSelect}
              />
            </div>
            <div style={{ width: 280, borderLeft: '1px solid rgba(255,255,255,0.1)', height: '100%' }}>
              <LayerInspector node={selectedNode} />
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default App
