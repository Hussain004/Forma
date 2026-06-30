import { useEffect, useState } from 'react'
import { ModelDropzone } from './components/ModelDropzone'
import { GraphCanvas } from './components/GraphCanvas'
import { LayerInspector } from './components/LayerInspector'
import { useOnnxWorker } from './hooks/useOnnxWorker'
import { toSelectableGraph, selectNode, deselectAll, type SelectableGraph } from './lib/graphUtils'
import type { OnnxNode } from './lib/onnxTypes'
import './index.css'

function App() {
  const { loadModel, graph, status, error, progress } = useOnnxWorker()
  const [selectableGraph, setSelectableGraph] = useState<SelectableGraph | null>(null)

  // Convert incoming OnnxGraph to SelectableGraph whenever graph changes.
  useEffect(() => {
    setSelectableGraph(graph ? toSelectableGraph(graph) : null)
  }, [graph])

  const selectedNode: OnnxNode | null =
    selectableGraph?.nodes.find((n) => n.selected) ?? null

  const selectedNodeId: string | null = selectedNode?.id ?? null

  const handleModelLoaded = (buffer: ArrayBuffer, filename: string) => {
    setSelectableGraph(null)
    loadModel(buffer, filename)
  }

  const handleNodeSelect = (nodeId: string) => {
    setSelectableGraph((sg) => sg ? selectNode(deselectAll(sg), nodeId) : sg)
  }

  const dropzoneStatus =
    status === 'running' ? 'loading' :
    status === 'ready' ? 'ready' :
    status

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', background: 'var(--bg-base)', overflow: 'hidden' }}>
      {status !== 'ready' && (
        <ModelDropzone
          onModelLoaded={handleModelLoaded}
          status={dropzoneStatus}
          error={error}
          progressLabel={progress?.stage ?? null}
        />
      )}
      {status === 'ready' && selectableGraph && (
        <>
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
        </>
      )}
    </div>
  )
}

export default App
