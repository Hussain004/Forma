import { useState } from 'react'
import { ModelDropzone } from './components/ModelDropzone'
import { GraphCanvas } from './components/GraphCanvas'
import { LayerInspector } from './components/LayerInspector'
import { useOnnxWorker } from './hooks/useOnnxWorker'
import type { OnnxNode } from './lib/onnxTypes'
import './index.css'

function App() {
  const { loadModel, graph, status, error, progress } = useOnnxWorker()
  const [selectedNode, setSelectedNode] = useState<OnnxNode | null>(null)

  const handleModelLoaded = (buffer: ArrayBuffer, filename: string) => {
    setSelectedNode(null)
    loadModel(buffer, filename)
  }

  const handleNodeSelect = (nodeId: string) => {
    const node = graph?.nodes.find((n) => n.id === nodeId) ?? null
    setSelectedNode(node)
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
      {status === 'ready' && graph && (
        <>
          <div style={{ flex: 1, height: '100%' }}>
            <GraphCanvas onnxNodes={graph.nodes} onnxEdges={graph.edges} onNodeSelect={handleNodeSelect} />
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
