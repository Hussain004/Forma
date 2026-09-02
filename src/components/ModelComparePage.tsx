import { useMemo, useRef, useState } from 'react'
import { useOnnxWorker } from '../hooks/useOnnxWorker'
import { compareOutputs, type SideRunResult } from '../lib/validationUtils'
import { hashModelBuffer, createShareHash } from '../lib/shareLinks'
import {
  compareModels,
  formatComparisonReport,
  attributeOnlyEditRecipe,
  type ModelComparison,
} from '../lib/modelComparison'
import type { BenchmarkResult } from '../hooks/useOnnxWorker'

interface ModelComparePageProps {
  onBack: () => void
}

function formatMetric(value: number | undefined): string {
  if (value === undefined) return 'n/a'
  if (value === 0) return '0'
  return Math.abs(value) < 0.001 ? value.toExponential(2) : value.toFixed(4)
}

function formatShapeCell(shape: (number | string)[] | undefined): string {
  return shape ? `[${shape.join(', ')}]` : '?'
}

interface CompareSlotProps {
  label: string
  status: 'idle' | 'loading' | 'ready' | 'error'
  modelName?: string
  error?: string | null
  onFile: (buffer: ArrayBuffer, filename: string) => void
}

// A compact, side-by-side drop target. Unlike ModelDropzone/DropZone
// (both fixed-position, full-screen), this needs to sit two-up in a normal
// flow layout, so it's its own small component rather than a reuse of either.
function CompareSlot({ label, status, modelName, error, onFile }: CompareSlotProps) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const readFile = (file: File) => {
    const reader = new FileReader()
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) onFile(reader.result, file.name)
    }
    reader.readAsArrayBuffer(file)
  }

  return (
    <div
      data-testid={`compare-slot-${label.toLowerCase()}`}
      onClick={() => status !== 'loading' && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); if (!dragging) setDragging(true) }}
      onDragLeave={(e) => { e.preventDefault(); setDragging(false) }}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        const file = e.dataTransfer.files?.[0]
        if (file) readFile(file)
      }}
      style={{
        flex: 1,
        minHeight: 140,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        padding: 20,
        cursor: status === 'loading' ? 'default' : 'pointer',
        border: dragging ? '1px dashed var(--color-amber)' : '1px dashed rgba(255,255,255,0.15)',
        borderRadius: 2,
        background: status === 'ready' ? 'rgba(82,197,122,0.05)' : 'transparent',
        transition: 'background 140ms ease, border-color 140ms ease',
        fontFamily: 'var(--font-mono)',
        textAlign: 'center',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".onnx,.tflite"
        onChange={(e) => { const file = e.target.files?.[0]; if (file) readFile(file); e.target.value = '' }}
        style={{ display: 'none' }}
      />
      <span style={{ color: 'var(--text-dim)', fontSize: 12, letterSpacing: '0.1em', textTransform: 'uppercase' }}>{label}</span>
      {status === 'loading' && <span style={{ color: 'var(--color-amber)', fontSize: 13 }}>Loading...</span>}
      {status === 'error' && <span style={{ color: 'var(--color-error)', fontSize: 13 }}>{error ?? 'Load failed'}</span>}
      {status === 'ready' && <span style={{ color: 'var(--color-green)', fontSize: 12 }}>{modelName}</span>}
      {status === 'idle' && <span style={{ color: 'var(--text-secondary)', fontSize: 13 }}>Drop .onnx model or click to browse</span>}
    </div>
  )
}

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <span style={{ color: 'var(--color-amber)', fontSize: 12, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
      {children}
    </span>
  )
}

export function ModelComparePage({ onBack }: ModelComparePageProps) {
  const baseline = useOnnxWorker()
  const candidate = useOnnxWorker()
  const [baselineHash, setBaselineHash] = useState<string | null>(null)
  const [baselineError, setBaselineError] = useState<string | null>(null)
  const [candidateError, setCandidateError] = useState<string | null>(null)
  type LatencySide = BenchmarkResult | { error: string }
  const [latencyResult, setLatencyResult] = useState<{ baseline: LatencySide; candidate: LatencySide } | null>(null)
  const [isBenchmarking, setIsBenchmarking] = useState(false)
  const [outputResult, setOutputResult] = useState<{ baseline: SideRunResult; candidate: SideRunResult } | null>(null)
  const [isRunningOutputs, setIsRunningOutputs] = useState(false)
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const [recipeMessage, setRecipeMessage] = useState<string | null>(null)

  const handleBaselineFile = async (buffer: ArrayBuffer, filename: string) => {
    setBaselineError(null)
    try {
      setBaselineHash(await hashModelBuffer(buffer))
    } catch {
      setBaselineHash(null)
    }
    setLatencyResult(null)
    setOutputResult(null)
    baseline.loadModel(buffer, filename)
  }

  const handleCandidateFile = (buffer: ArrayBuffer, filename: string) => {
    setCandidateError(null)
    setLatencyResult(null)
    setOutputResult(null)
    candidate.loadModel(buffer, filename)
  }

  const bothLoaded = baseline.status === 'ready' && candidate.status === 'ready' && !!baseline.graph && !!candidate.graph
  const formatIssue = bothLoaded && (baseline.graph!.format === 'tflite' || candidate.graph!.format === 'tflite')
  const canCompare = bothLoaded && !formatIssue

  const comparison: ModelComparison | null = useMemo(() => {
    if (!canCompare) return null
    return compareModels(baseline.graph!, candidate.graph!)
  }, [canCompare, baseline.graph, candidate.graph])

  const outputComparisons = useMemo(() => {
    if (!outputResult || !outputResult.baseline.inferenceOk || !outputResult.candidate.inferenceOk) return null
    return compareOutputs(outputResult.baseline.outputs, outputResult.candidate.outputs)
  }, [outputResult])

  const editRecipeEntries = useMemo(() => (comparison ? attributeOnlyEditRecipe(comparison) : null), [comparison])

  const handleRunLatency = () => {
    setIsBenchmarking(true)
    setStatusMessage(null)
    // allSettled, not all: an edit that breaks the candidate's shape contract
    // (e.g. a stride change the model's own Reshape wasn't updated for) is a
    // real result worth showing, not a reason to also lose the baseline's.
    Promise.allSettled([baseline.runBenchmark(10), candidate.runBenchmark(10)]).then(([b, c]) => {
      const toSide = (r: PromiseSettledResult<BenchmarkResult>): LatencySide =>
        r.status === 'fulfilled' ? r.value : { error: r.reason instanceof Error ? r.reason.message : 'Benchmark failed' }
      setLatencyResult({ baseline: toSide(b), candidate: toSide(c) })
      setIsBenchmarking(false)
    })
  }

  const handleRunOutputs = () => {
    setIsRunningOutputs(true)
    setStatusMessage(null)
    Promise.all([baseline.runGeneratedInference(), candidate.runGeneratedInference()])
      .then(([b, c]) => setOutputResult({ baseline: b, candidate: c }))
      .catch((err) => setStatusMessage(err instanceof Error ? err.message : 'Output comparison failed'))
      .finally(() => setIsRunningOutputs(false))
  }

  const handleExportReport = () => {
    if (!comparison || !baseline.graph || !candidate.graph) return
    const latencyOk = latencyResult && !('error' in latencyResult.baseline) && !('error' in latencyResult.candidate)
      ? (latencyResult as { baseline: BenchmarkResult; candidate: BenchmarkResult })
      : undefined
    const text = formatComparisonReport(comparison, baseline.graph.modelName, candidate.graph.modelName, {
      latency: latencyOk,
      outputs: outputComparisons?.map((c) => ({ name: c.name, shapeMatch: c.shapeMatch, maxAbsErr: c.maxAbsErr, cosineSim: c.cosineSim })),
    })
    const blob = new Blob([text], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'model_comparison_report.txt'
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleExportEditRecipe = async () => {
    if (!editRecipeEntries || !baselineHash || !baseline.graph) return
    try {
      const hash = createShareHash(baselineHash, baseline.graph.modelName, editRecipeEntries)
      const url = new URL(window.location.href)
      url.hash = hash
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard access is unavailable')
      await navigator.clipboard.writeText(url.toString())
      setRecipeMessage(`Edit recipe link copied (${editRecipeEntries.length} attribute change${editRecipeEntries.length === 1 ? '' : 's'}). Open it and load ${baseline.graph.modelName} to apply.`)
    } catch (recipeErr) {
      setRecipeMessage(recipeErr instanceof Error ? recipeErr.message : 'Could not create the edit recipe link')
    }
  }

  return (
    <div
      data-testid="model-compare-page"
      style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', background: 'var(--bg-base)', overflow: 'hidden', fontFamily: 'var(--font-mono)' }}
    >
      <div style={{ height: 52, flexShrink: 0, display: 'flex', alignItems: 'center', padding: '0 24px', gap: 16, background: '#0E1114', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
        <span style={{ color: 'var(--text-primary)', fontWeight: 500, letterSpacing: '0.06em' }}>Model Comparison</span>
        <button data-testid="compare-back" onClick={onBack} className="btn-bar btn-ghost" style={{ marginLeft: 'auto' }}>
          Back to editor
        </button>
      </div>

      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: 24, display: 'flex', flexDirection: 'column', gap: 24 }}>
        <div style={{ display: 'flex', gap: 16 }}>
          <CompareSlot
            label="Baseline"
            status={baseline.status === 'error' ? 'error' : baseline.status === 'loading' ? 'loading' : baseline.status !== 'idle' ? 'ready' : 'idle'}
            modelName={baseline.graph?.modelName}
            error={baseline.error ?? baselineError}
            onFile={(buf, name) => { void handleBaselineFile(buf, name) }}
          />
          <CompareSlot
            label="Candidate"
            status={candidate.status === 'error' ? 'error' : candidate.status === 'loading' ? 'loading' : candidate.status !== 'idle' ? 'ready' : 'idle'}
            modelName={candidate.graph?.modelName}
            error={candidate.error ?? candidateError}
            onFile={handleCandidateFile}
          />
        </div>

        {formatIssue && (
          <span style={{ color: 'var(--color-error)', fontSize: 13 }}>
            Model comparison only supports ONNX models. TFLite is read-only, so drop an .onnx file for both sides.
          </span>
        )}

        {statusMessage && <span style={{ color: 'var(--color-error)', fontSize: 13 }}>{statusMessage}</span>}

        {comparison && baseline.graph && candidate.graph && (
          <>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <SectionHeader>Summary</SectionHeader>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, auto)', columnGap: 24, rowGap: 2, fontSize: 13, color: 'var(--text-secondary)' }}>
                <span>Nodes</span><span>{comparison.baselineNodeCount} -&gt; {comparison.candidateNodeCount}</span>
                <span>Matched nodes</span><span>{comparison.nodeMatch.matches.length} (by {comparison.nodeMatch.matchedByName ? 'name' : 'op-type position'})</span>
                <span>Added / removed nodes</span><span>{comparison.nodeMatch.addedNodes.length} / {comparison.nodeMatch.removedNodes.length}</span>
              </div>
            </div>

            {comparison.metadata.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <SectionHeader>Metadata changes</SectionHeader>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, auto)', columnGap: 16, rowGap: 2, fontSize: 13, color: 'var(--text-secondary)' }}>
                  {comparison.metadata.map((m) => (
                    <div key={m.field} style={{ display: 'contents' }} data-testid={`compare-metadata-${m.field}`}>
                      <span>{m.field}</span><span>{String(m.baselineValue)}</span><span>-&gt; {String(m.candidateValue)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {comparison.opCounts.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <SectionHeader>Op type count changes ({comparison.opCounts.length})</SectionHeader>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, auto)', columnGap: 16, rowGap: 2, fontSize: 13, color: 'var(--text-secondary)' }}>
                  {comparison.opCounts.map((c) => (
                    <div key={c.opType} style={{ display: 'contents' }} data-testid={`compare-opcount-${c.opType}`}>
                      <span>{c.opType}</span><span>{c.baselineCount}</span><span>-&gt; {c.candidateCount}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {comparison.attributes.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <SectionHeader>Attribute changes ({comparison.attributes.length})</SectionHeader>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2, fontSize: 13, color: 'var(--text-secondary)' }}>
                  {comparison.attributes.map((a, i) => (
                    <span key={i} data-testid="compare-attribute-change">
                      {a.opType} {a.attrName}: {String(a.baselineValue ?? '(none)')} -&gt; {String(a.candidateValue ?? '(none)')}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {comparison.initializers.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <SectionHeader>Initializer changes ({comparison.initializers.length})</SectionHeader>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2, fontSize: 13, color: 'var(--text-secondary)' }}>
                  {comparison.initializers.map((init) => (
                    <span key={init.name} data-testid="compare-initializer-change">
                      {init.name}: {init.status} ({formatShapeCell(init.baselineShape)} -&gt; {formatShapeCell(init.candidateShape)})
                    </span>
                  ))}
                </div>
              </div>
            )}

            {comparison.io.some((c) => c.status !== 'unchanged') && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <SectionHeader>Graph I/O changes</SectionHeader>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2, fontSize: 13, color: 'var(--text-secondary)' }}>
                  {comparison.io.filter((c) => c.status !== 'unchanged').map((io) => (
                    <span key={`${io.ioKind}-${io.name}`} data-testid="compare-io-change">
                      [{io.ioKind}] {io.name}: {io.status} ({formatShapeCell(io.baselineShape)} -&gt; {formatShapeCell(io.candidateShape)})
                    </span>
                  ))}
                </div>
              </div>
            )}

            {comparison.opCounts.length === 0 && comparison.attributes.length === 0 && comparison.initializers.length === 0
              && comparison.nodeMatch.addedNodes.length === 0 && comparison.nodeMatch.removedNodes.length === 0
              && !comparison.io.some((c) => c.status !== 'unchanged') && comparison.metadata.length === 0 && (
              <span style={{ color: 'var(--color-green)', fontSize: 13 }}>No structural differences detected.</span>
            )}

            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 16 }}>
              <button type="button" data-testid="compare-run-latency" onClick={handleRunLatency} disabled={isBenchmarking || isRunningOutputs} className="btn-bar">
                {isBenchmarking ? 'Running...' : 'Run Latency Comparison'}
              </button>
              <button type="button" data-testid="compare-run-outputs" onClick={handleRunOutputs} disabled={isBenchmarking || isRunningOutputs} className="btn-bar">
                {isRunningOutputs ? 'Running...' : 'Run Output Comparison'}
              </button>
              <button type="button" data-testid="compare-export-report" onClick={handleExportReport} className="btn-bar btn-primary">
                Export Report
              </button>
              {editRecipeEntries && (
                <button type="button" data-testid="compare-export-recipe" onClick={() => { void handleExportEditRecipe() }} className="btn-bar btn-primary">
                  Export Edit Recipe
                </button>
              )}
            </div>

            {recipeMessage && <span style={{ color: 'var(--color-amber)', fontSize: 13 }}>{recipeMessage}</span>}

            {latencyResult && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <SectionHeader>Latency (avg / median ms, ONNX Runtime Web, this machine)</SectionHeader>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, auto)', columnGap: 16, rowGap: 2, fontSize: 13, color: 'var(--text-secondary)' }}>
                  <span>Baseline</span>
                  <span data-testid="compare-latency-baseline" style={'error' in latencyResult.baseline ? { color: 'var(--color-error)' } : undefined}>
                    {'error' in latencyResult.baseline ? latencyResult.baseline.error : `${latencyResult.baseline.avgMs.toFixed(2)} / ${latencyResult.baseline.medianMs.toFixed(2)}`}
                  </span>
                  <span>Candidate</span>
                  <span data-testid="compare-latency-candidate" style={'error' in latencyResult.candidate ? { color: 'var(--color-error)' } : undefined}>
                    {'error' in latencyResult.candidate ? latencyResult.candidate.error : `${latencyResult.candidate.avgMs.toFixed(2)} / ${latencyResult.candidate.medianMs.toFixed(2)}`}
                  </span>
                </div>
              </div>
            )}

            {outputResult && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <SectionHeader>Output comparison (generated inputs, ONNX Runtime Web)</SectionHeader>
                {(!outputResult.baseline.inferenceOk || !outputResult.candidate.inferenceOk) ? (
                  <span style={{ color: 'var(--color-error)', fontSize: 13 }}>
                    {outputResult.baseline.error ?? outputResult.candidate.error ?? 'Inference failed on one or both models'}
                  </span>
                ) : (
                  <div data-testid="compare-output-comparisons" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {outputComparisons?.map((c) => (
                      <div key={c.name} data-testid={`compare-output-${c.name}`} style={{ border: '1px solid rgba(255,255,255,0.08)', borderRadius: 2, padding: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
                        <span style={{ color: 'var(--text-primary)', fontSize: 13 }}>{c.name}</span>
                        {!c.presentInOriginal || !c.presentInModified ? (
                          <span style={{ color: 'var(--color-error)', fontSize: 12 }}>Only present in {c.presentInOriginal ? 'baseline' : 'candidate'}</span>
                        ) : !c.shapeMatch ? (
                          <span style={{ color: 'var(--color-error)', fontSize: 12 }}>Shape mismatch: {c.originalShape?.join('x')} vs {c.modifiedShape?.join('x')}</span>
                        ) : (
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, auto)', columnGap: 16, rowGap: 2, fontSize: 12, color: 'var(--text-secondary)' }}>
                            <span>Max abs error</span><span>{formatMetric(c.maxAbsErr)}</span>
                            <span>Cosine similarity</span><span>{formatMetric(c.cosineSim)}</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
