import { useMemo, useRef } from 'react'
import { compareOutputs, type ValidationRunResult, type SideRunResult } from '../lib/validationUtils'

interface ValidationPanelProps {
  isReadOnly: boolean
  isRunning: boolean
  currentResult: ValidationRunResult | null
  providedInputNames: string[] | null
  fileError: string | null
  onFileSelected: (file: File) => void
  onClearProvidedInputs: () => void
  onRun: () => void
  validatedStates: { index: number; ok: boolean }[]
  activeIndex: number
  onJumpToState: (index: number) => void
}

function formatMetric(value: number | undefined): string {
  if (value === undefined) return '--'
  if (value === 0) return '0'
  return Math.abs(value) < 0.001 ? value.toExponential(2) : value.toFixed(4)
}

function SideStatus({ label, side }: { label: string; side: SideRunResult }) {
  const ok = side.loaded && side.inferenceOk
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <span style={{ color: 'var(--text-dim)', fontSize: 11, letterSpacing: '0.1em', textTransform: 'uppercase' }}>{label}</span>
      <span style={{ color: ok ? 'var(--color-green)' : 'var(--color-error)', fontSize: 13 }}>
        {side.loaded ? 'Loaded' : 'Load failed'} / {side.inferenceOk ? 'Inference OK' : 'Inference failed'}
      </span>
      {side.error && <span style={{ color: 'var(--color-error)', fontSize: 12, overflowWrap: 'anywhere' }}>{side.error}</span>}
    </div>
  )
}

export function ValidationPanel({
  isReadOnly,
  isRunning,
  currentResult,
  providedInputNames,
  fileError,
  onFileSelected,
  onClearProvidedInputs,
  onRun,
  validatedStates,
  activeIndex,
  onJumpToState,
}: ValidationPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const comparisons = useMemo(() => {
    if (!currentResult || !currentResult.original.inferenceOk || !currentResult.modified.inferenceOk) return null
    return compareOutputs(currentResult.original.outputs, currentResult.modified.outputs)
  }, [currentResult])

  if (isReadOnly) {
    return (
      <section aria-label="Behavioral validation" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: 13, textAlign: 'center' }}>
        Behavioral validation is only available for editable ONNX models.
      </section>
    )
  }

  return (
    <section
      data-testid="validation-panel"
      aria-label="Behavioral validation"
      style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--bg-surface)', fontFamily: 'var(--font-mono)' }}
    >
      <div style={{ padding: '12px 16px 8px', borderBottom: '1px solid rgba(255,255,255,0.08)', display: 'flex', flexDirection: 'column', gap: 8 }}>
        <span style={{ color: 'var(--text-dim)', fontSize: 12, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          Runs the original and the current edits against identical inputs in ONNX Runtime Web, in this browser
        </span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <button type="button" data-testid="validate-run" onClick={onRun} disabled={isRunning} className="btn-bar btn-primary">
            {isRunning ? 'Running...' : 'Run validation'}
          </button>
          <button type="button" data-testid="validate-upload" onClick={() => fileInputRef.current?.click()} className="btn-bar btn-ghost">
            Load .npy / .npz inputs
          </button>
          <input
            ref={fileInputRef}
            data-testid="validate-file-input"
            type="file"
            accept=".npy,.npz"
            style={{ display: 'none' }}
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) onFileSelected(file)
              e.target.value = ''
            }}
          />
          {providedInputNames && providedInputNames.length > 0 && (
            <button type="button" data-testid="validate-clear-inputs" onClick={onClearProvidedInputs} className="btn-ghost" style={{ fontSize: 12, padding: '2px 8px' }}>
              Clear ({providedInputNames.join(', ')})
            </button>
          )}
        </div>
        {fileError && <span style={{ color: 'var(--color-error)', fontSize: 12 }}>{fileError}</span>}
        <span style={{ color: 'var(--text-dim)', fontSize: 12 }}>
          {providedInputNames && providedInputNames.length > 0
            ? `Using loaded inputs: ${providedInputNames.join(', ')}`
            : 'No file loaded: uses deterministic generated inputs (a smoke test, not real data).'}
        </span>
      </div>

      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 16 }}>
        {!currentResult ? (
          <span style={{ color: 'var(--text-dim)', fontSize: 13 }}>Not yet run for this edit state.</span>
        ) : (
          <>
            <div style={{ display: 'flex', gap: 24 }}>
              <SideStatus label="Original" side={currentResult.original} />
              <SideStatus label="Modified" side={currentResult.modified} />
            </div>
            <span style={{ color: 'var(--text-dim)', fontSize: 12 }}>
              Inputs: {currentResult.inputSource === 'provided' ? 'loaded file' : 'generated (smoke test)'}
            </span>
            {comparisons && (
              <div data-testid="validation-comparisons" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {comparisons.map((c) => (
                  <div key={c.name} data-testid={`validation-output-${c.name}`} style={{ border: '1px solid rgba(255,255,255,0.08)', borderRadius: 2, padding: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <span style={{ color: 'var(--text-primary)', fontSize: 13 }}>{c.name}</span>
                    {!c.presentInOriginal || !c.presentInModified ? (
                      <span style={{ color: 'var(--color-error)', fontSize: 12 }}>
                        Only present in {c.presentInOriginal ? 'original' : 'modified'}
                      </span>
                    ) : !c.shapeMatch ? (
                      <span style={{ color: 'var(--color-error)', fontSize: 12 }}>
                        Shape mismatch: {c.originalShape?.join('x')} vs {c.modifiedShape?.join('x')}
                      </span>
                    ) : (
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, auto)', columnGap: 16, rowGap: 2, fontSize: 12, color: 'var(--text-secondary)' }}>
                        <span>Max abs error</span><span>{formatMetric(c.maxAbsErr)}</span>
                        <span>Max rel error</span><span>{formatMetric(c.maxRelErr)}</span>
                        <span>Cosine similarity</span><span>{formatMetric(c.cosineSim)}</span>
                        <span>Top-{c.topK?.k ?? '-'} agreement</span><span>{c.topK ? `${c.topK.overlap}/${c.topK.k}` : 'n/a'}</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>

      {validatedStates.length > 0 && (
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', padding: '8px 16px', display: 'flex', flexDirection: 'column', gap: 6 }}>
          <span style={{ color: 'var(--text-dim)', fontSize: 11, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Validated edit states</span>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {validatedStates.map((entry) => (
              <button
                key={entry.index}
                type="button"
                data-testid={`validation-state-${entry.index}`}
                onClick={() => onJumpToState(entry.index)}
                className="btn-ghost"
                style={{
                  fontSize: 12,
                  padding: '2px 8px',
                  border: entry.index === activeIndex ? '1px solid var(--color-amber)' : '1px solid transparent',
                  color: entry.ok ? 'var(--color-green)' : 'var(--color-error)',
                }}
              >
                {String(entry.index).padStart(2, '0')}
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}
