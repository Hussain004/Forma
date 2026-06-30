import { useRef, useState } from 'react'

interface ModelDropzoneProps {
  onModelLoaded: (buffer: ArrayBuffer, filename: string) => void
  status: 'idle' | 'loading' | 'ready' | 'error'
  error?: string | null
  progressLabel?: string | null
  progressPercent?: number | null
}

function Crosshair() {
  return (
    <svg width="40" height="40" viewBox="0 0 40 40" aria-hidden="true">
      <g stroke="#8A8F9E" strokeWidth="1">
        <line x1="20" y1="0" x2="20" y2="14" />
        <line x1="20" y1="26" x2="20" y2="40" />
        <line x1="0" y1="20" x2="14" y2="20" />
        <line x1="26" y1="20" x2="40" y2="20" />
      </g>
    </svg>
  )
}

export function ModelDropzone({ onModelLoaded, status, error, progressLabel, progressPercent }: ModelDropzoneProps) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const readFile = (file: File) => {
    const reader = new FileReader()
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) {
        onModelLoaded(reader.result, file.name)
      }
    }
    reader.readAsArrayBuffer(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) readFile(file)
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) readFile(file)
  }

  const isBusy = status === 'loading'

  return (
    <div
      onClick={() => !isBusy && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        if (!dragging) setDragging(true)
      }}
      onDragLeave={(e) => {
        e.preventDefault()
        setDragging(false)
      }}
      onDrop={handleDrop}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 10,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 16,
        cursor: isBusy ? 'default' : 'pointer',
        background: dragging ? 'rgba(255, 176, 0, 0.05)' : 'var(--bg-base)',
        border: dragging ? '1px solid #FFB000' : '1px solid transparent',
        transition: 'background 0.12s ease',
        userSelect: 'none',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".onnx"
        onChange={handleFileInput}
        style={{ display: 'none' }}
      />

      {status === 'loading' && (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
          <span
            style={{
              color: 'var(--text-secondary)',
              textTransform: 'uppercase',
              letterSpacing: '0.16em',
              fontSize: 13,
            }}
          >
            {progressLabel ?? 'Parsing model...'}
          </span>
          <div
            style={{
              width: 240,
              height: 2,
              background: '#1C2128',
              borderRadius: 2,
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                height: '100%',
                width: `${progressPercent ?? 0}%`,
                background: '#FFB000',
                borderRadius: 2,
                transition: 'width 0.3s ease',
              }}
            />
          </div>
        </div>
      )}

      {status === 'error' && (
        <>
          <Crosshair />
          <span
            style={{
              color: 'var(--color-error)',
              textTransform: 'uppercase',
              letterSpacing: '0.12em',
              fontSize: 13,
              textAlign: 'center',
              maxWidth: 360,
            }}
          >
            {error ?? 'Failed to parse model'}
          </span>
          <span
            style={{
              color: 'var(--text-dim)',
              textTransform: 'uppercase',
              letterSpacing: '0.12em',
              fontSize: 11,
            }}
          >
            Drop another .onnx model
          </span>
        </>
      )}

      {(status === 'idle' || status === 'ready') && (
        <>
          <Crosshair />
          <span
            style={{
              color: 'var(--text-secondary)',
              textTransform: 'uppercase',
              letterSpacing: '0.16em',
              fontSize: 13,
            }}
          >
            Drop .onnx model
          </span>
        </>
      )}
    </div>
  )
}
