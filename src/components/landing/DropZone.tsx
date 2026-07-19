import { useRef, useState } from 'react'
import { motion } from 'framer-motion'

interface DropZoneProps {
  onModelLoaded: (buffer: ArrayBuffer, filename: string) => void
  status: 'idle' | 'loading' | 'ready' | 'error'
  error?: string | null
  progressLabel?: string | null
  progressPercent?: number | null
}

function friendlyErrorHeadline(raw: string | null | undefined): string {
  const lower = (raw ?? '').toLowerCase()
  if (lower.includes('protobuf parsing failed') || lower.includes("can't create a session") || lower.includes('failed to load model')) {
    return "This file doesn't look like a valid ONNX or TFLite model."
  }
  return 'Something went wrong while loading this model.'
}

function CrosshairSVG() {
  return (
    <svg width="36" height="36" viewBox="0 0 36 36" aria-hidden="true" className="landing-dropzone-crosshair">
      <g stroke="var(--color-amber)" strokeWidth="1" opacity="0.7">
        <line x1="18" y1="2" x2="18" y2="12" />
        <line x1="18" y1="24" x2="18" y2="34" />
        <line x1="2" y1="18" x2="12" y2="18" />
        <line x1="24" y1="18" x2="34" y2="18" />
      </g>
      <circle cx="18" cy="18" r="3" fill="none" stroke="var(--color-amber)" strokeWidth="1" opacity="0.4" />
    </svg>
  )
}

export function DropZone({ onModelLoaded, status, error, progressLabel, progressPercent }: DropZoneProps) {
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
    e.stopPropagation()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) readFile(file)
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) readFile(file)
  }

  const handleLoadSample = (e: React.MouseEvent) => {
    e.stopPropagation()
    fetch('/sample-model.onnx')
      .then((r) => r.arrayBuffer())
      .then((buf) => onModelLoaded(buf, 'sample-model.onnx'))
  }

  const isBusy = status === 'loading'

  return (
    <motion.div
      className="landing-dropzone-wrapper"
      initial={{ opacity: 0, y: 20, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, delay: 0.35, ease: [0.2, 0, 0, 1] }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".onnx,.tflite"
        onChange={handleFileInput}
        style={{ display: 'none' }}
      />

      {status === 'loading' && (
        <motion.div
          className="landing-dropzone"
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.25 }}
        >
          <div className="landing-loading">
            <span
              style={{
                color: 'var(--text-secondary)',
                textTransform: 'uppercase',
                letterSpacing: '0.14em',
                fontSize: 12,
              }}
            >
              {progressLabel ?? 'Parsing model...'}
            </span>
            <div className="landing-loading-bar">
              <div
                className="landing-dropzone-fill loading-fill-shimmer"
                style={{
                  width: `${progressPercent ?? 0}%`,
                }}
              />
            </div>
          </div>
        </motion.div>
      )}

      {status === 'error' && (
        <motion.div
          className="landing-dropzone"
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.25 }}
        >
          <CrosshairSVG />
          <span className="landing-error">
            {friendlyErrorHeadline(error)}
          </span>
          {error && (
            <span className="landing-error-detail">{error}</span>
          )}
          <span className="landing-dropzone-sublabel" style={{ marginTop: 8 }}>
            Drop another .onnx or .tflite model
          </span>
        </motion.div>
      )}

      {status !== 'loading' && status !== 'error' && (
        <motion.div
          className={`landing-dropzone${dragging ? ' dragging' : ''}`}
          onClick={() => !isBusy && inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            e.stopPropagation()
            if (!dragging) setDragging(true)
          }}
          onDragLeave={(e) => {
            e.preventDefault()
            e.stopPropagation()
            setDragging(false)
          }}
          onDrop={handleDrop}
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
          whileHover={{ borderColor: 'rgba(255, 176, 0, 0.35)' }}
        >
          <CrosshairSVG />

          <span className="landing-dropzone-label">
            Drop .onnx or .tflite
          </span>

          <span className="landing-dropzone-sublabel">
            or click anywhere to browse
          </span>

          <div className="landing-dropzone-actions">
            <button onClick={handleLoadSample} className="btn-bar btn-ghost">
              Load sample model
            </button>
            <a
              href="https://github.com/Hussain004/Forma"
              target="_blank"
              rel="noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="btn-link btn-bar btn-ghost"
            >
              GitHub
            </a>
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}
