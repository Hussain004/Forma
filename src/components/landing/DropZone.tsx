import { useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { NerdGlyph } from './NerdGlyph'

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

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault()
    event.stopPropagation()
    setDragging(false)
    const file = event.dataTransfer.files?.[0]
    if (file) readFile(file)
  }

  const handleFileInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) readFile(file)
    event.target.value = ''
  }

  const handleLoadSample = (event: React.MouseEvent) => {
    event.stopPropagation()
    fetch('/sample-model.onnx')
      .then((response) => {
        if (typeof response.ok === 'boolean' && !response.ok) {
          throw new Error(`Sample model request failed: ${response.status}`)
        }
        return response.arrayBuffer()
      })
      .then((buffer) => onModelLoaded(buffer, 'sample-model.onnx'))
  }

  const isBusy = status === 'loading'
  const openPicker = () => {
    if (!isBusy) inputRef.current?.click()
  }

  return (
    <motion.div
      className="landing-dropzone-wrapper"
      initial={{ opacity: 0, y: 16, scale: 0.985 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.45, delay: 0.3, ease: [0.2, 0, 0, 1] }}
    >
      <input
        ref={inputRef}
        className="landing-file-input"
        type="file"
        accept=".onnx,.tflite"
        onChange={handleFileInput}
      />

      <AnimatePresence>
        {status === 'loading' && (
          <motion.section
            key="loading"
            className="landing-dropzone landing-dropzone-state"
            aria-live="polite"
            initial={{ opacity: 0, scale: 0.985 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.985 }}
            transition={{ duration: 0.2 }}
          >
            <div className="landing-dropzone-meta">
              <span>DECODING GRAPH</span>
              <span>{Math.round(progressPercent ?? 0).toString().padStart(3, '0')}%</span>
            </div>
            <div className="landing-loading">
              <NerdGlyph glyph="fileCode" className="landing-dropzone-icon" />
              <span className="landing-dropzone-label">{progressLabel ?? 'Parsing model'}</span>
              <div className="landing-loading-bar" aria-hidden="true">
                <motion.div
                  className="landing-loading-fill"
                  animate={{ width: `${progressPercent ?? 0}%` }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                />
              </div>
            </div>
          </motion.section>
        )}

        {status === 'error' && (
          <motion.section
            key="error"
            className="landing-dropzone landing-dropzone-state landing-dropzone-error"
            aria-live="assertive"
            onClick={openPicker}
            initial={{ opacity: 0, scale: 0.985 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.985 }}
            transition={{ duration: 0.2 }}
          >
            <div className="landing-dropzone-meta">
              <span>INGRESS REJECTED</span>
              <span>ERR.01</span>
            </div>
            <NerdGlyph glyph="fileCode" className="landing-dropzone-icon" />
            <span className="landing-error">{friendlyErrorHeadline(error)}</span>
            {error && <span className="landing-error-detail">{error}</span>}
            <button
              type="button"
              className="landing-text-action"
              onClick={(event) => {
                event.stopPropagation()
                openPicker()
              }}
            >
              Select another model <NerdGlyph glyph="arrowRight" />
            </button>
          </motion.section>
        )}

        {status !== 'loading' && status !== 'error' && (
          <motion.section
            key="idle"
            className={`landing-dropzone${dragging ? ' dragging' : ''}`}
            role="button"
            tabIndex={0}
            aria-label="Upload an ONNX or TFLite model"
            onClick={openPicker}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                openPicker()
              }
            }}
            onDragOver={(event) => {
              event.preventDefault()
              event.stopPropagation()
              if (!dragging) setDragging(true)
            }}
            onDragLeave={(event) => {
              event.preventDefault()
              event.stopPropagation()
              if (event.currentTarget.contains(event.relatedTarget as Node | null)) return
              setDragging(false)
            }}
            onDrop={handleDrop}
            initial={{ opacity: 0, scale: 0.985 }}
            animate={{ opacity: 1, scale: dragging ? 1.006 : 1 }}
            exit={{ opacity: 0, scale: 0.985 }}
            transition={{ duration: 0.2 }}
          >
            <div className="landing-dropzone-meta">
              <span>MODEL INGRESS / LOCAL FILE</span>
              <span>.ONNX .TFLITE</span>
            </div>

            <div className="landing-dropzone-target">
              <NerdGlyph glyph="upload" className="landing-dropzone-icon" />
              <span className="landing-dropzone-label">Drop .onnx or .tflite model</span>
              <span className="landing-dropzone-sublabel">or click to select from this device</span>
            </div>

            <div className="landing-dropzone-actions">
              <span><NerdGlyph glyph="lock" /> Bytes stay in this browser</span>
              <button onClick={handleLoadSample} className="landing-text-action">
                Load sample model <NerdGlyph glyph="arrowRight" />
              </button>
            </div>
          </motion.section>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
