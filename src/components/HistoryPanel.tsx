import { useEffect, useRef } from 'react'
import { describeHistoryEntry, type HistoryEntry } from '../lib/graphUtils'

interface HistoryPanelProps {
  entries: HistoryEntry[]
  index: number
  onJump: (index: number) => void
}

export function HistoryPanel({ entries, index, onJump }: HistoryPanelProps) {
  const currentRowRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    currentRowRef.current?.scrollIntoView?.({ block: 'nearest' })
  }, [index])

  const rows = [
    { index: 0, label: 'Original model' },
    ...entries.map((entry, entryIndex) => ({
      index: entryIndex + 1,
      label: describeHistoryEntry(entry),
    })),
  ]

  return (
    <section
      data-testid="history-panel"
      aria-label="Edit history"
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        background: 'var(--bg-surface)',
        fontFamily: 'var(--font-mono)',
      }}
    >
      <div style={{ padding: '12px 16px 8px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
        <span style={{ color: 'var(--text-dim)', fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          {entries.length} {entries.length === 1 ? 'edit' : 'edits'}
        </span>
      </div>
      <div role="list" style={{ overflowY: 'auto', padding: '8px 0' }}>
        {rows.map((row) => {
          const isCurrent = row.index === index
          const isFuture = row.index > index
          return (
            <button
              key={row.index}
              ref={isCurrent ? currentRowRef : null}
              type="button"
              role="listitem"
              data-testid={`history-row-${row.index}`}
              data-history-state={isCurrent ? 'current' : isFuture ? 'future' : 'applied'}
              aria-current={isCurrent ? 'step' : undefined}
              onClick={() => onJump(row.index)}
              className="btn-ghost"
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'flex-start',
                gap: 8,
                padding: '8px 16px',
                textAlign: 'left',
                borderLeft: isCurrent ? '2px solid var(--color-amber)' : '2px solid transparent',
                background: isCurrent ? 'rgba(255,176,0,0.08)' : 'transparent',
                color: isCurrent ? 'var(--color-amber)' : isFuture ? 'var(--text-dim)' : 'var(--text-secondary)',
                opacity: isFuture ? 0.55 : 1,
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                letterSpacing: '0.02em',
                lineHeight: 1.5,
              }}
            >
              <span style={{ flexShrink: 0, color: isCurrent ? 'var(--color-amber)' : 'var(--text-dim)', fontSize: 10 }}>
                {String(row.index).padStart(2, '0')}
              </span>
              <span>{row.label}</span>
            </button>
          )
        })}
      </div>
    </section>
  )
}
