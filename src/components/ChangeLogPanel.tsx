import { describeHistoryEntry, type HistoryEntry } from '../lib/graphUtils'

interface ChangeLogPanelProps {
  entries: HistoryEntry[]
  onCopy?: () => void
}

function formatChangeLog(entries: HistoryEntry[]): string {
  if (entries.length === 0) return 'No active changes.'
  return entries
    .map((entry, index) => `${String(index + 1).padStart(2, '0')}  ${describeHistoryEntry(entry)}`)
    .join('\n')
}

export function ChangeLogPanel({ entries, onCopy }: ChangeLogPanelProps) {
  const text = formatChangeLog(entries)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text)
    onCopy?.()
  }

  return (
    <section
      data-testid="change-log-panel"
      aria-label="Active change log"
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        background: 'var(--bg-surface)',
        fontFamily: 'var(--font-mono)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 16px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
        <span style={{ color: 'var(--text-dim)', fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          {entries.length} active {entries.length === 1 ? 'change' : 'changes'}
        </span>
        <button
          type="button"
          onClick={() => { void handleCopy() }}
          disabled={entries.length === 0}
          className="btn-ghost"
          style={{ fontSize: 10, padding: '2px 8px' }}
        >
          Copy
        </button>
      </div>
      <pre
        data-testid="change-log-text"
        style={{
          flex: 1,
          minHeight: 0,
          overflow: 'auto',
          margin: 0,
          padding: 16,
          color: entries.length > 0 ? 'var(--text-secondary)' : 'var(--text-dim)',
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          lineHeight: 1.7,
          whiteSpace: 'pre-wrap',
          overflowWrap: 'anywhere',
        }}
      >
        {text}
      </pre>
    </section>
  )
}
