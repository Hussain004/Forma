import { motion } from 'framer-motion'

export function StatusBar() {
  return (
    <motion.div
      className="landing-statusbar"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6, delay: 0.8 }}
    >
      <div className="landing-statusbar-left">
        <span className="landing-statusbar-badge landing-statusbar-badge-amber">
          v1.6.0
        </span>
        <span className="landing-statusbar-badge">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
            <circle cx="6" cy="6" r="2.5" stroke="var(--color-green)" strokeWidth="1" />
            <path d="M6 1v2M6 9v2M1 6h2M9 6h2" stroke="var(--color-green)" strokeWidth="0.8" opacity="0.6" />
          </svg>
          0 runtime deps
        </span>
        <span className="landing-statusbar-badge">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
            <rect x="2" y="2" width="8" height="8" rx="1" stroke="var(--text-dim)" strokeWidth="0.8" />
            <path d="M4 6h4" stroke="var(--text-dim)" strokeWidth="0.8" />
          </svg>
          client-side only
        </span>
      </div>

      <div className="landing-statusbar-right">
        <a
          href="https://github.com/Hussain004/Forma"
          target="_blank"
          rel="noreferrer"
          className="landing-statusbar-link"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <path d="M7 0.5C3.4 0.5 0.5 3.4 0.5 7c0 2.9 1.9 5.3 4.5 6.2 0.3 0.1 0.4-0.1 0.4-0.3v-1.1c-1.8 0.4-2.2-0.9-2.2-0.9-0.3-0.8-0.7-1-0.7-1-0.6-0.4 0-0.4 0-0.4 0.7 0 1 0.7 1 0.7 0.6 1 1.5 0.7 1.9 0.5 0.1-0.4 0.2-0.7 0.4-0.9-1.5-0.2-3-0.7-3-3.3 0-0.7 0.3-1.3 0.7-1.8-0.1-0.2-0.3-0.8 0.1-1.7 0 0 0.6-0.2 1.8 0.7 0.5-0.1 1.1-0.2 1.7-0.2s1.2 0.1 1.7 0.2c1.2-0.9 1.8-0.7 1.8-0.7 0.4 0.9 0.2 1.5 0.1 1.7 0.4 0.5 0.7 1.1 0.7 1.8 0 2.6-1.5 3.1-3 3.3 0.2 0.2 0.4 0.6 0.4 1.1v1.7c0 0.2 0.1 0.4 0.4 0.3C11.6 12.3 13.5 9.9 13.5 7 13.5 3.4 10.6 0.5 7 0.5z" fill="currentColor" opacity="0.7" />
          </svg>
          source
        </a>
        <a
          href="https://github.com/Hussain004/Forma/blob/v1.6/README.md"
          target="_blank"
          rel="noreferrer"
          className="landing-statusbar-link"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <path d="M1 3l6-2.5L13 3v5.5c0 2-3 3.5-6 4.5-3-1-6-2.5-6-4.5V3z" stroke="currentColor" strokeWidth="0.8" opacity="0.7" fill="none" />
            <path d="M5 7l2 2 3-4" stroke="currentColor" strokeWidth="0.8" opacity="0.7" fill="none" />
          </svg>
          docs
        </a>
      </div>
    </motion.div>
  )
}
