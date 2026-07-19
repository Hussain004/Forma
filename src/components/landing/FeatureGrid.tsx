import { motion } from 'framer-motion'

const stagger = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.1, delayChildren: 0.55 },
  },
}

const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.45, ease: [0.2, 0, 0, 1] as const },
  },
}

function ParseIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
      <rect x="4" y="4" width="20" height="20" rx="2" stroke="var(--color-amber)" strokeWidth="1" opacity="0.4" />
      <path d="M8 10h12M8 14h8M8 18h10" stroke="var(--color-amber)" strokeWidth="1" opacity="0.6" />
      <circle cx="21" cy="7" r="3" fill="var(--bg-base)" stroke="var(--color-amber)" strokeWidth="1" opacity="0.7" />
      <path d="M20 7l0.7 0.7L22.5 5.9" stroke="var(--color-amber)" strokeWidth="0.8" opacity="0.7" fill="none" />
    </svg>
  )
}

function EditIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
      <circle cx="8" cy="10" r="3" stroke="var(--color-amber)" strokeWidth="1" opacity="0.5" />
      <circle cx="20" cy="10" r="3" stroke="var(--color-amber)" strokeWidth="1" opacity="0.5" />
      <circle cx="14" cy="20" r="3" stroke="var(--color-amber)" strokeWidth="1" opacity="0.5" />
      <line x1="10.5" y1="11.5" x2="11.5" y2="18" stroke="var(--color-amber)" strokeWidth="1" opacity="0.4" />
      <line x1="17.5" y1="11.5" x2="15.5" y2="18" stroke="var(--color-amber)" strokeWidth="1" opacity="0.4" />
      <line x1="11" y1="10" x2="17" y2="10" stroke="var(--color-amber)" strokeWidth="1" opacity="0.4" />
    </svg>
  )
}

function ExportIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
      <path d="M14 4v14" stroke="var(--color-amber)" strokeWidth="1" opacity="0.6" />
      <path d="M10 12l4 4 4-4" stroke="var(--color-amber)" strokeWidth="1" opacity="0.6" fill="none" />
      <path d="M6 18v4a2 2 0 002 2h12a2 2 0 002-2v-4" stroke="var(--color-amber)" strokeWidth="1" opacity="0.4" fill="none" />
    </svg>
  )
}

interface Feature {
  icon: React.ReactNode
  name: string
  desc: string
}

const FEATURES: Feature[] = [
  {
    icon: <ParseIcon />,
    name: 'Parse in-browser',
    desc: 'ONNX and TFLite via WASM. Your model never leaves this device.',
  },
  {
    icon: <EditIcon />,
    name: 'Edit and rewire',
    desc: 'Insert, delete, reconnect nodes. Attribute editing with undo.',
  },
  {
    icon: <ExportIcon />,
    name: 'Export modified',
    desc: 'Byte-preserving ONNX writer. Verified by onnxruntime load.',
  },
]

export function FeatureGrid() {
  return (
    <motion.div
      className="landing-features"
      variants={stagger}
      initial="hidden"
      animate="visible"
    >
      {FEATURES.map((f) => (
        <motion.div key={f.name} className="landing-feature" variants={fadeUp}>
          <span className="landing-feature-icon">
            {f.icon}
          </span>
          <span className="landing-feature-name">{f.name}</span>
          <span className="landing-feature-desc">{f.desc}</span>
        </motion.div>
      ))}
    </motion.div>
  )
}
