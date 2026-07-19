import { motion } from 'framer-motion'

const stagger = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.12, delayChildren: 0.1 },
  },
}

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: [0.2, 0, 0, 1] as const },
  },
}

function CircuitIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
      <rect x="10" y="10" width="8" height="8" rx="1" stroke="var(--color-amber)" strokeWidth="1" opacity="0.5" />
      <line x1="14" y1="0" x2="14" y2="10" stroke="var(--color-amber)" strokeWidth="1" opacity="0.35" />
      <line x1="14" y1="18" x2="14" y2="28" stroke="var(--color-amber)" strokeWidth="1" opacity="0.35" />
      <line x1="0" y1="14" x2="10" y2="14" stroke="var(--color-amber)" strokeWidth="1" opacity="0.35" />
      <line x1="18" y1="14" x2="28" y2="14" stroke="var(--color-amber)" strokeWidth="1" opacity="0.35" />
      <circle cx="14" cy="14" r="2" fill="var(--color-amber)" opacity="0.6" />
    </svg>
  )
}

export function HeroSection() {
  return (
    <motion.div
      className="landing-hero"
      variants={stagger}
      initial="hidden"
      animate="visible"
    >
      <motion.div className="landing-title" variants={fadeUp}>
        <CircuitIcon />
        <span className="landing-title-text">
          FOR<span className="landing-title-accent">M</span>A
        </span>
        <CircuitIcon />
      </motion.div>

      <motion.div className="landing-tagline" variants={fadeUp}>
        Browser-native neural network editor.
        <span className="landing-tagline-br" />
        Inspect, edit, and export ONNX models. Zero server. Zero install.
      </motion.div>

      {/* Decorative technical readout */}
      <motion.div
        className="landing-readout"
        variants={fadeUp}
      >
        <span className="landing-readout-dot" />
        <span>wasm runtime loaded</span>
        <span className="landing-readout-sep">|</span>
        <span>onnx + tflite</span>
        <span className="landing-readout-sep">|</span>
        <span>no server required</span>
      </motion.div>
    </motion.div>
  )
}
