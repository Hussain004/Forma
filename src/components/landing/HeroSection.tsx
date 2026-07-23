import { motion } from 'framer-motion'
import { NerdGlyph } from './NerdGlyph'

const stagger = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.08, delayChildren: 0.08 },
  },
}

const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.42, ease: [0.2, 0, 0, 1] as const },
  },
}

export function HeroSection() {
  return (
    <motion.section
      className="landing-hero"
      variants={stagger}
      initial="hidden"
      animate="visible"
    >
      <motion.div className="landing-kicker" variants={fadeUp}>
        <NerdGlyph glyph="terminal" />
        <span>BROWSER NATIVE MODEL WORKBENCH</span>
        <span className="landing-kicker-code">SYS.01</span>
      </motion.div>

      <motion.h1 className="landing-title" variants={fadeUp}>
        FOR<span className="landing-title-accent">M</span>A
      </motion.h1>

      <motion.p className="landing-tagline" variants={fadeUp}>
        Shape the graph. Keep the model local.
      </motion.p>

      <motion.p className="landing-summary" variants={fadeUp}>
        Parse, inspect, edit, and export ONNX models entirely in your browser.
      </motion.p>

      <motion.div className="landing-readout" variants={fadeUp}>
        <span><NerdGlyph glyph="lock" /> Local execution</span>
        <span><NerdGlyph glyph="bolt" /> WASM pipeline</span>
        <span><NerdGlyph glyph="cube" /> ONNX + TFLite</span>
      </motion.div>
    </motion.section>
  )
}
