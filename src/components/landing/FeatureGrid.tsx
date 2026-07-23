import { motion } from 'framer-motion'
import { NerdGlyph } from './NerdGlyph'

const PIPELINE = ['PARSE', 'INSPECT', 'EDIT', 'EXPORT']

export function FeatureGrid() {
  return (
    <motion.div
      className="landing-pipeline"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.55 }}
      aria-label="Forma model workflow"
    >
      <span className="landing-pipeline-label">GRAPH PIPELINE</span>
      {PIPELINE.map((stage, index) => (
        <span key={stage} className="landing-pipeline-stage">
          <span className="landing-pipeline-index">{String(index + 1).padStart(2, '0')}</span>
          {stage}
          {index < PIPELINE.length - 1 && <NerdGlyph glyph="arrowRight" />}
        </span>
      ))}
    </motion.div>
  )
}
