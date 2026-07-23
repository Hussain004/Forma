import { motion } from 'framer-motion'
import { NerdGlyph } from './NerdGlyph'

export function StatusBar() {
  return (
    <>
      <motion.header
        className="landing-header"
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
      >
        <a className="landing-brand" href="/" aria-label="Forma home">
          <img className="landing-brand-mark" src="/favicon.svg" alt="" />
          <span>FORMA</span>
        </a>
        <span className="landing-header-label">LOCAL GRAPH WORKBENCH</span>
      </motion.header>

      <motion.footer
        className="landing-statusbar"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.45, delay: 0.7 }}
      >
        <div className="landing-statusbar-group">
          <span>CLIENT / BROWSER</span>
          <span>TRANSPORT / NONE</span>
        </div>
        <div className="landing-statusbar-group">
          <a href="https://donatr.ee/hussain" target="_blank" rel="noreferrer">
            <NerdGlyph glyph="bolt" /> SUPPORT
          </a>
          <a href="https://github.com/Hussain004/Forma" target="_blank" rel="noreferrer">
            <NerdGlyph glyph="github" /> SOURCE
          </a>
          <a
            href="https://github.com/Hussain004/Forma/blob/master/README.md"
            target="_blank"
            rel="noreferrer"
          >
            README <NerdGlyph glyph="external" />
          </a>
        </div>
      </motion.footer>
    </>
  )
}
