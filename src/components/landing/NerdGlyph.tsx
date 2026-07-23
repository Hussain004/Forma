const NERD_GLYPHS = {
  arrowRight: '\uf061',
  bolt: '\uf0e7',
  check: '\uf00c',
  cube: '\uf1b2',
  external: '\uf08e',
  fileCode: '\uf1c9',
  github: '\uf09b',
  lock: '\uf023',
  terminal: '\uf120',
  upload: '\uf093',
} as const

interface NerdGlyphProps {
  glyph: keyof typeof NERD_GLYPHS
  className?: string
}

export function NerdGlyph({ glyph, className }: NerdGlyphProps) {
  return (
    <span className={className ? `nerd-glyph ${className}` : 'nerd-glyph'} aria-hidden="true">
      {NERD_GLYPHS[glyph]}
    </span>
  )
}
