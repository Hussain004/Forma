export function inferAttrType(v: string | number): 'int' | 'float' | 'array' | 'string' {
  if (typeof v === 'number') return Number.isInteger(v) ? 'int' : 'float'
  if (typeof v === 'string' && v.startsWith('[')) return 'array'
  return 'string'
}

export function parseAttrEdit(raw: string, original: string | number): string | number {
  const trimmed = raw.trim()
  if (!trimmed) return original
  const type = inferAttrType(original)
  if (type === 'int') {
    const n = parseInt(trimmed, 10)
    return isNaN(n) ? original : n
  }
  if (type === 'float') {
    const n = parseFloat(trimmed)
    return isNaN(n) ? original : parseFloat(n.toPrecision(6))
  }
  if (type === 'array') {
    const inner = trimmed.replace(/^\[|\]$/g, '').split(',').map(s => s.trim())
    const nums = inner.map(p => parseFloat(p))
    if (nums.some(n => isNaN(n))) return original
    return '[' + nums.join(', ') + ']'
  }
  return trimmed
}
