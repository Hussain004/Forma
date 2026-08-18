// Pure comparison math for behavioral validation: given the original model's
// and the edited model's outputs on identical inputs, how far apart are they.
// Kept dependency-free of onnxruntime/the worker so it can run (and be
// tested) on the main thread against plain typed arrays.

export interface TensorOutput {
  shape: number[]
  data: Float32Array
}

export type SideRunResult = {
  loaded: boolean
  inferenceOk: boolean
  error?: string
  outputs: Record<string, TensorOutput>
}

export interface ValidationRunResult {
  inputSource: 'provided' | 'generated'
  original: SideRunResult
  modified: SideRunResult
}

export function maxAbsError(a: Float32Array, b: Float32Array): number {
  let max = 0
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]))
  return max
}

// Relative to the original value's magnitude, with a small floor so a
// near-zero original doesn't blow the ratio up over a negligible absolute
// difference.
export function maxRelError(a: Float32Array, b: Float32Array, eps = 1e-6): number {
  let max = 0
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]) / (Math.abs(a[i]) + eps))
  return max
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0, magA = 0, magB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    magA += a[i] * a[i]
    magB += b[i] * b[i]
  }
  if (magA === 0 || magB === 0) return magA === magB ? 1 : 0
  return dot / (Math.sqrt(magA) * Math.sqrt(magB))
}

function topKIndices(values: Float32Array, k: number): Set<number> {
  return new Set(
    Array.from(values)
      .map((value, index) => ({ value, index }))
      .sort((x, y) => y.value - x.value)
      .slice(0, k)
      .map((entry) => entry.index),
  )
}

// Only meaningful for classification-shaped outputs (a flat score vector).
// Returns null when it isn't applicable rather than a misleading number.
export function topKAgreement(a: Float32Array, b: Float32Array, k = 5): { k: number; overlap: number } | null {
  if (a.length < 2) return null
  const effectiveK = Math.min(k, a.length)
  const topA = topKIndices(a, effectiveK)
  const topB = topKIndices(b, effectiveK)
  let overlap = 0
  for (const index of topA) if (topB.has(index)) overlap++
  return { k: effectiveK, overlap }
}

export interface OutputComparison {
  name: string
  presentInOriginal: boolean
  presentInModified: boolean
  shapeMatch: boolean
  originalShape?: number[]
  modifiedShape?: number[]
  maxAbsErr?: number
  maxRelErr?: number
  cosineSim?: number
  topK?: { k: number; overlap: number } | null
}

export function compareOutputs(
  original: Record<string, TensorOutput>,
  modified: Record<string, TensorOutput>,
  topK = 5,
): OutputComparison[] {
  const names = [...new Set([...Object.keys(original), ...Object.keys(modified)])]
  return names.map((name) => {
    const o = original[name]
    const m = modified[name]
    if (!o || !m) {
      return { name, presentInOriginal: !!o, presentInModified: !!m, shapeMatch: false }
    }
    const shapeMatch = o.shape.length === m.shape.length && o.shape.every((d, i) => d === m.shape[i])
    if (!shapeMatch) {
      return {
        name, presentInOriginal: true, presentInModified: true, shapeMatch: false,
        originalShape: o.shape, modifiedShape: m.shape,
      }
    }
    return {
      name,
      presentInOriginal: true,
      presentInModified: true,
      shapeMatch: true,
      originalShape: o.shape,
      modifiedShape: m.shape,
      maxAbsErr: maxAbsError(o.data, m.data),
      maxRelErr: maxRelError(o.data, m.data),
      cosineSim: cosineSimilarity(o.data, m.data),
      topK: topKAgreement(o.data, m.data, topK),
    }
  })
}
