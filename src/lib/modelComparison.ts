// v2.5's structural side of model comparison: given two independently loaded
// OnnxGraphs (baseline and candidate, not an original/edited pair of the
// SAME file, two separate files), diff their op-type histogram, node-level
// attributes, initializers, and graph I/O, purely from data the worker
// already produces on LOAD_MODEL. No inference involved; see
// ModelComparePage.tsx for the output/latency side, which does need it.
import { computeOpCounts, type AttrHistoryEntry } from './graphUtils'
import type { OnnxDim, OnnxGraph, OnnxNode, TensorMetadata } from './onnxTypes'

function isCompute(n: OnnxNode): boolean {
  return n.opType !== 'Input' && n.opType !== 'Output'
}

function dimsToNumbers(shape?: OnnxDim[]): number[] | undefined {
  return shape?.map((d) => ('value' in d ? d.value : -1))
}

function dimsToDisplay(shape?: OnnxDim[]): (number | string)[] | undefined {
  return shape?.map((d) => ('value' in d ? d.value : d.param))
}

export interface OpCountChange {
  opType: string
  baselineCount: number
  candidateCount: number
}

export function diffOpCounts(baseline: OnnxGraph, candidate: OnnxGraph): OpCountChange[] {
  const b = computeOpCounts(baseline.nodes)
  const c = computeOpCounts(candidate.nodes)
  const opTypes = [...new Set([...Object.keys(b), ...Object.keys(c)])].sort()
  return opTypes
    .map((opType) => ({ opType, baselineCount: b[opType] ?? 0, candidateCount: c[opType] ?? 0 }))
    .filter((entry) => entry.baselineCount !== entry.candidateCount)
}

export interface NodeMatch {
  baselineId: string
  candidateId: string
  opType: string
}

export interface NodeMatchResult {
  matches: NodeMatch[]
  addedNodes: OnnxNode[]
  removedNodes: OnnxNode[]
  matchedByName: boolean
}

// Two graphs are two independent files, not one graph before/after an edit,
// so there's no shared node-index addressing to lean on. If every node on
// both sides has a unique name (true for most real exports: PyTorch/TF
// assign names like "/layer1/conv1/Conv"), match by name. Otherwise fall
// back to pairing the Nth occurrence of each op type on one side with the
// Nth on the other: naive, but exactly right for "same architecture,
// changed hyperparameters" and degrades gracefully (extra nodes just show as
// added/removed) for anything else.
export function matchNodes(baseline: OnnxGraph, candidate: OnnxGraph): NodeMatchResult {
  const baseNodes = baseline.nodes.filter(isCompute)
  const candNodes = candidate.nodes.filter(isCompute)

  const uniquelyNamed = (nodes: OnnxNode[]) =>
    nodes.every((n) => !!n.name) && new Set(nodes.map((n) => n.name)).size === nodes.length
  const matchedByName = uniquelyNamed(baseNodes) && uniquelyNamed(candNodes)

  const matches: NodeMatch[] = []
  const matchedBaseIds = new Set<string>()
  const matchedCandIds = new Set<string>()

  if (matchedByName) {
    const candByName = new Map(candNodes.map((n) => [n.name!, n]))
    for (const b of baseNodes) {
      const c = candByName.get(b.name!)
      if (!c) continue
      matches.push({ baselineId: b.id, candidateId: c.id, opType: b.opType })
      matchedBaseIds.add(b.id)
      matchedCandIds.add(c.id)
    }
  } else {
    const groupByOp = (nodes: OnnxNode[]) => {
      const groups = new Map<string, OnnxNode[]>()
      for (const n of nodes) {
        const list = groups.get(n.opType) ?? []
        list.push(n)
        groups.set(n.opType, list)
      }
      return groups
    }
    const baseByOp = groupByOp(baseNodes)
    const candByOp = groupByOp(candNodes)
    for (const [opType, bList] of baseByOp) {
      const cList = candByOp.get(opType) ?? []
      const n = Math.min(bList.length, cList.length)
      for (let i = 0; i < n; i++) {
        matches.push({ baselineId: bList[i].id, candidateId: cList[i].id, opType })
        matchedBaseIds.add(bList[i].id)
        matchedCandIds.add(cList[i].id)
      }
    }
  }

  return {
    matches,
    addedNodes: candNodes.filter((n) => !matchedCandIds.has(n.id)),
    removedNodes: baseNodes.filter((n) => !matchedBaseIds.has(n.id)),
    matchedByName,
  }
}

export interface AttributeChange {
  baselineNodeId: string
  candidateNodeId: string
  opType: string
  attrName: string
  baselineValue: string | number | boolean | undefined
  candidateValue: string | number | boolean | undefined
}

export function diffAttributes(baseline: OnnxGraph, candidate: OnnxGraph, matches: NodeMatch[]): AttributeChange[] {
  const baseById = new Map(baseline.nodes.map((n) => [n.id, n]))
  const candById = new Map(candidate.nodes.map((n) => [n.id, n]))
  const changes: AttributeChange[] = []
  for (const m of matches) {
    const b = baseById.get(m.baselineId)
    const c = candById.get(m.candidateId)
    if (!b || !c) continue
    const keys = new Set([...Object.keys(b.attributes), ...Object.keys(c.attributes)])
    for (const attrName of keys) {
      const baselineValue = b.attributes[attrName]
      const candidateValue = c.attributes[attrName]
      if (baselineValue !== candidateValue) {
        changes.push({ baselineNodeId: m.baselineId, candidateNodeId: m.candidateId, opType: m.opType, attrName, baselineValue, candidateValue })
      }
    }
  }
  return changes
}

// Initializers aren't exposed as their own top-level list on OnnxGraph (only
// small ones carry `values`, folded into each consuming node's
// inputMetadata), so reconstruct the set as "any tensor name consumed by a
// node that no node (including the Input pseudo-nodes) produces."
function collectInitializers(graph: OnnxGraph): Map<string, TensorMetadata> {
  const producers = new Set<string>()
  for (const n of graph.nodes) for (const out of n.outputs) producers.add(out)
  const result = new Map<string, TensorMetadata>()
  for (const n of graph.nodes) {
    n.inputs.forEach((name, i) => {
      if (producers.has(name)) return
      const meta = n.inputMetadata?.[i]
      if (meta && (meta.shape !== undefined || meta.elemType !== undefined)) result.set(name, meta)
    })
  }
  return result
}

export interface InitializerChange {
  name: string
  status: 'added' | 'removed' | 'changed' | 'unchanged'
  baselineShape?: number[]
  candidateShape?: number[]
  baselineElemType?: number
  candidateElemType?: number
}

export function diffInitializers(baseline: OnnxGraph, candidate: OnnxGraph): InitializerChange[] {
  const b = collectInitializers(baseline)
  const c = collectInitializers(candidate)
  const names = [...new Set([...b.keys(), ...c.keys()])].sort()
  return names.map((name) => {
    const bm = b.get(name)
    const cm = c.get(name)
    const baselineShape = dimsToNumbers(bm?.shape)
    const candidateShape = dimsToNumbers(cm?.shape)
    if (bm && !cm) return { name, status: 'removed' as const, baselineShape, baselineElemType: bm.elemType }
    if (!bm && cm) return { name, status: 'added' as const, candidateShape, candidateElemType: cm.elemType }
    const shapeSame = JSON.stringify(baselineShape) === JSON.stringify(candidateShape)
    const typeSame = bm!.elemType === cm!.elemType
    return {
      name,
      status: shapeSame && typeSame ? ('unchanged' as const) : ('changed' as const),
      baselineShape, candidateShape,
      baselineElemType: bm!.elemType, candidateElemType: cm!.elemType,
    }
  })
}

export interface IOChange {
  ioKind: 'input' | 'output'
  name: string
  status: 'added' | 'removed' | 'changed' | 'unchanged'
  baselineShape?: (number | string)[]
  candidateShape?: (number | string)[]
  baselineElemType?: number
  candidateElemType?: number
}

function graphOutputsOf(graph: OnnxGraph): { name: string; shape?: OnnxDim[]; elemType?: number }[] {
  return graph.nodes
    .filter((n) => n.opType === 'Output')
    .map((n) => ({ name: n.inputs[0], shape: n.inputShapes?.[0], elemType: n.inputMetadata?.[0]?.elemType }))
}

function diffIOList(
  ioKind: 'input' | 'output',
  baseList: { name: string; shape?: OnnxDim[]; elemType?: number }[],
  candList: { name: string; shape?: OnnxDim[]; elemType?: number }[],
): IOChange[] {
  const b = new Map(baseList.map((v) => [v.name, v]))
  const c = new Map(candList.map((v) => [v.name, v]))
  const names = [...new Set([...b.keys(), ...c.keys()])]
  return names.map((name) => {
    const bv = b.get(name)
    const cv = c.get(name)
    const baselineShape = dimsToDisplay(bv?.shape)
    const candidateShape = dimsToDisplay(cv?.shape)
    if (bv && !cv) return { ioKind, name, status: 'removed' as const, baselineShape, baselineElemType: bv.elemType }
    if (!bv && cv) return { ioKind, name, status: 'added' as const, candidateShape, candidateElemType: cv.elemType }
    const same = JSON.stringify(baselineShape) === JSON.stringify(candidateShape) && bv!.elemType === cv!.elemType
    return {
      ioKind, name,
      status: same ? ('unchanged' as const) : ('changed' as const),
      baselineShape, candidateShape,
      baselineElemType: bv!.elemType, candidateElemType: cv!.elemType,
    }
  })
}

export function diffGraphIO(baseline: OnnxGraph, candidate: OnnxGraph): IOChange[] {
  const inputs = diffIOList('input', baseline.graphInputs ?? [], candidate.graphInputs ?? [])
  const outputs = diffIOList('output', graphOutputsOf(baseline), graphOutputsOf(candidate))
  return [...inputs, ...outputs]
}

export interface MetadataChange {
  field: string
  baselineValue: string | number
  candidateValue: string | number
}

export function diffMetadata(baseline: OnnxGraph, candidate: OnnxGraph): MetadataChange[] {
  const bm = baseline.metadata
  const cm = candidate.metadata
  const changes: MetadataChange[] = []
  const fields: [string, (m: typeof bm) => string | number | undefined][] = [
    ['Opset version', (m) => m?.opsetVersion],
    ['IR version', (m) => m?.irVersion],
    ['Producer', (m) => m?.producerName],
    ['Producer version', (m) => m?.producerVersion],
  ]
  for (const [field, get] of fields) {
    const baselineValue = get(bm)
    const candidateValue = get(cm)
    if (baselineValue !== candidateValue) {
      changes.push({ field, baselineValue: baselineValue ?? '', candidateValue: candidateValue ?? '' })
    }
  }
  return changes
}

export interface ModelComparison {
  opCounts: OpCountChange[]
  nodeMatch: NodeMatchResult
  attributes: AttributeChange[]
  initializers: InitializerChange[]
  io: IOChange[]
  metadata: MetadataChange[]
  baselineNodeCount: number
  candidateNodeCount: number
}

export function compareModels(baseline: OnnxGraph, candidate: OnnxGraph): ModelComparison {
  const nodeMatch = matchNodes(baseline, candidate)
  return {
    opCounts: diffOpCounts(baseline, candidate),
    nodeMatch,
    attributes: diffAttributes(baseline, candidate, nodeMatch.matches),
    initializers: diffInitializers(baseline, candidate).filter((i) => i.status !== 'unchanged'),
    io: diffGraphIO(baseline, candidate),
    metadata: diffMetadata(baseline, candidate),
    baselineNodeCount: baseline.nodes.filter(isCompute).length,
    candidateNodeCount: candidate.nodes.filter(isCompute).length,
  }
}

// An "applicable edit recipe" (per the v2.5 roadmap line) only makes sense
// when candidate is reachable from baseline purely by attribute edits:
// same nodes matched by name, nothing added/removed/reshaped. In that case
// the diff is literally a Forma attribute-edit history, and the existing
// share-link mechanism (shareLinks.ts) already knows how to encode and
// replay exactly that against the baseline model.
export function attributeOnlyEditRecipe(comparison: ModelComparison): AttrHistoryEntry[] | null {
  if (!comparison.nodeMatch.matchedByName) return null
  if (comparison.nodeMatch.addedNodes.length > 0 || comparison.nodeMatch.removedNodes.length > 0) return null
  if (comparison.initializers.length > 0) return null
  if (comparison.io.some((c) => c.status !== 'unchanged')) return null
  if (comparison.attributes.length === 0) return null
  return comparison.attributes.map((change) => ({
    type: 'attr' as const,
    nodeId: change.baselineNodeId,
    attrName: change.attrName,
    value: typeof change.candidateValue === 'boolean' ? String(change.candidateValue) : change.candidateValue ?? '',
  }))
}

function formatShape(shape?: (number | string)[]): string {
  if (!shape) return '?'
  return `[${shape.join(', ')}]`
}

function formatValue(value: string | number | boolean | undefined): string {
  if (value === undefined) return '(none)'
  return String(value)
}

export function formatComparisonReport(
  comparison: ModelComparison,
  baselineName: string,
  candidateName: string,
  extra?: {
    latency?: { baseline: { avgMs: number; medianMs: number }; candidate: { avgMs: number; medianMs: number } }
    outputs?: { name: string; shapeMatch: boolean; maxAbsErr?: number; cosineSim?: number }[]
  },
): string {
  const lines: string[] = []
  lines.push('Forma Model Comparison Report')
  lines.push('==============================')
  lines.push('')
  lines.push(`Baseline:  ${baselineName}`)
  lines.push(`Candidate: ${candidateName}`)
  lines.push('')
  lines.push('Summary')
  lines.push('-------')
  lines.push(`Nodes: ${comparison.baselineNodeCount} -> ${comparison.candidateNodeCount}`)
  lines.push(`Nodes matched: ${comparison.nodeMatch.matches.length} (by ${comparison.nodeMatch.matchedByName ? 'name' : 'op-type position'})`)
  lines.push(`Nodes added: ${comparison.nodeMatch.addedNodes.length}`)
  lines.push(`Nodes removed: ${comparison.nodeMatch.removedNodes.length}`)
  lines.push('')

  if (comparison.metadata.length > 0) {
    lines.push('Metadata changes')
    lines.push('-----------------')
    for (const m of comparison.metadata) lines.push(`${m.field}: ${m.baselineValue} -> ${m.candidateValue}`)
    lines.push('')
  }

  if (comparison.opCounts.length > 0) {
    lines.push('Op type count changes')
    lines.push('----------------------')
    for (const c of comparison.opCounts) lines.push(`${c.opType}: ${c.baselineCount} -> ${c.candidateCount}`)
    lines.push('')
  }

  if (comparison.nodeMatch.addedNodes.length > 0) {
    const header = `Nodes added in candidate (${comparison.nodeMatch.addedNodes.length})`
    lines.push(header, '-'.repeat(header.length))
    for (const n of comparison.nodeMatch.addedNodes) lines.push(`  ${n.opType} (${n.name || n.id})`)
    lines.push('')
  }

  if (comparison.nodeMatch.removedNodes.length > 0) {
    const header = `Nodes removed from baseline (${comparison.nodeMatch.removedNodes.length})`
    lines.push(header, '-'.repeat(header.length))
    for (const n of comparison.nodeMatch.removedNodes) lines.push(`  ${n.opType} (${n.name || n.id})`)
    lines.push('')
  }

  if (comparison.attributes.length > 0) {
    const header = `Attribute changes (${comparison.attributes.length})`
    lines.push(header, '-'.repeat(header.length))
    for (const a of comparison.attributes) {
      lines.push(`  ${a.opType} ${a.attrName}: ${formatValue(a.baselineValue)} -> ${formatValue(a.candidateValue)}`)
    }
    lines.push('')
  }

  if (comparison.initializers.length > 0) {
    const header = `Initializer changes (${comparison.initializers.length})`
    lines.push(header, '-'.repeat(header.length))
    for (const i of comparison.initializers) {
      lines.push(`  ${i.name}: ${i.status} (${formatShape(i.baselineShape)} -> ${formatShape(i.candidateShape)})`)
    }
    lines.push('')
  }

  const changedIO = comparison.io.filter((c) => c.status !== 'unchanged')
  if (changedIO.length > 0) {
    const header = `Graph I/O changes (${changedIO.length})`
    lines.push(header, '-'.repeat(header.length))
    for (const io of changedIO) {
      lines.push(`  [${io.ioKind}] ${io.name}: ${io.status} (${formatShape(io.baselineShape)} -> ${formatShape(io.candidateShape)})`)
    }
    lines.push('')
  }

  if (extra?.latency) {
    lines.push('Latency (avg / median ms, ONNX Runtime Web, this machine)')
    lines.push('-----------------------------------------------------------')
    lines.push(`Baseline:  ${extra.latency.baseline.avgMs.toFixed(2)} / ${extra.latency.baseline.medianMs.toFixed(2)}`)
    lines.push(`Candidate: ${extra.latency.candidate.avgMs.toFixed(2)} / ${extra.latency.candidate.medianMs.toFixed(2)}`)
    lines.push('')
  }

  if (extra?.outputs && extra.outputs.length > 0) {
    lines.push('Output comparison (generated inputs, ONNX Runtime Web)')
    lines.push('--------------------------------------------------------')
    for (const o of extra.outputs) {
      if (!o.shapeMatch) { lines.push(`  ${o.name}: shape mismatch`); continue }
      lines.push(`  ${o.name}: max abs err ${o.maxAbsErr?.toFixed(6)}, cosine sim ${o.cosineSim?.toFixed(6)}`)
    }
    lines.push('')
  }

  return lines.join('\n')
}
