import {
  addCustomNode,
  deleteNodeWithReconnect,
  insertPassthroughNode,
  rewireEdge,
  structuralNodeIndex,
  toSelectableGraph,
  validateRewire,
  type HistoryEntry,
  type SelectableGraph,
} from './graphUtils'
import type { OnnxGraph } from './onnxTypes'

const SHARE_PREFIX = '#s='
const SHARE_VERSION = 1
const MAX_HASH_LENGTH = 100_000
const MAX_EDIT_COUNT = 1_000
const MAX_TEXT_LENGTH = 2_048

type EncodedDelete = ['d', number, number | null]
type EncodedEdit =
  | ['a', number, string, string | number]
  | EncodedDelete
  | ['b', EncodedDelete[]]
  | ['p', number, number, number]
  | ['r', number, number, number]
  | ['n', number, string, number, number, number]

interface CompactSharePayload {
  v: 1
  h: string
  n?: string
  e: EncodedEdit[]
}

export interface ShareLinkPayload {
  version: 1
  modelHash: string
  modelName?: string
  edits: EncodedEdit[]
}

export interface RestoredShareHistory {
  entries: HistoryEntry[]
  passthroughCounter: number
  customNodeCounter: number
}

export class ShareLinkError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ShareLinkError'
  }
}

function bytesToBase64Url(bytes: Uint8Array): string {
  let binary = ''
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000))
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '')
}

function base64UrlToBytes(encoded: string): Uint8Array {
  if (!/^[A-Za-z0-9_-]+$/.test(encoded)) {
    throw new ShareLinkError('The shared edit link contains invalid characters')
  }
  const padding = '='.repeat((4 - encoded.length % 4) % 4)
  let binary: string
  try {
    binary = atob(encoded.replace(/-/g, '+').replace(/_/g, '/') + padding)
  } catch {
    throw new ShareLinkError('The shared edit link could not be decoded')
  }
  return Uint8Array.from(binary, (character) => character.charCodeAt(0))
}

function assertInteger(value: unknown, label: string, minimum = -1_000_000, maximum = 1_000_000): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    throw new ShareLinkError(`${label} is invalid`)
  }
  return value as number
}

function assertText(value: unknown, label: string, maximum = MAX_TEXT_LENGTH): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > maximum) {
    throw new ShareLinkError(`${label} is invalid`)
  }
  return value
}

function assertEditValue(value: unknown): string | number {
  if (typeof value === 'string' && value.length <= MAX_TEXT_LENGTH) return value
  if (typeof value === 'number' && Number.isFinite(value)) return value
  throw new ShareLinkError('An attribute value in the shared edit link is invalid')
}

function encodeDelete(entry: Extract<HistoryEntry, { type: 'delete' }>): EncodedDelete {
  return ['d', entry.nodeIndex, entry.keepInputPosition]
}

function encodeHistoryEntry(entry: HistoryEntry): EncodedEdit {
  switch (entry.type) {
    case 'attr': {
      const nodeIndex = structuralNodeIndex(entry.nodeId)
      if (nodeIndex === null) throw new ShareLinkError('An attribute edit has an invalid node')
      return ['a', nodeIndex, entry.attrName, entry.value]
    }
    case 'delete':
      return encodeDelete(entry)
    case 'bulkDelete':
      return ['b', entry.deletions.map(encodeDelete)]
    case 'insertPassthrough': {
      const match = /^passthrough_(\d+)$/.exec(entry.newNodeId)
      if (!match) throw new ShareLinkError('A passthrough edit has an invalid node id')
      return ['p', entry.targetNodeIndex, entry.inputPosition, Number(match[1])]
    }
    case 'rewire':
      return ['r', entry.sourceNodeIndex, entry.targetNodeIndex, entry.inputPosition]
    case 'addNode':
      return [
        'n',
        entry.newNodeIndex,
        entry.opType,
        entry.inputCount,
        entry.position.x,
        entry.position.y,
      ]
  }
}

function operationCount(edits: EncodedEdit[]): number {
  return edits.reduce((total, edit) => total + (edit[0] === 'b' ? edit[1].length : 1), 0)
}

function decodeDelete(raw: unknown, label: string): EncodedDelete {
  if (!Array.isArray(raw) || raw.length !== 3 || raw[0] !== 'd') {
    throw new ShareLinkError(`${label} is invalid`)
  }
  const nodeIndex = assertInteger(raw[1], `${label} node`)
  const keepInputPosition = raw[2] === null
    ? null
    : assertInteger(raw[2], `${label} input`, 0, 255)
  return ['d', nodeIndex, keepInputPosition]
}

function decodeEdit(raw: unknown, editIndex: number): EncodedEdit {
  const label = `Shared edit ${editIndex + 1}`
  if (!Array.isArray(raw) || typeof raw[0] !== 'string') {
    throw new ShareLinkError(`${label} is invalid`)
  }
  switch (raw[0]) {
    case 'a':
      if (raw.length !== 4) throw new ShareLinkError(`${label} is invalid`)
      return [
        'a',
        assertInteger(raw[1], `${label} node`),
        assertText(raw[2], `${label} attribute`, 256),
        assertEditValue(raw[3]),
      ]
    case 'd':
      return decodeDelete(raw, label)
    case 'b': {
      if (raw.length !== 2 || !Array.isArray(raw[1]) || raw[1].length === 0 || raw[1].length > MAX_EDIT_COUNT) {
        throw new ShareLinkError(`${label} is invalid`)
      }
      return ['b', raw[1].map((deletion, index) => decodeDelete(deletion, `${label} deletion ${index + 1}`))]
    }
    case 'p':
      if (raw.length !== 4) throw new ShareLinkError(`${label} is invalid`)
      return [
        'p',
        assertInteger(raw[1], `${label} target`),
        assertInteger(raw[2], `${label} input`, 0, 255),
        assertInteger(raw[3], `${label} passthrough`, 1, 1_000_000),
      ]
    case 'r':
      if (raw.length !== 4) throw new ShareLinkError(`${label} is invalid`)
      return [
        'r',
        assertInteger(raw[1], `${label} source`),
        assertInteger(raw[2], `${label} target`),
        assertInteger(raw[3], `${label} input`, 0, 255),
      ]
    case 'n': {
      if (raw.length !== 6) throw new ShareLinkError(`${label} is invalid`)
      const x = raw[4]
      const y = raw[5]
      if (typeof x !== 'number' || !Number.isFinite(x) || typeof y !== 'number' || !Number.isFinite(y)) {
        throw new ShareLinkError(`${label} position is invalid`)
      }
      return [
        'n',
        assertInteger(raw[1], `${label} node`, 1, 1_000_000),
        assertText(raw[2], `${label} operator`, 128),
        assertInteger(raw[3], `${label} input count`, 1, 255),
        x,
        y,
      ]
    }
    default:
      throw new ShareLinkError(`${label} uses an unsupported operation`)
  }
}

export async function hashModelBuffer(buffer: ArrayBuffer): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new ShareLinkError('This browser does not support secure model verification')
  }
  const digest = await globalThis.crypto.subtle.digest('SHA-256', buffer)
  return bytesToBase64Url(new Uint8Array(digest))
}

export function createShareHash(
  modelHash: string,
  modelName: string,
  entries: HistoryEntry[],
): string {
  if (!/^[A-Za-z0-9_-]{43}$/.test(modelHash)) {
    throw new ShareLinkError('The model fingerprint is invalid')
  }
  if (entries.length === 0 || entries.length > MAX_EDIT_COUNT) {
    throw new ShareLinkError('The edit history cannot be shared')
  }
  const edits = entries.map(encodeHistoryEntry)
  if (operationCount(edits) > MAX_EDIT_COUNT) {
    throw new ShareLinkError('The edit history is too large for a share link')
  }
  const payload: CompactSharePayload = {
    v: SHARE_VERSION,
    h: modelHash,
    n: modelName.slice(0, 255),
    e: edits,
  }
  const bytes = new TextEncoder().encode(JSON.stringify(payload))
  const hash = SHARE_PREFIX + bytesToBase64Url(bytes)
  if (hash.length > MAX_HASH_LENGTH) {
    throw new ShareLinkError('The edit history is too large for a share link')
  }
  return hash
}

export function parseShareHash(hash: string): ShareLinkPayload | null {
  if (!hash.startsWith(SHARE_PREFIX)) return null
  if (hash.length > MAX_HASH_LENGTH) {
    throw new ShareLinkError('The shared edit link is too large')
  }
  const encoded = hash.slice(SHARE_PREFIX.length)
  if (!encoded) throw new ShareLinkError('The shared edit link is empty')

  let raw: unknown
  try {
    raw = JSON.parse(new TextDecoder().decode(base64UrlToBytes(encoded)))
  } catch (error) {
    if (error instanceof ShareLinkError) throw error
    throw new ShareLinkError('The shared edit link contains invalid data')
  }
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new ShareLinkError('The shared edit link contains invalid data')
  }
  const compact = raw as Partial<CompactSharePayload>
  if (compact.v !== SHARE_VERSION) {
    throw new ShareLinkError('This shared edit link uses an unsupported version')
  }
  if (typeof compact.h !== 'string' || !/^[A-Za-z0-9_-]{43}$/.test(compact.h)) {
    throw new ShareLinkError('The shared edit link has an invalid model fingerprint')
  }
  if (compact.n !== undefined && (typeof compact.n !== 'string' || compact.n.length > 255)) {
    throw new ShareLinkError('The shared edit link has an invalid model name')
  }
  if (!Array.isArray(compact.e) || compact.e.length === 0 || compact.e.length > MAX_EDIT_COUNT) {
    throw new ShareLinkError('The shared edit link has an invalid edit history')
  }
  const edits = compact.e.map(decodeEdit)
  if (operationCount(edits) > MAX_EDIT_COUNT) {
    throw new ShareLinkError('The shared edit link has too many operations')
  }
  return {
    version: SHARE_VERSION,
    modelHash: compact.h,
    modelName: compact.n,
    edits,
  }
}

function originalNodeIds(graph: OnnxGraph): Map<number, string> {
  const result = new Map<number, string>()
  for (const node of graph.nodes) {
    const index = structuralNodeIndex(node.id)
    if (index !== null && index >= 0) result.set(index, node.id)
  }
  return result
}

function requireNodeId(nodes: Map<number, string>, nodeIndex: number, label: string): string {
  const nodeId = nodes.get(nodeIndex)
  if (!nodeId) throw new ShareLinkError(`${label} does not exist in this model state`)
  return nodeId
}

function applyDelete(
  graph: SelectableGraph,
  nodes: Map<number, string>,
  encoded: EncodedDelete,
): { graph: SelectableGraph; entry: Extract<HistoryEntry, { type: 'delete' }> } {
  const [, nodeIndex, keepInputPosition] = encoded
  const nodeId = requireNodeId(nodes, nodeIndex, 'A deleted node')
  const node = graph.nodes.find((candidate) => candidate.id === nodeId)
  if (!node || (keepInputPosition !== null && keepInputPosition >= node.inputs.length)) {
    throw new ShareLinkError('A delete operation is not valid for this model state')
  }
  const nextGraph = deleteNodeWithReconnect(graph, nodeId, keepInputPosition)
  if (nextGraph.nodes.some((candidate) => candidate.id === nodeId)) {
    throw new ShareLinkError('A delete operation could not be replayed')
  }
  nodes.delete(nodeIndex)
  return {
    graph: nextGraph,
    entry: { type: 'delete', nodeId, nodeIndex, keepInputPosition },
  }
}

export function restoreSharedHistory(payload: ShareLinkPayload, graph: OnnxGraph): RestoredShareHistory {
  const entries: HistoryEntry[] = []
  const nodes = originalNodeIds(graph)
  let liveGraph = toSelectableGraph(graph)
  let passthroughCounter = 0
  let customNodeCounter = 0

  for (const encoded of payload.edits) {
    switch (encoded[0]) {
      case 'a': {
        const [, nodeIndex, attrName, value] = encoded
        const nodeId = requireNodeId(nodes, nodeIndex, 'An edited node')
        const node = liveGraph.nodes.find((candidate) => candidate.id === nodeId)
        if (!node || !(attrName in node.attributes)) {
          throw new ShareLinkError('An attribute edit does not match this model')
        }
        entries.push({ type: 'attr', nodeId, attrName, value })
        break
      }
      case 'd': {
        const result = applyDelete(liveGraph, nodes, encoded)
        liveGraph = result.graph
        entries.push(result.entry)
        break
      }
      case 'b': {
        const deletions: Extract<HistoryEntry, { type: 'delete' }>[] = []
        for (const deletion of encoded[1]) {
          const result = applyDelete(liveGraph, nodes, deletion)
          liveGraph = result.graph
          deletions.push(result.entry)
        }
        entries.push({ type: 'bulkDelete', deletions })
        break
      }
      case 'p': {
        const [, targetNodeIndex, inputPosition, newNodeNumber] = encoded
        const targetNodeId = requireNodeId(nodes, targetNodeIndex, 'A passthrough target')
        const newNodeId = `passthrough_${newNodeNumber}`
        const nextGraph = insertPassthroughNode(liveGraph, targetNodeId, inputPosition, newNodeId)
        if (!nextGraph.nodes.some((node) => node.id === newNodeId)) {
          throw new ShareLinkError('A passthrough operation could not be replayed')
        }
        liveGraph = nextGraph
        passthroughCounter = Math.max(passthroughCounter, newNodeNumber)
        entries.push({ type: 'insertPassthrough', targetNodeId, targetNodeIndex, inputPosition, newNodeId })
        break
      }
      case 'r': {
        const [, sourceNodeIndex, targetNodeIndex, inputPosition] = encoded
        const sourceNodeId = requireNodeId(nodes, sourceNodeIndex, 'A rewire source')
        const targetNodeId = requireNodeId(nodes, targetNodeIndex, 'A rewire target')
        const validation = validateRewire(liveGraph, sourceNodeId, targetNodeId, inputPosition)
        if (!validation.valid) {
          throw new ShareLinkError(`A rewire operation is invalid: ${validation.reason ?? 'unknown reason'}`)
        }
        liveGraph = rewireEdge(liveGraph, sourceNodeId, targetNodeId, inputPosition)
        entries.push({ type: 'rewire', sourceNodeId, sourceNodeIndex, targetNodeId, targetNodeIndex, inputPosition })
        break
      }
      case 'n': {
        const [, newNodeIndex, opType, inputCount, x, y] = encoded
        const nodeIndex = -newNodeIndex
        if (nodes.has(nodeIndex)) throw new ShareLinkError('A custom node id is duplicated')
        const newNodeId = `custom_${newNodeIndex}`
        liveGraph = addCustomNode(liveGraph, newNodeId, opType, inputCount, { x, y })
        nodes.set(nodeIndex, newNodeId)
        customNodeCounter = Math.max(customNodeCounter, newNodeIndex)
        entries.push({ type: 'addNode', newNodeId, newNodeIndex, opType, inputCount, position: { x, y } })
        break
      }
    }
  }

  return { entries, passthroughCounter, customNodeCounter }
}
