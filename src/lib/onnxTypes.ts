import type { OnnxDim } from './onnxProtoParser'

export type { OnnxDim }

export interface OnnxNode {
  id: string
  opType: string
  inputs: string[]
  outputs: string[]
  attributes: Record<string, string | number | boolean>
  paramCount: number
  estimatedSizeMB: number
  inputShapes?: OnnxDim[][]
  outputShapes?: OnnxDim[][]
}

export interface OnnxEdge {
  id: string
  source: string
  target: string
  label?: string
}

export interface OnnxGraph {
  nodes: OnnxNode[]
  edges: OnnxEdge[]
  modelName: string
  totalParams: number
  totalSizeMB: number
}
