import type { OnnxDim, ParsedValueInfo, ModelMetadata } from './onnxProtoParser'

export type { OnnxDim, ParsedValueInfo, ModelMetadata }

export interface OnnxNode {
  id: string
  name?: string
  opType: string
  inputs: string[]
  outputs: string[]
  attributes: Record<string, string | number | boolean>
  paramCount: number
  estimatedSizeMB: number
  inputShapes?: OnnxDim[][]
  outputShapes?: OnnxDim[][]
  dimmed?: boolean
  excluded?: boolean
  isModified?: boolean
}

export interface OnnxEdge {
  id: string
  source: string
  target: string
  label?: string
  shape?: OnnxDim[]
}

export interface OnnxGraph {
  nodes: OnnxNode[]
  edges: OnnxEdge[]
  modelName: string
  totalParams: number
  totalSizeMB: number
  graphInputs?: ParsedValueInfo[]
  metadata?: ModelMetadata
  format?: 'onnx' | 'tflite'
}
