import type { OnnxDim, ParsedValueInfo, ModelMetadata } from './onnxProtoParser'

export type { OnnxDim, ParsedValueInfo, ModelMetadata }

export interface TensorMetadata {
  shape?: OnnxDim[]
  elemType?: number
  // Only set for a small initializer (see SMALL_TENSOR_MAX_ELEMENTS) -- lets
  // the inspector show and edit a constant like a Reshape target shape or a
  // scalar bias inline, without attempting the same for a weight tensor.
  values?: number[]
}

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
  inputMetadata?: TensorMetadata[]
  outputMetadata?: TensorMetadata[]
  dimmed?: boolean
  excluded?: boolean
  isModified?: boolean
  diffStatus?: 'deleted'
  // Set only for a node the user manually placed on the canvas (addCustomNode).
  // GraphCanvas's dagre layout treats a node with this set as fixed -- it's
  // excluded from auto-layout entirely rather than repositioned every render.
  position?: { x: number; y: number }
}

export interface OnnxEdge {
  id: string
  source: string
  target: string
  label?: string
  shape?: OnnxDim[]
  diffStatus?: 'changed' | 'removed'
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
