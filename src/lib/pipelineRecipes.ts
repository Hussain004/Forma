// Curated menu for v2.4's guided pipeline-recipe insertion: a small set of
// common preprocessing ops (attached right after a graph input) and
// postprocessing ops (attached right before a graph output), each a single
// real ONNX node with sensible default attributes the user can fine-tune
// afterward through the normal attribute editor. See graphUtils.insertRecipeNode
// and onnxProtoWriter's 'insertRecipe' StructuralOp for how a recipe here
// becomes a wired node at both the canvas and export layers.
//
// Deliberately out of scope: detection postprocessing (NonMaxSuppression).
// NMS needs two pre-existing internal tensors (boxes and scores) as inputs,
// not one boundary tensor -- it doesn't fit this insert-at-a-boundary model,
// which is why it's absent below despite being on the v2.4 roadmap line.

export interface RecipeAttr {
  name: string
  kind: 'I' | 'F' | 'S' | 'INTS' | 'FLOATS'
  value: string | number
}

// A slot appended to the node's inputs after the boundary tensor. 'empty'
// encodes ONNX's own convention for an omitted optional input (used by
// Resize's unused `roi`); 'const' creates a small new initializer.
export interface RecipeInputSlot {
  kind: 'empty' | 'const'
  elemType?: number
  dims?: number[]
  values?: number[]
}

export interface PipelineRecipe {
  id: string
  label: string
  category: 'preprocess' | 'postprocess'
  opType: string
  description: string
  attrs: RecipeAttr[]
  extraInputs?: RecipeInputSlot[]
  // Additional outputs beyond the primary one, auto-promoted as extra graph
  // outputs (e.g. TopK's Indices alongside its Values). Declared unranked but
  // WITH the correct dtype -- TopK's Indices is always int64 regardless of
  // the primary output's dtype, so this can't just inherit the primary's
  // elem type the way the shape is safely dropped. Index i pairs with the
  // i-th extra output; a missing entry defaults to float32.
  extraOutputCount?: number
  extraOutputElemTypes?: number[]
  // Unlike Top-K (schema-adaptable across the opset boundary, see
  // resolveRecipe), Resize genuinely did not exist as an op before opset 10
  // -- there's no legacy encoding to fall back to. Set on a recipe that has
  // no pre-opset-10 equivalent, so it can be hidden for an older model
  // instead of failing opaquely at export time.
  minOpset?: number
}

export const PIPELINE_RECIPES: PipelineRecipe[] = [
  {
    id: 'cast-float32',
    label: 'Cast to float32',
    category: 'preprocess',
    opType: 'Cast',
    description: 'Converts the input tensor to float32 before the rest of the graph runs.',
    attrs: [{ name: 'to', kind: 'I', value: 1 }],
  },
  {
    id: 'resize-2x-nearest',
    label: 'Resize 2x (nearest)',
    category: 'preprocess',
    opType: 'Resize',
    description: 'Upsamples the input 2x on every dimension using nearest-neighbor interpolation.',
    attrs: [{ name: 'mode', kind: 'S', value: 'nearest' }],
    extraInputs: [
      { kind: 'empty' }, // roi: unused unless coordinate_transformation_mode is tf_crop_and_resize
      { kind: 'const', elemType: 1, dims: [4], values: [1, 1, 2, 2] }, // scales
    ],
    minOpset: 10,
  },
  {
    id: 'transpose-nhwc-nchw',
    label: 'Transpose NHWC to NCHW',
    category: 'preprocess',
    opType: 'Transpose',
    description: 'Reorders a 4D input from channels-last to channels-first.',
    attrs: [{ name: 'perm', kind: 'INTS', value: '[0, 3, 1, 2]' }],
  },
  {
    id: 'l2-normalize',
    label: 'L2 Normalize',
    category: 'preprocess',
    opType: 'LpNormalization',
    description: 'Normalizes the input to unit L2 norm along an axis.',
    attrs: [
      { name: 'axis', kind: 'I', value: 1 },
      { name: 'p', kind: 'I', value: 2 },
    ],
  },
  {
    id: 'softmax',
    label: 'Softmax',
    category: 'postprocess',
    opType: 'Softmax',
    description: 'Converts raw output scores to probabilities.',
    attrs: [{ name: 'axis', kind: 'I', value: -1 }],
  },
  {
    id: 'sigmoid',
    label: 'Sigmoid',
    category: 'postprocess',
    opType: 'Sigmoid',
    description: 'Squashes each output score to (0, 1) independently.',
    attrs: [],
  },
  {
    id: 'top-k-5',
    label: 'Top-K (5)',
    category: 'postprocess',
    opType: 'TopK',
    description: 'Keeps the 5 highest-scoring values and their indices; adds a second graph output for the indices.',
    attrs: [
      { name: 'axis', kind: 'I', value: -1 },
      { name: 'largest', kind: 'I', value: 1 },
      { name: 'sorted', kind: 'I', value: 1 },
    ],
    extraInputs: [{ kind: 'const', elemType: 7, dims: [1], values: [5] }], // K
    extraOutputCount: 1,
    extraOutputElemTypes: [7], // Indices: always int64
  },
  {
    id: 'transpose-nchw-nhwc',
    label: 'Transpose NCHW to NHWC',
    category: 'postprocess',
    opType: 'Transpose',
    description: 'Reorders a 4D output from channels-first to channels-last.',
    attrs: [{ name: 'perm', kind: 'INTS', value: '[0, 2, 3, 1]' }],
  },
]

// Most of this catalog is schema-stable across ONNX opset versions (Cast,
// Transpose, Softmax, Sigmoid, and LpNormalization never changed shape), but
// TopK moved its `k` from a required attribute (opset 1-9) to a tensor input
// (opset 10+, alongside axis/largest/sorted attributes that didn't exist
// before). Inserting the modern 2-input encoding into an older model is
// rejected by onnxruntime with a schema-mismatch error -- resolveRecipe
// adapts to the loaded model's own declared opset instead, so callers should
// always insert the result of this rather than a catalog entry directly.
export function resolveRecipe(recipe: PipelineRecipe, opsetVersion: number | undefined): PipelineRecipe {
  if (recipe.id === 'top-k-5' && opsetVersion !== undefined && opsetVersion < 10) {
    return {
      ...recipe,
      attrs: [
        { name: 'axis', kind: 'I', value: -1 },
        { name: 'k', kind: 'I', value: 5 },
      ],
      extraInputs: [],
      extraOutputCount: 1,
    }
  }
  return recipe
}
