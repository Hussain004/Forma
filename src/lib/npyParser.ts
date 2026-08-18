// Minimal reader for NumPy .npy (single array) and .npz (zip of named .npy
// entries) files, used by behavioral validation to load reusable test inputs
// instead of only ever generating synthetic ones. See the NPY format spec:
// https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
//
// .npz support only covers what `numpy.savez` actually produces: a plain
// (STORED) or DEFLATE-compressed zip of `<name>.npy` entries, no zip64, no
// encryption, no nested directories. DEFLATE is decoded with the browser's
// native DecompressionStream rather than a bundled zip library.

export interface ParsedArray {
  data: Float32Array
  shape: number[]
}

const NPY_MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59] // \x93NUMPY

export function isNpyBuffer(bytes: Uint8Array): boolean {
  return bytes.length >= 8 && NPY_MAGIC.every((b, i) => bytes[i] === b)
}

export function isNpzBuffer(bytes: Uint8Array): boolean {
  // Local file header signature "PK\x03\x04" -- every zip starts with one.
  return bytes.length >= 4 && bytes[0] === 0x50 && bytes[1] === 0x4b && bytes[2] === 0x03 && bytes[3] === 0x04
}

type DescrReader = (view: DataView, offset: number) => number

const DESCR_READERS: Record<string, { size: number; read: DescrReader }> = {
  '<f4': { size: 4, read: (v, o) => v.getFloat32(o, true) },
  '<f8': { size: 8, read: (v, o) => v.getFloat64(o, true) },
  '<i4': { size: 4, read: (v, o) => v.getInt32(o, true) },
  '<i8': { size: 8, read: (v, o) => Number(v.getBigInt64(o, true)) },
  '<u4': { size: 4, read: (v, o) => v.getUint32(o, true) },
  '<u8': { size: 8, read: (v, o) => Number(v.getBigUint64(o, true)) },
  '<i2': { size: 2, read: (v, o) => v.getInt16(o, true) },
  '<u2': { size: 2, read: (v, o) => v.getUint16(o, true) },
  '|i1': { size: 1, read: (v, o) => v.getInt8(o) },
  '<i1': { size: 1, read: (v, o) => v.getInt8(o) },
  '|u1': { size: 1, read: (v, o) => v.getUint8(o) },
  '<b1': { size: 1, read: (v, o) => v.getUint8(o) },
  '|b1': { size: 1, read: (v, o) => v.getUint8(o) },
}

export function parseNpy(buffer: ArrayBuffer): ParsedArray {
  const bytes = new Uint8Array(buffer)
  if (!isNpyBuffer(bytes)) throw new Error('Not a valid .npy file (bad magic bytes)')
  const view = new DataView(buffer)
  const majorVersion = bytes[6]
  const headerStart = majorVersion === 1 ? 10 : 12
  const headerLen = majorVersion === 1 ? view.getUint16(8, true) : view.getUint32(8, true)
  const header = new TextDecoder('latin1').decode(bytes.slice(headerStart, headerStart + headerLen))

  const descrMatch = /'descr'\s*:\s*'([^']+)'/.exec(header)
  const shapeMatch = /'shape'\s*:\s*\(([^)]*)\)/.exec(header)
  const fortranMatch = /'fortran_order'\s*:\s*(True|False)/.exec(header)
  if (!descrMatch || !shapeMatch) throw new Error('Could not parse the .npy header')
  if (fortranMatch?.[1] === 'True') throw new Error('Fortran-ordered .npy arrays are not supported')

  const descr = DESCR_READERS[descrMatch[1]]
  if (!descr) throw new Error(`Unsupported .npy dtype: ${descrMatch[1]}`)

  const shape = shapeMatch[1].split(',').map((s) => s.trim()).filter(Boolean).map(Number)
  const finalShape = shape.length > 0 ? shape : [1]
  const count = finalShape.reduce((a, b) => a * b, 1)

  const dataStart = headerStart + headerLen
  const data = new Float32Array(count)
  for (let i = 0; i < count; i++) data[i] = descr.read(view, dataStart + i * descr.size)
  return { data, shape: finalShape }
}

// `bytes` is always a fresh copy from Uint8Array#slice (byteOffset 0, exact
// length), so its `.buffer` is safe to hand to Blob as a plain ArrayBuffer.
async function inflate(bytes: Uint8Array): Promise<ArrayBuffer> {
  const stream = new Blob([bytes.buffer as ArrayBuffer]).stream().pipeThrough(new DecompressionStream('deflate-raw'))
  return new Response(stream).arrayBuffer()
}

// Reads only the central directory (the authoritative entry list) plus each
// entry's local header to locate its data -- no need to walk the file linearly.
export async function parseNpz(buffer: ArrayBuffer): Promise<Record<string, ParsedArray>> {
  const bytes = new Uint8Array(buffer)
  const view = new DataView(buffer)

  const EOCD_SIGNATURE = 0x06054b50
  const scanFloor = Math.max(0, bytes.length - 66000)
  let eocdOffset = -1
  for (let i = bytes.length - 22; i >= scanFloor; i--) {
    if (view.getUint32(i, true) === EOCD_SIGNATURE) { eocdOffset = i; break }
  }
  if (eocdOffset === -1) throw new Error('Not a valid .npz file (no end-of-central-directory record found)')

  const entryCount = view.getUint16(eocdOffset + 10, true)
  let pos = view.getUint32(eocdOffset + 16, true)

  const CENTRAL_DIR_SIGNATURE = 0x02014b50
  const result: Record<string, ParsedArray> = {}
  for (let e = 0; e < entryCount; e++) {
    if (view.getUint32(pos, true) !== CENTRAL_DIR_SIGNATURE) throw new Error('Malformed .npz central directory')
    const method = view.getUint16(pos + 10, true)
    const compressedSize = view.getUint32(pos + 20, true)
    const nameLen = view.getUint16(pos + 28, true)
    const extraLen = view.getUint16(pos + 30, true)
    const commentLen = view.getUint16(pos + 32, true)
    const localHeaderOffset = view.getUint32(pos + 42, true)
    const name = new TextDecoder().decode(bytes.slice(pos + 46, pos + 46 + nameLen))
    pos += 46 + nameLen + extraLen + commentLen

    if (!name.endsWith('.npy')) continue

    const localNameLen = view.getUint16(localHeaderOffset + 26, true)
    const localExtraLen = view.getUint16(localHeaderOffset + 28, true)
    const dataStart = localHeaderOffset + 30 + localNameLen + localExtraLen
    const compressed = bytes.slice(dataStart, dataStart + compressedSize)

    let npyBuffer: ArrayBuffer
    if (method === 0) npyBuffer = compressed.buffer.slice(compressed.byteOffset, compressed.byteOffset + compressed.byteLength) as ArrayBuffer
    else if (method === 8) npyBuffer = await inflate(compressed)
    else throw new Error(`Unsupported .npz compression method (${method}) for entry "${name}"`)

    result[name.slice(0, -4)] = parseNpy(npyBuffer)
  }
  return result
}
