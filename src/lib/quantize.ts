export function estimateInt8Size(elemCount: number): number {
  return (elemCount * 1) / (1024 * 1024)
}

export function compressionRatio(originalSizeMB: number, elemCount: number): number {
  const int8SizeMB = estimateInt8Size(elemCount)
  return int8SizeMB > 0 ? originalSizeMB / int8SizeMB : 0
}
