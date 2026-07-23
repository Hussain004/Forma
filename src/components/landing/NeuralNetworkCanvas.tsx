import { useEffect, useRef } from 'react'
import * as THREE from 'three'

const LAYER_COUNTS = [4, 7, 9, 9, 7, 4]
const NODE_COUNT = LAYER_COUNTS.reduce((total, count) => total + count, 0)
const PROXIMITY_RADIUS = 2.6

interface NetworkTopology {
  positions: Float32Array
  edges: Array<[number, number]>
}

function seededRandom(seed: number) {
  let state = seed
  return () => {
    state = (state * 16807) % 2147483647
    return (state - 1) / 2147483646
  }
}

function createTopology(): NetworkTopology {
  const random = seededRandom(1701)
  const positions = new Float32Array(NODE_COUNT * 3)
  const layerStarts: number[] = []
  let nodeIndex = 0

  LAYER_COUNTS.forEach((count, layerIndex) => {
    layerStarts.push(nodeIndex)
    const x = -11 + layerIndex * (22 / (LAYER_COUNTS.length - 1))
    for (let index = 0; index < count; index += 1) {
      const normalizedY = count === 1 ? 0 : index / (count - 1)
      positions[nodeIndex * 3] = x + (random() - 0.5) * 0.45
      positions[nodeIndex * 3 + 1] = 5.4 - normalizedY * 10.8 + (random() - 0.5) * 0.42
      positions[nodeIndex * 3 + 2] = (random() - 0.5) * 2.4
      nodeIndex += 1
    }
  })

  const edgeKeys = new Set<string>()
  const edges: Array<[number, number]> = []
  const addEdge = (source: number, target: number) => {
    const key = `${source}:${target}`
    if (edgeKeys.has(key)) return
    edgeKeys.add(key)
    edges.push([source, target])
  }

  for (let layer = 0; layer < LAYER_COUNTS.length - 1; layer += 1) {
    const sourceStart = layerStarts[layer]
    const targetStart = layerStarts[layer + 1]
    const sourceCount = LAYER_COUNTS[layer]
    const targetCount = LAYER_COUNTS[layer + 1]

    for (let sourceOffset = 0; sourceOffset < sourceCount; sourceOffset += 1) {
      const source = sourceStart + sourceOffset
      const sourceY = positions[source * 3 + 1]
      const targets = Array.from({ length: targetCount }, (_, offset) => targetStart + offset)
        .sort((a, b) => Math.abs(positions[a * 3 + 1] - sourceY) - Math.abs(positions[b * 3 + 1] - sourceY))
      addEdge(source, targets[0])
      if ((sourceOffset + layer) % 2 === 0 && targets[1] !== undefined) addEdge(source, targets[1])
    }

    for (let targetOffset = 0; targetOffset < targetCount; targetOffset += 1) {
      const target = targetStart + targetOffset
      const hasInput = edges.some(([, edgeTarget]) => edgeTarget === target)
      if (hasInput) continue
      let nearestSource = sourceStart
      for (let offset = 1; offset < sourceCount; offset += 1) {
        const candidate = sourceStart + offset
        const nearestDistance = Math.abs(positions[nearestSource * 3 + 1] - positions[target * 3 + 1])
        const candidateDistance = Math.abs(positions[candidate * 3 + 1] - positions[target * 3 + 1])
        if (candidateDistance < nearestDistance) nearestSource = candidate
      }
      addEdge(nearestSource, target)
    }
  }

  return { positions, edges }
}

export function NeuralNetworkCanvas() {
  const containerRef = useRef<HTMLDivElement>(null)
  const frameRef = useRef(0)

  useEffect(() => {
    const container = containerRef.current
    if (typeof window.WebGLRenderingContext === 'undefined') return
    if (!container) return

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 80)
    camera.position.set(0, 0, 18)

    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: 'low-power',
      })
    } catch {
      return
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75))
    renderer.setClearColor(0x12161a, 0)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    container.appendChild(renderer.domElement)

    const { positions: basePositions, edges } = createTopology()
    const positions = basePositions.slice()
    const proximityByNode = new Float32Array(NODE_COUNT)
    const phases = new Float32Array(NODE_COUNT)
    const random = seededRandom(208)
    for (let index = 0; index < NODE_COUNT; index += 1) phases[index] = random() * Math.PI * 2

    const nodeGeometry = new THREE.BoxGeometry(0.16, 0.16, 0.16)
    const nodeMaterial = new THREE.MeshBasicMaterial({ vertexColors: false })
    const nodes = new THREE.InstancedMesh(nodeGeometry, nodeMaterial, NODE_COUNT)
    nodes.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    scene.add(nodes)

    const edgeGeometry = new THREE.BufferGeometry()
    edgeGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(edges.length * 6), 3))
    edgeGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(edges.length * 6), 3))
    const edgeMaterial = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.5 })
    const edgeLines = new THREE.LineSegments(edgeGeometry, edgeMaterial)
    scene.add(edgeLines)

    const pointer = new THREE.Vector2(10, 10)
    const pointerWorld = new THREE.Vector3(100, 100, 0)
    const dummy = new THREE.Object3D()
    const idleNodeColor = new THREE.Color(0x5d6770)
    const greenNodeColor = new THREE.Color(0x4a5d23)
    const activeNodeColor = new THREE.Color(0xffb000)
    const idleEdgeColor = new THREE.Color(0x283138)
    const activeEdgeColor = new THREE.Color(0xffb000)
    const workingColor = new THREE.Color()
    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false

    const updatePointerWorld = () => {
      const projected = new THREE.Vector3(pointer.x, pointer.y, 0.5).unproject(camera)
      const direction = projected.sub(camera.position).normalize()
      const distance = -camera.position.z / direction.z
      pointerWorld.copy(camera.position).add(direction.multiplyScalar(distance))
    }

    const onPointerMove = (event: PointerEvent) => {
      const rect = container.getBoundingClientRect()
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
      updatePointerWorld()
    }

    const onPointerLeave = () => {
      pointer.set(10, 10)
      pointerWorld.set(100, 100, 0)
    }

    window.addEventListener('pointermove', onPointerMove)
    document.documentElement.addEventListener('pointerleave', onPointerLeave)

    const onResize = () => {
      const width = Math.max(1, container.clientWidth)
      const height = Math.max(1, container.clientHeight)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height, false)
      updatePointerWorld()
    }
    onResize()
    const resizeObserver = new ResizeObserver(onResize)
    resizeObserver.observe(container)

    const renderFrame = (timestamp: number) => {
      const elapsed = timestamp / 1000
      const driftAmount = reducedMotion ? 0 : 0.11

      camera.position.x += ((pointer.x < 2 ? pointer.x * 0.26 : 0) - camera.position.x) * 0.018
      camera.position.y += ((pointer.y < 2 ? pointer.y * 0.18 : 0) - camera.position.y) * 0.018
      camera.lookAt(0, 0, 0)
      updatePointerWorld()

      for (let index = 0; index < NODE_COUNT; index += 1) {
        const offset = index * 3
        let x = basePositions[offset]
        let y = basePositions[offset + 1] + Math.sin(elapsed * 0.22 + phases[index]) * driftAmount
        const z = basePositions[offset + 2]
        const dx = x - pointerWorld.x
        const dy = y - pointerWorld.y
        const distance = Math.hypot(dx, dy)
        const proximity = Math.max(0, 1 - distance / PROXIMITY_RADIUS)
        proximityByNode[index] = proximity

        if (proximity > 0 && distance > 0.001) {
          x += (dx / distance) * proximity * 0.18
          y += (dy / distance) * proximity * 0.18
        }
        positions[offset] = x
        positions[offset + 1] = y
        positions[offset + 2] = z

        dummy.position.set(x, y, z)
        const scale = 1 + proximity * 1.8
        dummy.scale.set(1.5 * scale, scale, scale)
        dummy.rotation.z = Math.sin(phases[index]) * 0.35
        dummy.updateMatrix()
        nodes.setMatrixAt(index, dummy.matrix)

        workingColor.copy(index % 7 === 0 ? greenNodeColor : idleNodeColor)
        workingColor.lerp(activeNodeColor, proximity)
        nodes.setColorAt(index, workingColor)
      }
      nodes.instanceMatrix.needsUpdate = true
      if (nodes.instanceColor) nodes.instanceColor.needsUpdate = true

      const positionAttribute = edgeGeometry.getAttribute('position') as THREE.BufferAttribute
      const colorAttribute = edgeGeometry.getAttribute('color') as THREE.BufferAttribute
      for (let edgeIndex = 0; edgeIndex < edges.length; edgeIndex += 1) {
        const [source, target] = edges[edgeIndex]
        const sourceOffset = source * 3
        const targetOffset = target * 3
        const attributeOffset = edgeIndex * 6
        positionAttribute.array[attributeOffset] = positions[sourceOffset]
        positionAttribute.array[attributeOffset + 1] = positions[sourceOffset + 1]
        positionAttribute.array[attributeOffset + 2] = positions[sourceOffset + 2]
        positionAttribute.array[attributeOffset + 3] = positions[targetOffset]
        positionAttribute.array[attributeOffset + 4] = positions[targetOffset + 1]
        positionAttribute.array[attributeOffset + 5] = positions[targetOffset + 2]

        const proximity = Math.max(proximityByNode[source], proximityByNode[target])
        workingColor.copy(idleEdgeColor).lerp(activeEdgeColor, proximity * 0.9)
        for (let vertex = 0; vertex < 2; vertex += 1) {
          const colorOffset = attributeOffset + vertex * 3
          colorAttribute.array[colorOffset] = workingColor.r
          colorAttribute.array[colorOffset + 1] = workingColor.g
          colorAttribute.array[colorOffset + 2] = workingColor.b
        }
      }
      positionAttribute.needsUpdate = true
      colorAttribute.needsUpdate = true

      renderer.render(scene, camera)
      frameRef.current = requestAnimationFrame(renderFrame)
    }

    frameRef.current = requestAnimationFrame(renderFrame)

    return () => {
      cancelAnimationFrame(frameRef.current)
      window.removeEventListener('pointermove', onPointerMove)
      document.documentElement.removeEventListener('pointerleave', onPointerLeave)
      resizeObserver.disconnect()
      nodeGeometry.dispose()
      nodeMaterial.dispose()
      edgeGeometry.dispose()
      edgeMaterial.dispose()
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [])

  return <div ref={containerRef} className="landing-canvas" aria-hidden="true" />
}
