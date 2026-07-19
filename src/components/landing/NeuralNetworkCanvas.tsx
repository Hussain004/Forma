import { useEffect, useRef } from 'react'
import * as THREE from 'three'

const NODE_COUNT = 52
const PROXIMITY_RADIUS = 3.0
const DRIFT_SPEED = 0.08
const PROXIMITY_PULSE_SPEED = 4.0

// Seed-based pseudo-random for deterministic layout
function seededRandom(seed: number) {
  let s = seed
  return () => {
    s = (s * 16807 + 0) % 2147483647
    return (s - 1) / 2147483646
  }
}

// Generate edges connecting nearby nodes in the initial layout
function generateEdges(positions: Float32Array, nodeCount: number): [number, number][] {
  const edges: [number, number][] = []
  const maxEdges = nodeCount * 2
  const rand = seededRandom(42)
  for (let i = 0; i < nodeCount && edges.length < maxEdges; i++) {
    const ix = positions[i * 3]
    const iy = positions[i * 3 + 1]
    const iz = positions[i * 3 + 2]
    let closest = -1
    let closestDist = Infinity
    for (let j = 0; j < nodeCount; j++) {
      if (j === i) continue
      const alreadyConnected = edges.some(
        (e) => (e[0] === i && e[1] === j) || (e[0] === j && e[1] === i),
      )
      if (alreadyConnected) continue
      const dx = positions[j * 3] - ix
      const dy = positions[j * 3 + 1] - iy
      const dz = positions[j * 3 + 2] - iz
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (dist < closestDist) {
        closestDist = dist
        closest = j
      }
    }
    if (closest >= 0 && rand() < 0.6) {
      edges.push([i, closest])
    }
  }
  return edges
}

export function NeuralNetworkCanvas() {
  const containerRef = useRef<HTMLDivElement>(null)
  const mouseRef = useRef({ x: 9999, y: 9999 })
  const frameRef = useRef<number>(0)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100)
    camera.position.set(0, 0, 18)
    camera.lookAt(0, 0, 0)

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
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x000000, 0)
    container.appendChild(renderer.domElement)

    // Generate node positions
    const rand = seededRandom(123)
    const positions = new Float32Array(NODE_COUNT * 3)
    const basePositions = new Float32Array(NODE_COUNT * 3)
    const phases = new Float32Array(NODE_COUNT * 3)
    for (let i = 0; i < NODE_COUNT; i++) {
      const x = (rand() - 0.5) * 22
      const y = (rand() - 0.5) * 14
      const z = (rand() - 0.5) * 8
      positions[i * 3] = x
      positions[i * 3 + 1] = y
      positions[i * 3 + 2] = z
      basePositions[i * 3] = x
      basePositions[i * 3 + 1] = y
      basePositions[i * 3 + 2] = z
      phases[i * 3] = rand() * Math.PI * 2
      phases[i * 3 + 1] = rand() * Math.PI * 2
      phases[i * 3 + 2] = rand() * Math.PI * 2
    }

    const edges = generateEdges(positions, NODE_COUNT)

    // Node instanced mesh -- small octahedrons for a technical look
    const nodeGeo = new THREE.OctahedronGeometry(0.08, 0)
    const nodeMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true })
    const nodeMesh = new THREE.InstancedMesh(nodeGeo, nodeMat, NODE_COUNT)
    nodeMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    scene.add(nodeMesh)

    // Node glow instances (slightly larger, more transparent)
    const glowGeo = new THREE.OctahedronGeometry(0.16, 0)
    const glowMat = new THREE.MeshBasicMaterial({ color: 0xffb000, transparent: true, opacity: 0 })
    const glowMesh = new THREE.InstancedMesh(glowGeo, glowMat, NODE_COUNT)
    glowMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    scene.add(glowMesh)

    // Edge lines
    const edgePositions = new Float32Array(edges.length * 6)
    const edgeColors = new Float32Array(edges.length * 6)
    const edgeGeometry = new THREE.BufferGeometry()
    edgeGeometry.setAttribute('position', new THREE.BufferAttribute(edgePositions, 3))
    edgeGeometry.setAttribute('color', new THREE.BufferAttribute(edgeColors, 3))
    const edgeMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.4,
    })
    const edgeLines = new THREE.LineSegments(edgeGeometry, edgeMaterial)
    scene.add(edgeLines)

    // Dummy for instance matrix updates
    const dummy = new THREE.Object3D()
    const tmpColor = new THREE.Color()

    // Mouse tracking
    const onMouseMove = (e: MouseEvent) => {
      mouseRef.current.x = (e.clientX / window.innerWidth) * 2 - 1
      mouseRef.current.y = -(e.clientY / window.innerHeight) * 2 + 1
    }
    window.addEventListener('mousemove', onMouseMove)

    // Resize
    const onResize = () => {
      const w = container.clientWidth
      const h = container.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    onResize()
    const resizeObserver = new ResizeObserver(onResize)
    resizeObserver.observe(container)

    // Animation loop
    const clock = new THREE.Clock()
    const animate = () => {
      const elapsed = clock.getElapsedTime()
      frameRef.current = requestAnimationFrame(animate)

      // Project mouse to 3D plane at z=0
      const mouse3D = new THREE.Vector3(mouseRef.current.x * 14, mouseRef.current.y * 8, 0)

      // Update node positions (drift)
      for (let i = 0; i < NODE_COUNT; i++) {
        const i3 = i * 3
        positions[i3] = basePositions[i3] + Math.sin(elapsed * DRIFT_SPEED + phases[i3]) * 0.4
        positions[i3 + 1] = basePositions[i3 + 1] + Math.cos(elapsed * DRIFT_SPEED * 0.7 + phases[i3 + 1]) * 0.3
        positions[i3 + 2] = basePositions[i3 + 2] + Math.sin(elapsed * DRIFT_SPEED * 0.5 + phases[i3 + 2]) * 0.2

        // Distance to mouse
        const dx = positions[i3] - mouse3D.x
        const dy = positions[i3 + 1] - mouse3D.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        const proximity = Math.max(0, 1 - dist / PROXIMITY_RADIUS)

        // Node transform
        dummy.position.set(positions[i3], positions[i3 + 1], positions[i3 + 2])
        const scale = 1 + proximity * 1.5
        dummy.scale.setScalar(scale)
        dummy.updateMatrix()
        nodeMesh.setMatrixAt(i, dummy.matrix)

        // Node color: idle dim white, proximity amber
        const brightness = 0.15 + proximity * 0.85
        tmpColor.setRGB(brightness, brightness, brightness)
        if (proximity > 0.1) {
          tmpColor.lerp(new THREE.Color(0xffb000), proximity * 0.7)
        }
        nodeMesh.setColorAt(i, tmpColor)

        // Glow
        dummy.scale.setScalar(1 + proximity * 3)
        dummy.updateMatrix()
        glowMesh.setMatrixAt(i, dummy.matrix)
        glowMat.opacity = 0.15
        glowMesh.setColorAt(i, new THREE.Color(0xffb000))
      }

      nodeMesh.instanceMatrix.needsUpdate = true
      nodeMesh.instanceColor!.needsUpdate = true
      glowMesh.instanceMatrix.needsUpdate = true

      // Update edge positions and colors
      const posAttr = edgeGeometry.getAttribute('position') as THREE.BufferAttribute
      const colAttr = edgeGeometry.getAttribute('color') as THREE.BufferAttribute
      for (let e = 0; e < edges.length; e++) {
        const [a, b] = edges[e]
        const a3 = a * 3
        const b3 = b * 3
        posAttr.array[e * 6] = positions[a3]
        posAttr.array[e * 6 + 1] = positions[a3 + 1]
        posAttr.array[e * 6 + 2] = positions[a3 + 2]
        posAttr.array[e * 6 + 3] = positions[b3]
        posAttr.array[e * 6 + 4] = positions[b3 + 1]
        posAttr.array[e * 6 + 5] = positions[b3 + 2]

        // Edge brightness based on proximity of either endpoint
        const dax = positions[a3] - mouse3D.x
        const day = positions[a3 + 1] - mouse3D.y
        const proxA = Math.max(0, 1 - Math.sqrt(dax * dax + day * day) / PROXIMITY_RADIUS)
        const dbx = positions[b3] - mouse3D.x
        const dby = positions[b3 + 1] - mouse3D.y
        const proxB = Math.max(0, 1 - Math.sqrt(dbx * dbx + dby * dby) / PROXIMITY_RADIUS)
        const edgeProximity = Math.max(proxA, proxB)

        const r = 0.08 + edgeProximity * 0.92
        const g = 0.08 + edgeProximity * 0.42
        const bl = 0.1 + edgeProximity * (-0.06)
        colAttr.array[e * 6] = r
        colAttr.array[e * 6 + 1] = g
        colAttr.array[e * 6 + 2] = bl
        colAttr.array[e * 6 + 3] = r
        colAttr.array[e * 6 + 4] = g
        colAttr.array[e * 6 + 5] = bl
      }
      posAttr.needsUpdate = true
      colAttr.needsUpdate = true

      // Pulse edge opacity with proximity
      edgeMaterial.opacity = 0.12 + Math.sin(elapsed * PROXIMITY_PULSE_SPEED) * 0.03

      renderer.render(scene, camera)
    }
    frameRef.current = requestAnimationFrame(animate)

    return () => {
      cancelAnimationFrame(frameRef.current)
      window.removeEventListener('mousemove', onMouseMove)
      resizeObserver.disconnect()
      renderer.dispose()
      nodeGeo.dispose()
      nodeMat.dispose()
      glowGeo.dispose()
      glowMat.dispose()
      edgeGeometry.dispose()
      edgeMaterial.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  return <div ref={containerRef} className="landing-canvas" />
}
