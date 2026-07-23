import { NeuralNetworkCanvas } from './NeuralNetworkCanvas'
import { HeroSection } from './HeroSection'
import { DropZone } from './DropZone'
import { FeatureGrid } from './FeatureGrid'
import { StatusBar } from './StatusBar'

interface LandingPageProps {
  onModelLoaded: (buffer: ArrayBuffer, filename: string) => void
  status: 'idle' | 'loading' | 'ready' | 'error'
  error?: string | null
  progressLabel?: string | null
  progressPercent?: number | null
}

export function LandingPage({ onModelLoaded, status, error, progressLabel, progressPercent }: LandingPageProps) {
  return (
    <div className="landing-root">
      <NeuralNetworkCanvas />
      <div className="landing-grid-frame" aria-hidden="true" />
      <StatusBar />
      <main className="landing-content">
        <HeroSection />
        <DropZone
          onModelLoaded={onModelLoaded}
          status={status}
          error={error}
          progressLabel={progressLabel}
          progressPercent={progressPercent}
        />
        <FeatureGrid />
      </main>
    </div>
  )
}
