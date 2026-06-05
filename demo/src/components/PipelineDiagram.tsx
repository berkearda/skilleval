import { useEffect, useState } from 'react'
import { ArrowDown, ArrowRight } from 'lucide-react'
import { getMasteryColor } from '@/lib/colors'

// Native redraw of the paper's pipeline figure in the site's design system:
// theme-aware, crisp at any size, and the text stays selectable.

function useDarkMode(): boolean {
  const [dark, setDark] = useState<boolean>(
    typeof document !== 'undefined' &&
      document.documentElement.classList.contains('dark')
  )
  useEffect(() => {
    const root = document.documentElement
    const obs = new MutationObserver(() => {
      setDark(root.classList.contains('dark'))
    })
    obs.observe(root, { attributes: true, attributeFilter: ['class'] })
    return () => obs.disconnect()
  }, [])
  return dark
}

const kicker =
  'text-[11px] font-medium uppercase tracking-wider text-muted-foreground'

function StageCard({
  step,
  title,
  children,
  foot,
}: {
  step: string
  title: string
  children: React.ReactNode
  foot: string
}) {
  return (
    <div className="flex min-w-0 flex-1 flex-col rounded-lg border border-border bg-surface p-4">
      <div className="flex items-center gap-2">
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-brand text-[11px] font-semibold text-white">
          {step}
        </span>
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
      </div>
      <div className="mt-3 flex flex-1 items-center justify-center">
        {children}
      </div>
      <p className="mt-3 text-[12px] leading-4 text-muted-foreground">{foot}</p>
    </div>
  )
}

function Connector() {
  return (
    <div className="flex items-center justify-center self-center text-muted-foreground">
      <ArrowRight className="hidden h-4 w-4 shrink-0 lg:block" />
      <ArrowDown className="h-4 w-4 shrink-0 lg:hidden" />
    </div>
  )
}

// Stage 1: an item with its skill tags.
function ItemVisual() {
  return (
    <div className="w-full max-w-[210px] rounded-md border border-border bg-background p-2.5">
      <div className={kicker}>Item</div>
      <p className="mt-1 font-mono text-[11px] leading-4 text-foreground">
        Solve x² − 5x + 6 = 0 for both roots.
      </p>
      <div className="mt-2 flex flex-wrap gap-1">
        <span className="rounded-full bg-brand/10 px-1.5 py-0.5 text-[10px] font-medium text-brand">
          quadratic formula
        </span>
        <span className="rounded-full bg-brand/10 px-1.5 py-0.5 text-[10px] font-medium text-brand">
          symbolic algebra
        </span>
      </div>
    </div>
  )
}

// Stage 2: the binary response matrix (filled = correct, faint = incorrect).
const MATRIX: number[][] = [
  [1, 0, 1, 1, 0, 1, 1, 0],
  [1, 1, 0, 1, 1, 1, 0, 1],
  [0, 1, 1, 0, 1, 0, 1, 1],
  [1, 0, 1, 1, 0, 1, 1, 1],
  [0, 1, 0, 1, 1, 0, 1, 0],
  [1, 1, 1, 0, 1, 1, 0, 1],
]

function MatrixVisual() {
  return (
    <div className="flex items-center gap-2">
      <div
        className="grid gap-[3px]"
        style={{ gridTemplateColumns: 'repeat(8, 10px)' }}
        aria-hidden
      >
        {MATRIX.flatMap((row, i) =>
          row.map((v, j) => (
            <span
              key={`${i}-${j}`}
              className="h-[10px] w-[10px] rounded-[2px]"
              style={{
                backgroundColor: v
                  ? 'hsl(var(--brand) / 0.85)'
                  : 'hsl(var(--muted-foreground) / 0.22)',
              }}
            />
          ))
        )}
      </div>
      <div className="text-[10px] leading-4 text-muted-foreground">
        <div>models ↓</div>
        <div>items →</div>
      </div>
    </div>
  )
}

// Stage 3: the model.
function ModelVisual() {
  return (
    <div className="text-center">
      <code className="whitespace-nowrap rounded-md border border-border bg-background px-2 py-1.5 font-mono text-[10px] text-foreground">
        p(correct) = σ(α<sub>s</sub>(θ<sub>m</sub> − β<sub>s</sub>))
      </code>
      <div className="mt-2 text-[11px] leading-4 text-muted-foreground">
        an encoder reads the item text to set
        <br />
        difficulty β<sub>s</sub> and discrimination α<sub>s</sub>
      </div>
    </div>
  )
}

// Stage 4: a mastery profile as a mini bar strip on the real color ramp.
const PROFILE = [
  0.92, 0.31, 0.74, 0.55, 0.18, 0.83, 0.46, 0.66, 0.27, 0.95, 0.51, 0.38,
  0.71, 0.6, 0.24, 0.88,
]

function ProfileVisual({ dark }: { dark: boolean }) {
  return (
    <div className="flex h-[64px] items-end gap-[3px]" aria-hidden>
      {PROFILE.map((v, i) => (
        <span
          key={i}
          className="w-[8px] rounded-t-[2px]"
          style={{
            height: `${Math.round(v * 60) + 4}px`,
            backgroundColor: getMasteryColor(v, dark),
          }}
        />
      ))}
    </div>
  )
}

export function PipelineDiagram() {
  const dark = useDarkMode()
  return (
    <div className="mt-4 flex flex-col gap-2 lg:flex-row">
      <StageCard
        step="1"
        title="Tag items with skills"
        foot="9,523 items from MATH, BBH, GPQA, MuSR, and IFEval, clustered into 100 named skills."
      >
        <ItemVisual />
      </StageCard>
      <Connector />
      <StageCard
        step="2"
        title="Collect responses"
        foot="3,811 models answer every item; a held-out slice of the matrix is kept for validation."
      >
        <MatrixVisual />
      </StageCard>
      <Connector />
      <StageCard
        step="3"
        title="Fit the diagnostic model"
        foot="Mastery θ is learned per model and per skill, jointly with the item parameters."
      >
        <ModelVisual />
      </StageCard>
      <Connector />
      <StageCard
        step="4"
        title="Read the profiles"
        foot="Every model gets a 100-dimensional mastery profile, browsable on the leaderboard."
      >
        <ProfileVisual dark={dark} />
      </StageCard>
    </div>
  )
}
