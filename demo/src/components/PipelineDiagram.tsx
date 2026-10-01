import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
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

// Stage 1: a real item (MATH, item #165) with the two skills the Q-matrix gives it.
const EXAMPLE_SKILLS = [
  { id: 6, label: 'find values from equations' },
  { id: 39, label: 'determine values from mathematical expressions' },
]

function ItemVisual() {
  return (
    <div className="w-full max-w-[220px] rounded-md border border-border bg-background p-2.5">
      <div className={kicker}>MATH item #165</div>
      <p className="mt-1 font-mono text-[11px] leading-4 text-foreground">
        Find the sum of the squares of the solutions to 2x² + 4x − 1 = 0.
      </p>
      <div className="mt-2 flex flex-wrap gap-1">
        {EXAMPLE_SKILLS.map((s) => (
          <Link
            key={s.id}
            to={`/skill/${s.id}`}
            className="rounded-md bg-brand/10 px-1.5 py-1 text-[10px] font-medium leading-3 text-brand hover:bg-brand/20"
          >
            {s.label}
          </Link>
        ))}
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

// Stage 3: the model, as in the paper: p = sigmoid(f(alpha_j (theta_m - d_j) * q_j)).
function ModelVisual() {
  return (
    <div className="text-center">
      <code className="inline-block rounded-md border border-border bg-background px-2 py-1.5 font-mono text-[10px] leading-4 text-foreground">
        p(correct) =
        <br />
        σ(f(α<sub>j</sub>(θ<sub>m</sub> − d<sub>j</sub>) ⊙ q<sub>j</sub>))
      </code>
      <div className="mt-2 text-[11px] leading-4 text-muted-foreground">
        α<sub>j</sub>, d<sub>j</sub>: item j's discrimination and per-skill
        difficulty, from its text · q<sub>j</sub>: its skills · f: more mastery
        never lowers p
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
        foot="An LLM lists the skills each of the 9,523 items needs; similar skills are grouped into 100 named skills."
      >
        <ItemVisual />
      </StageCard>
      <Connector />
      <StageCard
        step="2"
        title="Collect responses"
        foot="3,811 models answer every item of MATH, BBH, GPQA, MuSR and IFEval; 20% of the items are held out to test the predictions."
      >
        <MatrixVisual />
      </StageCard>
      <Connector />
      <StageCard
        step="3"
        title="Fit the diagnostic model"
        foot="Mastery θ is learned for every model and skill, together with the layers that turn item text into item parameters."
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
