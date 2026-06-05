import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight } from 'lucide-react'
import { PipelineDiagram } from '@/components/PipelineDiagram'
import { useSkillEvalData } from '@/hooks/useSkillEvalData'
import { displaySkillLabel } from '@/lib/labels'
import {
  getBenchmarkColor,
  getFamilyColor,
  getMasteryColor,
  getMasteryTextColor,
  rgbToRgba,
} from '@/lib/colors'

interface RankedModel {
  id: number
  name: string
  family: string
  tier: string
  params: number | null
  accuracy?: number
  theta: number[]
  meanTheta: number
}

function formatParams(p: number | null): string {
  if (p == null) return '—'
  if (p >= 1) return `${p.toFixed(p >= 10 ? 0 : 1)}B`
  return `${(p * 1000).toFixed(0)}M`
}

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

const MEDALS = ['🥇', '🥈', '🥉']

/** Mastery profile as a sorted, downsampled sparkline: skills ranked from
 * strongest to weakest, so the SHAPE of the curve is what differs between
 * models (flat = generalist, steep = narrow specialist). Raw unsorted bars
 * all look like identical noise at this size. */
function ProfileSparkline({
  theta,
  darkMode,
}: {
  theta: number[]
  darkMode: boolean
}) {
  const W = 100
  const H = 20
  const BUCKETS = 25
  const bars = useMemo(() => {
    const sorted = [...theta].sort((a, b) => b - a)
    const per = Math.max(1, Math.floor(sorted.length / BUCKETS))
    return Array.from({ length: BUCKETS }, (_, i) => {
      const slice = sorted.slice(i * per, (i + 1) * per)
      return slice.length
        ? slice.reduce((a, b) => a + b, 0) / slice.length
        : 0
    })
  }, [theta])
  return (
    <svg
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="Skill profile, strongest to weakest"
      className="block"
    >
      {bars.map((t, i) => {
        const h = Math.max(1.5, t * H)
        return (
          <rect
            key={i}
            x={i * 4}
            y={H - h}
            width={3}
            height={h}
            rx={0.75}
            fill={getMasteryColor(t, darkMode)}
          />
        )
      })}
    </svg>
  )
}

/** Deterministic per-cell pseudo-random in [0,1): no Math.random so frames
 * are reproducible. */
function cellRand(i: number, tick: number): number {
  const x = Math.sin(i * 127.1 + tick * 311.7) * 43758.5453
  return x - Math.floor(x)
}

const NOISE_TICKS = 4 // distinct response samples shown while flickering
const NOISE_MS = 420 // per flicker frame
const HOLD_MS = 4600 // resolved heatmap hold
const RESOLVE_MS = 550 // per-cell color transition

/** The hero visual tells the method's story: raw correct/incorrect responses
 * (real Bernoulli draws from each cell's actual theta, so strong rows read
 * denser even as noise) resolve wave-like into the mastery heatmap, hold,
 * and loop. Static heatmap when the user prefers reduced motion. */
function MatrixMosaic({
  models,
  darkMode,
}: {
  models: RankedModel[]
  darkMode: boolean
}) {
  const rows = models.slice(0, 9)
  const cols = Array.from({ length: 14 }, (_, i) => i * 7 + 2)

  const reduced = useMemo(
    () =>
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    []
  )
  const [resolved, setResolved] = useState(reduced)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    if (reduced || rows.length === 0) return
    let alive = true
    let t: ReturnType<typeof setTimeout>
    const loop = (phase: 'noise' | 'hold', step: number) => {
      if (!alive) return
      if (phase === 'noise') {
        if (step < NOISE_TICKS) {
          setTick(step)
          t = setTimeout(() => loop('noise', step + 1), NOISE_MS)
        } else {
          setResolved(true)
          t = setTimeout(() => loop('hold', 0), HOLD_MS)
        }
      } else {
        setResolved(false)
        t = setTimeout(() => loop('noise', 0), NOISE_MS)
      }
    }
    loop('noise', 0)
    return () => {
      alive = false
      clearTimeout(t)
    }
  }, [reduced, rows.length])

  if (rows.length === 0) {
    return (
      <div className="h-[210px] w-[300px] animate-pulse rounded-lg bg-muted" />
    )
  }

  const onColor = darkMode ? 'hsl(220 18% 72%)' : 'hsl(222 32% 32%)'
  const offColor = darkMode ? 'hsl(222 18% 15%)' : 'hsl(220 28% 89%)'

  return (
    <div aria-hidden>
      <div
        className="grid gap-[3px]"
        style={{ gridTemplateColumns: `repeat(${cols.length}, 17px)` }}
      >
        {rows.flatMap((m, r) =>
          cols.map((c, ci) => {
            const theta = m.theta[c] ?? 0
            const i = r * cols.length + ci
            const correct = cellRand(i, tick) < theta
            return (
              <span
                key={`${r}-${c}`}
                className="h-[17px] w-[17px] rounded-[3px]"
                style={{
                  backgroundColor: resolved
                    ? getMasteryColor(theta, darkMode)
                    : correct
                      ? onColor
                      : offColor,
                  transition: `background-color ${RESOLVE_MS}ms ease`,
                  // left-to-right wave when resolving; snap back together
                  transitionDelay: resolved ? `${ci * 55 + r * 14}ms` : '0ms',
                }}
              />
            )
          })
        )}
      </div>
      <div className="relative mt-2 h-4 text-right text-[11px] text-muted-foreground">
        <span
          className="absolute inset-0 whitespace-nowrap transition-opacity duration-500"
          style={{ opacity: resolved ? 0 : 1 }}
        >
          raw right-or-wrong responses, one sample
        </span>
        <span
          className="absolute inset-0 whitespace-nowrap transition-opacity duration-500"
          style={{ opacity: resolved ? 1 : 0 }}
        >
          resolved: a slice of the real mastery matrix
        </span>
      </div>
    </div>
  )
}

export function HomePage() {
  const { models, skills, loading } = useSkillEvalData()
  const darkMode = useDarkMode()

  const top = useMemo<RankedModel[]>(() => {
    return models
      .map((m) => {
        let sum = 0
        for (const v of m.theta) sum += v
        return {
          id: m.id,
          name: m.name,
          family: m.family,
          tier: m.tier,
          params: m.params,
          accuracy: m.accuracy,
          theta: m.theta,
          meanTheta: m.theta.length ? sum / m.theta.length : 0,
        }
      })
      .sort((a, b) => b.meanTheta - a.meanTheta)
      .slice(0, 12)
  }, [models])

  const stats = [
    { value: '3,811', label: 'open-weights models' },
    { value: '100', label: 'skills' },
    { value: '9,523', label: 'items' },
    { value: '5', label: 'benchmarks' },
  ]

  // Skills grouped by source benchmark, in the grid's column order.
  const skillGroups = useMemo(() => {
    const order = ['MATH', 'BBH', 'GPQA', 'IFEval', 'MuSR']
    const byBench = new Map<string, typeof skills>()
    for (const s of skills) {
      const b = s.primary_benchmark
      if (!byBench.has(b)) byBench.set(b, [])
      byBench.get(b)!.push(s)
    }
    const benches = [
      ...order.filter((b) => byBench.has(b)),
      ...[...byBench.keys()].filter((b) => !order.includes(b)),
    ]
    return benches.map((b) => ({ benchmark: b, skills: byBench.get(b)! }))
  }, [skills])

  return (
    <div>
      {/* Hero: full-bleed blue wash band with a data-true matrix mosaic */}
      <div className="border-b border-border bg-gradient-to-b from-brand/[0.08] via-brand/[0.03] to-transparent">
        <section className="mx-auto flex max-w-6xl flex-col gap-10 px-6 py-12 sm:py-16 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-[60ch]">
            <div className="text-xs font-medium uppercase tracking-wider text-brand">
              Item-response leaderboard
            </div>
            <h1 className="mt-3 text-balance text-4xl font-semibold tracking-tight sm:text-5xl">
              Skill-level mastery for every language model.
            </h1>
            <p className="mt-4 text-lg leading-relaxed text-muted-foreground">
              SkillEval estimates how 3,811 models master 100 academic skills
              with a cognitive-diagnostic item-response model. One profile per
              model, not one score.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-x-3 gap-y-2">
              {stats.map((s, i) => (
                <span key={s.label} className="flex items-center gap-3 text-sm">
                  <span>
                    <span className="tabular font-semibold text-foreground">
                      {s.value}
                    </span>{' '}
                    <span className="text-muted-foreground">{s.label}</span>
                  </span>
                  {i < stats.length - 1 ? (
                    <span aria-hidden className="h-3.5 w-px bg-border" />
                  ) : null}
                </span>
              ))}
            </div>
            <div className="mt-7 flex flex-wrap gap-3">
              <Link
                to="/leaderboard"
                className="inline-flex items-center gap-2 rounded-lg bg-brand px-4 py-2 text-sm font-medium text-brand-foreground shadow-sm transition-colors hover:bg-brand/90"
              >
                View full leaderboard
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                to="/about"
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-sm font-medium text-foreground shadow-sm transition-colors hover:bg-accent"
              >
                How it works
              </Link>
            </div>
          </div>
          <div className="hidden lg:block">
            <MatrixMosaic models={top} darkMode={darkMode} />
          </div>
        </section>
      </div>

      <div className="mx-auto max-w-6xl px-6 pb-14">
      {/* Compact leaderboard preview */}
      <section className="mt-12">
        <div className="flex items-baseline justify-between">
          <h2 className="text-xl font-semibold tracking-tight">
            Top models by overall mastery
          </h2>
          <Link
            to="/leaderboard"
            className="text-sm font-medium text-brand transition-colors hover:underline"
          >
            See the full 3,811 × 100 grid →
          </Link>
        </div>
        <div className="mt-4 overflow-hidden rounded-xl border border-border bg-card">
          <table className="w-full text-sm tabular">
            <thead>
              <tr className="border-b border-border bg-surface text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="w-12 px-3 py-2 text-center font-semibold">#</th>
                <th className="px-3 py-2 font-semibold">Model</th>
                <th className="w-24 px-3 py-2 font-semibold">Family</th>
                <th className="w-20 px-3 py-2 text-right font-semibold">Params</th>
                <th className="w-20 px-3 py-2 text-right font-semibold">Acc</th>
                <th className="w-24 px-3 py-2 text-right font-semibold">Mean θ</th>
                <th className="w-[120px] px-3 py-2 font-semibold">Profile</th>
              </tr>
            </thead>
            <tbody>
              {loading
                ? Array.from({ length: 12 }).map((_, i) => (
                    <tr key={i} className="border-b border-border last:border-0">
                      <td colSpan={7} className="px-3 py-2.5">
                        <div className="h-4 w-full animate-pulse rounded bg-muted" />
                      </td>
                    </tr>
                  ))
                : top.map((m, i) => {
                    const rank = i + 1
                    return (
                      <tr
                        key={m.id}
                        className="border-b border-border last:border-0 transition-colors hover:bg-surface-elevated/60"
                      >
                        <td
                          className="px-3 py-2.5 text-center"
                          aria-label={`Rank ${rank}`}
                        >
                          {rank <= 3 ? (
                            <span aria-hidden className="text-base">
                              {MEDALS[rank - 1]}
                            </span>
                          ) : (
                            <span className="tabular text-muted-foreground">
                              {rank}
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-2.5">
                          <Link
                            to="/leaderboard"
                            className="flex items-center gap-2 font-medium transition-colors hover:text-brand"
                            title={m.name}
                          >
                            <span
                              className="h-2 w-2 shrink-0 rounded-full"
                              style={{ backgroundColor: getFamilyColor(m.family) }}
                            />
                            <span className="truncate">{m.name}</span>
                          </Link>
                        </td>
                        <td className="px-3 py-2.5">
                          <span
                            className="inline-flex h-5 items-center rounded-full px-2 text-[10px] font-semibold text-white"
                            style={{ backgroundColor: getFamilyColor(m.family) }}
                            title={
                              m.family === 'Other'
                                ? 'Open-weights model outside the six major families (community fine-tunes and merges)'
                                : `${m.family} family`
                            }
                          >
                            {m.family}
                          </span>
                        </td>
                        <td className="px-3 py-2.5 text-right font-mono text-xs">
                          {formatParams(m.params)}
                        </td>
                        <td
                          className="px-3 py-2.5 text-right font-mono text-xs"
                          title="Fraction of 9,523 items answered correctly"
                        >
                          {m.accuracy != null
                            ? `${(m.accuracy * 100).toFixed(2)}%`
                            : '—'}
                        </td>
                        <td className="px-3 py-2.5 text-right">
                          <span
                            className="inline-flex min-w-[3.5rem] justify-end rounded px-1.5 py-0.5 font-mono text-xs font-semibold"
                            style={{
                              backgroundColor: rgbToRgba(
                                getMasteryColor(m.meanTheta, darkMode),
                                0.7
                              ),
                              color: getMasteryTextColor(m.meanTheta, darkMode),
                            }}
                          >
                            {m.meanTheta.toFixed(3)}
                          </span>
                        </td>
                        <td className="px-3 py-2.5">
                          <Link
                            to="/leaderboard"
                            title="100-skill mastery profile"
                            className="inline-block"
                          >
                            <ProfileSparkline theta={m.theta} darkMode={darkMode} />
                          </Link>
                        </td>
                      </tr>
                    )
                  })}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          Ranked by mean θ over all 100 skills. Mastery estimates carry sampling
          noise, so small gaps between adjacent ranks are not meaningful. All
          models are open-weights; the Other family groups models outside the
          six major families, mostly community fine-tunes and merges. Snapshot
          2026-05-22.
        </p>
      </section>

      {/* How it works */}
      <section className="mt-16">
        <h2 className="text-xl font-semibold tracking-tight">How it works</h2>
        <PipelineDiagram />
        <div className="mt-4">
          <Link
            to="/about"
            className="text-sm font-medium text-brand transition-colors hover:underline"
          >
            Read the methodology →
          </Link>
        </div>
      </section>

      {/* Skill explorer teaser, grouped by benchmark */}
      <section className="mt-16">
        <div className="flex items-baseline justify-between">
          <h2 className="text-xl font-semibold tracking-tight">
            Mastery, skill by skill.
          </h2>
          <Link
            to="/leaderboard"
            className="text-sm font-medium text-brand transition-colors hover:underline"
          >
            Browse every skill in the grid →
          </Link>
        </div>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {skillGroups.map((g) => (
            <div
              key={g.benchmark}
              className="rounded-xl border border-border bg-card p-4"
            >
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-2 text-sm font-semibold">
                  <span
                    className="h-2 w-2 shrink-0 rounded-full"
                    style={{ backgroundColor: getBenchmarkColor(g.benchmark) }}
                  />
                  {g.benchmark}
                </span>
                <span className="text-xs text-muted-foreground">
                  {g.skills.length} skills
                </span>
              </div>
              <ul className="mt-3 space-y-1">
                {g.skills.slice(0, 5).map((s) => (
                  <li key={s.id}>
                    <Link
                      to={`/skill/${s.id}`}
                      className="block truncate rounded px-1.5 py-1 text-[13px] text-foreground/85 transition-colors hover:bg-accent hover:text-brand"
                      title={displaySkillLabel(s.label_english ?? s.label)}
                    >
                      {s.label_english ?? s.label}
                    </Link>
                  </li>
                ))}
              </ul>
              {g.skills.length > 5 ? (
                <Link
                  to={`/skill/${g.skills[5].id}`}
                  className="mt-2 inline-block px-1.5 text-xs font-medium text-muted-foreground transition-colors hover:text-brand"
                >
                  + {g.skills.length - 5} more
                </Link>
              ) : null}
            </div>
          ))}
        </div>
      </section>
      </div>
    </div>
  )
}
