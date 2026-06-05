import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, Boxes, FileText, Layers, Library } from 'lucide-react'
import { useSkillEvalData } from '@/hooks/useSkillEvalData'
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

/** 100-skill mastery profile as a compact SVG bar sparkline. */
function ProfileSparkline({
  theta,
  darkMode,
}: {
  theta: number[]
  darkMode: boolean
}) {
  const W = 100
  const H = 20
  return (
    <svg
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="100-skill mastery profile"
      className="block"
    >
      {theta.map((t, i) => {
        const h = Math.max(1, t * H)
        return (
          <rect
            key={i}
            x={i}
            y={H - h}
            width={1}
            height={h}
            fill={getMasteryColor(t, darkMode)}
          />
        )
      })}
    </svg>
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

  const chips = [
    { icon: <Boxes className="h-4 w-4" />, label: '3,811 models' },
    { icon: <Layers className="h-4 w-4" />, label: '100 skills' },
    { icon: <FileText className="h-4 w-4" />, label: '9,523 items' },
    { icon: <Library className="h-4 w-4" />, label: '5 benchmarks' },
  ]

  return (
    <div className="mx-auto max-w-6xl px-6 py-10 sm:py-14">
      {/* Hero */}
      <section className="max-w-[60ch]">
        <div className="text-xs font-medium uppercase tracking-wider text-brand">
          Item-response leaderboard
        </div>
        <h1 className="mt-3 text-balance text-4xl font-semibold tracking-tight sm:text-5xl">
          Skill-level mastery for every language model.
        </h1>
        <p className="mt-4 text-lg leading-relaxed text-muted-foreground">
          SkillEval estimates how 3,811 models master 100 academic skills with a
          cognitive-diagnostic item-response model. One profile per model, not
          one score.
        </p>
        <div className="mt-6 flex flex-wrap gap-2">
          {chips.map((c) => (
            <span
              key={c.label}
              className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-3 py-1 text-sm"
            >
              {c.icon}
              {c.label}
            </span>
          ))}
        </div>
        <div className="mt-7 flex flex-wrap gap-3">
          <Link
            to="/leaderboard"
            className="inline-flex items-center gap-2 rounded-md bg-brand px-4 py-2 text-sm font-medium text-brand-foreground transition-colors hover:bg-brand/90"
          >
            View full leaderboard
            <ArrowRight className="h-4 w-4" />
          </Link>
          <Link
            to="/about"
            className="inline-flex items-center gap-2 rounded-md border border-border px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent"
          >
            How it works
          </Link>
        </div>
      </section>

      {/* Trust / scale strip */}
      <section className="mt-10 border-y border-border py-4">
        <p className="text-sm text-muted-foreground">
          Profiling 3,811 open-weights models across the Llama, Qwen, Gemma,
          Mistral, Phi, and Mixtral families, plus community fine-tunes.
        </p>
      </section>

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
          noise, so small gaps between adjacent ranks are not meaningful.
          Snapshot 2026-05-22.
        </p>
      </section>

      {/* How it works */}
      <section className="mt-16">
        <h2 className="text-xl font-semibold tracking-tight">How it works</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-3">
          <div className="rounded-xl border border-border bg-surface p-5">
            <svg
              width={64}
              height={40}
              viewBox="0 0 64 40"
              aria-hidden
              className="mb-3"
            >
              {Array.from({ length: 5 }).map((_, r) =>
                Array.from({ length: 8 }).map((_, c) => (
                  <rect
                    key={`${r}-${c}`}
                    x={c * 8}
                    y={r * 8}
                    width={6}
                    height={6}
                    rx={1}
                    className="fill-muted-foreground/30"
                  />
                ))
              )}
            </svg>
            <h3 className="text-base font-semibold">Score matrix</h3>
            <p className="mt-1 text-sm leading-6 text-muted-foreground">
              3,811 models answer 9,523 items from MATH, BBH, GPQA, MuSR, and
              IFEval, clustered into 100 skills.
            </p>
          </div>
          <div className="rounded-xl border border-border bg-surface p-5">
            <div className="mb-3 rounded-md border border-border bg-card px-3 py-2 font-mono text-sm">
              p(correct) = σ( α_s · (θ_m − β_s) )
            </div>
            <h3 className="text-base font-semibold">Fit the model</h3>
            <p className="mt-1 text-sm leading-6 text-muted-foreground">
              A cognitive-diagnostic model estimates each model's per-skill
              mastery θ jointly with each skill's discrimination α and difficulty
              β.
            </p>
          </div>
          <div className="rounded-xl border border-border bg-surface p-5">
            <div className="mb-3 h-[40px]">
              {top.length > 0 ? (
                <ProfileSparkline theta={top[0].theta} darkMode={darkMode} />
              ) : null}
            </div>
            <h3 className="text-base font-semibold">Read the profile</h3>
            <p className="mt-1 text-sm leading-6 text-muted-foreground">
              Every model gets a 100-skill mastery profile, sortable and
              searchable on the full grid.
            </p>
          </div>
        </div>
        <div className="mt-4">
          <Link
            to="/about"
            className="text-sm font-medium text-brand transition-colors hover:underline"
          >
            Read the methodology →
          </Link>
        </div>
      </section>

      {/* Skill explorer teaser */}
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
        <div className="mt-4 flex flex-wrap gap-2">
          {skills.map((s) => (
            <Link
              key={s.id}
              to={`/skill/${s.id}`}
              className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-1 text-xs text-foreground transition-colors hover:border-brand hover:text-brand"
            >
              <span
                className="h-1.5 w-1.5 shrink-0 rounded-full"
                style={{
                  backgroundColor: getBenchmarkColor(s.primary_benchmark),
                }}
              />
              <span className="max-w-[16ch] truncate">
                {s.label_english ?? s.label}
              </span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  )
}
