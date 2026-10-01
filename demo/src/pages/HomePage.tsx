import { Fragment, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { PipelineDiagram } from '@/components/PipelineDiagram'
import { useSkillEvalData } from '@/hooks/useSkillEvalData'
import { usePageTitle } from '@/hooks/usePageTitle'
import { AFFILIATIONS, AUTHORS, DATA_URL, PAPER_TITLE, REPO_URL } from '@/lib/citation'
import type { Skill } from '@/lib/types'
import {
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

// Top-3 rank badges: tinted typographic circles instead of medal emojis.
const RANK_BADGE = [
  'bg-brand text-brand-foreground',
  'bg-brand/15 text-brand',
  'bg-brand/[0.08] text-brand/80',
]

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

/** A real slice of the mastery matrix: the nine models with the highest mean
 * mastery (rows) on 14 of the 100 skills (columns), on the site's colour scale. */
function MatrixMosaic({
  models,
  skills,
  darkMode,
}: {
  models: RankedModel[]
  skills: Skill[]
  darkMode: boolean
}) {
  const rows = models.slice(0, 9)
  const cols = Array.from({ length: 14 }, (_, i) => i * 7 + 2)
  if (rows.length === 0) {
    return <div className="h-[210px] w-[277px] animate-pulse rounded-lg bg-muted" />
  }
  const label = new Map(skills.map((s) => [s.id, s.label]))
  const ramp = [0, 0.25, 0.5, 0.75, 1].map((t) => getMasteryColor(t, darkMode)).join(', ')
  return (
    <figure className="w-[277px]">
      <div
        className="grid gap-[3px]"
        style={{ gridTemplateColumns: `repeat(${cols.length}, 17px)` }}
      >
        {rows.flatMap((m) =>
          cols.map((c) => {
            const theta = m.theta[c] ?? 0
            return (
              <span
                key={`${m.id}-${c}`}
                className="h-[17px] w-[17px] rounded-[3px]"
                style={{ backgroundColor: getMasteryColor(theta, darkMode) }}
                title={`${m.name} · ${label.get(c) ?? `skill ${c}`}: ${theta.toFixed(2)}`}
              />
            )
          })
        )}
      </div>
      <figcaption className="mt-3 flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <span>Top 9 models × 14 of the 100 skills</span>
        <span className="flex items-center gap-1.5">
          0
          <span className="h-2 w-14 rounded-sm" style={{ background: `linear-gradient(90deg, ${ramp})` }} />
          1
        </span>
      </figcaption>
    </figure>
  )
}

export function HomePage() {
  usePageTitle('SkillEval: interpretable ability profiles of LLMs')
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

  // Pre-seeded comparison: the strongest model vs the best one at <=13B,
  // which is the weak-beats-strong story in one click.
  const comparePair = useMemo(() => {
    if (top.length === 0) return null
    let bestSmall: RankedModel | null = null
    for (const m of models) {
      if (m.params == null || m.params > 13) continue
      let sum = 0
      for (const v of m.theta) sum += v
      const mean = m.theta.length ? sum / m.theta.length : 0
      if (!bestSmall || mean > bestSmall.meanTheta) {
        bestSmall = { ...m, meanTheta: mean } as RankedModel
      }
    }
    if (!bestSmall || bestSmall.id === top[0].id) return null
    return `${top[0].id},${bestSmall.id}`
  }, [top, models])



  return (
    <div>
      <div className="border-b border-border bg-gradient-to-b from-brand/[0.06] via-brand/[0.02] to-transparent">
        <section className="page flex flex-col gap-10 py-12 sm:py-14 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-[46rem]">
            <h1 className="text-balance text-3xl font-semibold leading-tight sm:text-[2.5rem]">
              {PAPER_TITLE}
            </h1>
            <p className="mt-5 text-[15px] leading-6 text-foreground">
              {AUTHORS.map((a, i) => (
                <Fragment key={a.name}>
                  {/* a name never breaks; the space after its comma can */}
                  <span className="whitespace-nowrap">
                    {a.name}
                    <sup className="ml-px text-[10px] text-muted-foreground">{a.affiliation}</sup>
                    {i < AUTHORS.length - 1 ? ',' : ''}
                  </span>
                  {i < AUTHORS.length - 1 ? ' ' : ''}
                </Fragment>
              ))}
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              {AFFILIATIONS.map((a, i) => (
                <span key={a} className="mr-4">
                  <sup className="mr-0.5 text-[10px]">{i + 1}</sup>
                  {a}
                </span>
              ))}
            </p>
            <p className="mt-6 max-w-[62ch] text-[15px] leading-7 text-foreground/90">
              SkillEval profiles 3,811 open language models on 100 named skills.
              The profiles come from a cognitive diagnosis model fitted to the
              models' answers to 9,523 items from MATH, BBH, GPQA, MuSR and
              IFEval.
            </p>
            <nav
              className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm font-medium"
              aria-label="Paper resources"
            >
              <span className="text-muted-foreground">Paper (forthcoming)</span>
              <a href={REPO_URL} className="text-brand hover:underline">
                Code
              </a>
              <a href={DATA_URL} className="text-brand hover:underline">
                Data
              </a>
              <Link to="/about#cite" className="text-brand hover:underline">
                Citation
              </Link>
              <Link to="/leaderboard" className="text-brand hover:underline">
                Leaderboard
              </Link>
            </nav>
          </div>
          <div className="hidden shrink-0 lg:block">
            <MatrixMosaic models={top} skills={skills} darkMode={darkMode} />
          </div>
        </section>
      </div>

      <div className="page pb-14">
      {/* Compact leaderboard preview */}
      <section className="mt-12">
        <div className="flex items-baseline justify-between">
          <h2 className="text-xl font-semibold tracking-tight">
            Top 12 models by mean mastery
          </h2>
          <Link
            to="/leaderboard"
            className="text-sm font-medium text-brand transition-colors hover:underline"
          >
            Full leaderboard
          </Link>
        </div>
        <div className="mt-4 overflow-x-auto rounded-lg border border-border bg-card">
          <table className="w-full text-sm tabular">
            <thead>
              <tr className="border-b border-border bg-surface text-left text-xs text-muted-foreground">
                <th className="w-12 px-3 py-2 text-center font-semibold">#</th>
                <th className="px-3 py-2 font-semibold">Model</th>
                <th className="hidden w-24 px-3 py-2 font-semibold md:table-cell">Family</th>
                <th className="hidden w-20 px-3 py-2 text-right font-semibold md:table-cell">Size</th>
                <th className="hidden w-24 px-3 py-2 text-right font-semibold sm:table-cell">Accuracy</th>
                <th className="w-28 px-3 py-2 text-right font-semibold">Mean mastery</th>
                <th className="hidden w-[120px] px-3 py-2 font-semibold sm:table-cell">Profile</th>
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
                            <span
                              className={`tabular inline-flex h-6 w-6 items-center justify-center rounded-full text-xs font-semibold ${RANK_BADGE[rank - 1]}`}
                            >
                              {rank}
                            </span>
                          ) : (
                            <span className="tabular text-muted-foreground">
                              {rank}
                            </span>
                          )}
                        </td>
                        <td className="max-w-[11rem] px-3 py-2.5 sm:max-w-none">
                          <Link
                            to={`/model/${m.id}`}
                            className="flex min-w-0 items-center gap-2 font-medium transition-colors hover:text-brand"
                            title={`Open the model page for ${m.name}`}
                          >
                            <span
                              className="h-2 w-2 shrink-0 rounded-full"
                              style={{ backgroundColor: getFamilyColor(m.family) }}
                            />
                            <span className="truncate">{m.name}</span>
                          </Link>
                        </td>
                        <td className="hidden px-3 py-2.5 md:table-cell">
                          <span
                            className="inline-flex h-5 items-center rounded-full border px-2 text-[10px] font-semibold"
                            style={{
                              color: getFamilyColor(m.family),
                              borderColor: `${getFamilyColor(m.family)}55`,
                              backgroundColor: `${getFamilyColor(m.family)}14`,
                            }}
                            title={
                              m.family === 'Other'
                                ? 'Open-weights model outside the six major families (community fine-tunes and merges)'
                                : `${m.family} family`
                            }
                          >
                            {m.family}
                          </span>
                        </td>
                        <td className="hidden px-3 py-2.5 text-right font-mono text-xs md:table-cell">
                          {formatParams(m.params)}
                        </td>
                        <td
                          className="hidden px-3 py-2.5 text-right font-mono text-xs sm:table-cell"
                          title="Fraction of 9,523 items answered correctly"
                        >
                          {m.accuracy != null
                            ? `${(m.accuracy * 100).toFixed(1)}%`
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
                        <td className="hidden px-3 py-2.5 sm:table-cell">
                          <Link
                            to={`/model/${m.id}`}
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
          Ranked by mean mastery over all 100 skills. Mastery estimates carry sampling
          noise, so small gaps between adjacent ranks are not meaningful. All
          models are open-weights; the Other family groups models outside the
          six major families, mostly community fine-tunes and merges.
        </p>
        <p className="mt-3 text-sm text-foreground/90">
          To see two or three models side by side, open{' '}
          <Link
            to={comparePair ? `/compare?m=${comparePair}` : '/compare'}
            className="font-medium text-brand hover:underline"
          >
            Compare
          </Link>
          .
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
            Read the method
          </Link>
        </div>
      </section>

      </div>
    </div>
  )
}
