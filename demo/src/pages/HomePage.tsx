import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, Grid3X3, ListOrdered, User } from 'lucide-react'
import { useSkillEvalData } from '@/hooks/useSkillEvalData'
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
  params: number | null
  accuracy?: number
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

export function HomePage() {
  const { models, skills, loading } = useSkillEvalData()
  const darkMode = useDarkMode()

  const top10 = useMemo<RankedModel[]>(() => {
    return models
      .map((m) => {
        let sum = 0
        for (const v of m.theta) sum += v
        return {
          id: m.id,
          name: m.name,
          family: m.family,
          params: m.params,
          accuracy: m.accuracy,
          meanTheta: m.theta.length ? sum / m.theta.length : 0,
        }
      })
      .sort((a, b) => b.meanTheta - a.meanTheta)
      .slice(0, 10)
  }, [models])

  const nModels = models.length > 0 ? models.length : 3811
  const nSkills = skills.length > 0 ? skills.length : 100

  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      {/* Hero */}
      <section className="max-w-[68ch]">
        <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Overview
        </div>
        <h1 className="mt-2 text-4xl font-semibold tracking-tight sm:text-5xl">
          SkillEval
        </h1>
        <p className="mt-4 text-[15px] leading-7 text-foreground/90">
          Skill-level ability estimates for {nModels.toLocaleString()} language
          models across {nSkills} cognitive skills, from a cognitive-diagnostic
          item-response model.
        </p>
        <p className="mt-2 text-sm text-muted-foreground tabular">
          Snapshot last updated 2026-05-22. {nModels.toLocaleString()} models,{' '}
          {nSkills} skills, 9,523 items, 5 public benchmarks.
        </p>
        <p className="mt-4 text-[15px] leading-7 text-foreground/90">
          A single benchmark accuracy hides what a model is actually good at.
          SkillEval decomposes performance into a profile over latent skills
          using cognitive diagnostic models from psychometrics: every model
          gets a mastery score θ on each of {nSkills} named skills, estimated
          jointly from its answers across five public benchmarks. Two models
          with the same average can diverge sharply on the skills users
          actually care about, and this site lets you see where.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link
            to="/browse"
            className="inline-flex items-center gap-2 rounded-md bg-foreground px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-90"
          >
            Browse the full table
            <ArrowRight className="h-4 w-4" />
          </Link>
          <Link
            to="/about"
            className="inline-flex items-center gap-2 rounded-md border border-border px-4 py-2 text-sm font-medium text-foreground hover:bg-accent"
          >
            About the method
          </Link>
        </div>
      </section>

      {/* Stats */}
      <section className="mt-16 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard value={nModels.toLocaleString()} label="LLMs profiled" />
        <StatCard value={String(nSkills)} label="named skills" />
        <StatCard value="9,523" label="benchmark items" />
        <StatCard value="5" label="benchmarks (MATH, BBH, GPQA, MuSR, IFEval)" />
      </section>

      {/* Top-10 preview */}
      <section className="mt-16">
        <div className="flex items-baseline justify-between">
          <h2 className="text-xl font-semibold tracking-tight">
            Top 10 by mean mastery
          </h2>
          <Link
            to="/browse"
            className="text-sm text-muted-foreground hover:text-foreground"
          >
            full table →
          </Link>
        </div>
        <div className="mt-3 overflow-hidden rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-surface text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="px-3 py-2 font-semibold">#</th>
                <th className="px-3 py-2 font-semibold">Model</th>
                <th className="px-3 py-2 font-semibold">Family</th>
                <th className="px-3 py-2 text-right font-semibold">Params</th>
                <th className="px-3 py-2 text-right font-semibold">Acc</th>
                <th className="px-3 py-2 text-right font-semibold">Mean θ</th>
              </tr>
            </thead>
            <tbody>
              {loading
                ? Array.from({ length: 10 }).map((_, i) => (
                    <tr key={i} className="border-b border-border last:border-0">
                      <td colSpan={6} className="px-3 py-2">
                        <div className="h-4 w-full animate-pulse rounded bg-muted" />
                      </td>
                    </tr>
                  ))
                : top10.map((m, i) => (
                    <tr
                      key={m.id}
                      className="border-b border-border last:border-0 hover:bg-surface-elevated/60"
                    >
                      <td className="tabular px-3 py-2 text-muted-foreground">
                        {i + 1}
                      </td>
                      <td
                        className="max-w-[280px] truncate px-3 py-2"
                        title={m.name}
                      >
                        {m.name}
                      </td>
                      <td className="px-3 py-2">
                        <span
                          className="inline-flex h-5 items-center rounded-full px-2 text-[10px] font-semibold text-white"
                          style={{ backgroundColor: getFamilyColor(m.family) }}
                        >
                          {m.family}
                        </span>
                      </td>
                      <td className="tabular px-3 py-2 text-right font-mono text-xs">
                        {formatParams(m.params)}
                      </td>
                      <td className="tabular px-3 py-2 text-right font-mono text-xs">
                        {m.accuracy != null
                          ? `${(m.accuracy * 100).toFixed(1)}%`
                          : '—'}
                      </td>
                      <td className="px-3 py-2 text-right">
                        <span
                          className="tabular inline-flex min-w-[3.5rem] justify-end rounded px-1.5 py-0.5 font-mono text-xs font-semibold"
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
                    </tr>
                  ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          Mean θ averages the model's mastery over all {nSkills} skills; Acc is
          the fraction of all 9,523 items answered correctly. A snapshot for
          inspection, not a leaderboard ranking.
        </p>
      </section>

      {/* What you can do */}
      <section className="mt-16 grid gap-4 md:grid-cols-3">
        <FeatureCard
          icon={<Grid3X3 className="h-5 w-5" />}
          title="Browse the grid"
          body={`The full ${nModels.toLocaleString()} × ${nSkills} mastery matrix: search any model, filter by family or tier, and sort by accuracy, mean θ, or any single skill to find the specialists.`}
          to="/browse"
          cta="Open the table"
        />
        <FeatureCard
          icon={<User className="h-5 w-5" />}
          title="Per-model profiles"
          body="Click any row to open that model's full skill profile: all skills ranked from strongest to weakest, in a scrollable list, next to its raw accuracy."
          to="/browse"
          cta="Pick a model"
        />
        <FeatureCard
          icon={<ListOrdered className="h-5 w-5" />}
          title="Per-skill pages"
          body="Every skill has its own page with the top-ranked LLMs on that skill and example benchmark items drawn from the underlying cluster."
          to={skills.length > 0 ? `/skill/${skills[0].id}` : '/browse'}
          cta="See an example"
        />
      </section>
    </div>
  )
}

function StatCard({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="text-3xl font-semibold tabular leading-none">{value}</div>
      <div className="mt-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
    </div>
  )
}

function FeatureCard({
  icon,
  title,
  body,
  to,
  cta,
}: {
  icon: React.ReactNode
  title: string
  body: string
  to: string
  cta: string
}) {
  return (
    <div className="flex flex-col rounded-lg border border-border bg-surface p-5">
      <div className="flex items-center gap-2 text-foreground">
        {icon}
        <h3 className="text-base font-semibold">{title}</h3>
      </div>
      <p className="mt-2 flex-1 text-sm leading-6 text-muted-foreground">
        {body}
      </p>
      <Link
        to={to}
        className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-foreground hover:text-brand"
      >
        {cta}
        <ArrowRight className="h-3.5 w-3.5" />
      </Link>
    </div>
  )
}
