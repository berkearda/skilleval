import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, ExternalLink, GitCompareArrows } from 'lucide-react'
import { useSkillEvalData } from '@/hooks/useSkillEvalData'
import { usePageTitle } from '@/hooks/usePageTitle'
import { displaySkillLabel } from '@/lib/labels'
import { ModelFingerprint } from '@/components/ModelFingerprint'
import {
  getFamilyColor,
  getMasteryColor,
  getMasteryTextColor,
  rgbToRgba,
} from '@/lib/colors'
import type { Skill } from '@/lib/types'

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

function formatParams(p: number | null): string {
  if (p == null) return '—'
  if (p >= 1) return `${p.toFixed(p >= 10 ? 0 : 1)}B`
  return `${(p * 1000).toFixed(0)}M`
}

const kicker =
  'text-xs font-medium uppercase tracking-wider text-muted-foreground'

function MetaStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className={kicker}>{label}</span>
      <span className="tabular font-mono text-sm text-foreground">{value}</span>
    </div>
  )
}

export function ModelPage() {
  const { id } = useParams<{ id: string }>()
  const { models, skills, loading, error } = useSkillEvalData()
  const darkMode = useDarkMode()

  const modelId = useMemo(() => {
    if (id == null) return null
    const n = Number.parseInt(id, 10)
    return Number.isFinite(n) ? n : null
  }, [id])

  const model = useMemo(
    () => models.find((m) => m.id === modelId) ?? null,
    [models, modelId]
  )

  usePageTitle(model ? `${model.name} · SkillEval` : 'Model · SkillEval')

  // Grid-ordered skills for the fingerprint; theta-ranked for the list.
  const orderedSkills = useMemo(() => {
    const order = ['MATH', 'BBH', 'GPQA', 'IFEval', 'MuSR']
    const byBench = new Map<string, Skill[]>()
    for (const s of skills) {
      const b = s.primary_benchmark
      if (!byBench.has(b)) byBench.set(b, [])
      byBench.get(b)!.push(s)
    }
    const benches = [
      ...order.filter((b) => byBench.has(b)),
      ...[...byBench.keys()].filter((b) => !order.includes(b)),
    ]
    return benches.flatMap((b) => byBench.get(b)!)
  }, [skills])

  const ranked = useMemo(() => {
    if (!model) return []
    return skills
      .map((s) => ({ skill: s, theta: model.theta[s.id] ?? 0 }))
      .sort((a, b) => b.theta - a.theta)
      .map((r, i) => ({ ...r, rank: i + 1 }))
  }, [model, skills])

  const meanTheta = useMemo(() => {
    if (!model || model.theta.length === 0) return 0
    let sum = 0
    for (const v of model.theta) sum += v
    return sum / model.theta.length
  }, [model])

  if (error) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-10">
        <div className="rounded-md border border-destructive bg-destructive/10 p-4 text-sm text-destructive">
          Error loading data: {error}
        </div>
      </div>
    )
  }
  if (loading) {
    return (
      <div className="mx-auto max-w-5xl px-6 py-10">
        <div className="h-72 animate-pulse rounded-xl bg-muted" />
      </div>
    )
  }
  if (!model) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-16">
        <h1 className="text-2xl font-semibold tracking-tight">
          Model not found
        </h1>
        <p className="mt-3 text-muted-foreground">
          No model matches the id{' '}
          <code className="rounded bg-muted px-1.5 py-0.5 text-sm">
            {id ?? '(missing)'}
          </code>
          .
        </p>
        <Link
          to="/leaderboard"
          className="mt-6 inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to leaderboard
        </Link>
      </div>
    )
  }

  const familyColor = getFamilyColor(model.family)
  const hfUrl = model.hf_id
    ? `https://huggingface.co/${model.hf_id.replace('__', '/')}`
    : null

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <Link
        to="/leaderboard"
        className="inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Back to leaderboard
      </Link>

      {/* header */}
      <header className="mt-6 border-b border-border pb-6">
        <div className={kicker}>Model</div>
        <h1 className="mt-2 break-words text-2xl font-semibold tracking-tight sm:text-3xl">
          {model.name}
        </h1>
        <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2">
          <span
            className="inline-flex h-5 items-center rounded-full border px-2 text-[11px] font-semibold"
            style={{
              color: familyColor,
              borderColor: `${familyColor}55`,
              backgroundColor: `${familyColor}14`,
            }}
          >
            {model.family}
          </span>
          <MetaStat label="Tier" value={model.tier} />
          <MetaStat label="Params" value={formatParams(model.params)} />
          <MetaStat
            label="Accuracy"
            value={
              model.accuracy != null
                ? `${(model.accuracy * 100).toFixed(1)}%`
                : '—'
            }
          />
          <MetaStat label="Mean θ" value={meanTheta.toFixed(3)} />
          {hfUrl ? (
            <a
              href={hfUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm text-brand hover:underline"
            >
              Hugging Face
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
          ) : null}
          <Link
            to={`/compare?m=${model.id}`}
            className="inline-flex items-center gap-1.5 rounded-lg bg-brand px-3 py-1.5 text-xs font-medium text-brand-foreground shadow-sm transition-colors hover:bg-brand/90"
          >
            <GitCompareArrows className="h-3.5 w-3.5" />
            Compare with...
          </Link>
        </div>
      </header>

      {/* fingerprint + ranked skills */}
      <div className="mt-8 flex flex-col gap-8 lg:flex-row">
        <div className="shrink-0 lg:sticky lg:top-20 lg:self-start">
          <div className="flex flex-col items-center rounded-xl border border-border bg-card p-6">
            <ModelFingerprint
              model={model}
              orderedSkills={orderedSkills}
              darkMode={darkMode}
              size={240}
            />
            <p className="mt-3 max-w-[240px] text-center text-[11px] leading-4 text-muted-foreground">
              The skill fingerprint: one spoke per skill in grid order, length
              and color by mastery, benchmark sectors on the rim.
            </p>
          </div>
        </div>

        <div className="min-w-0 flex-1">
          <h2 className="text-lg font-semibold tracking-tight">
            All {ranked.length} skills, ranked by mastery
          </h2>
          <ul className="mt-3 space-y-1">
            {ranked.map(({ skill, theta, rank }) => {
              const bg = getMasteryColor(theta, darkMode)
              const fg = getMasteryTextColor(theta, darkMode)
              const barColor = rgbToRgba(bg, darkMode ? 0.25 : 0.18)
              return (
                <li key={skill.id}>
                  <Link
                    to={`/skill/${skill.id}`}
                    className="group relative flex items-center gap-2 overflow-hidden rounded-md border border-border bg-card px-2 py-1 text-sm hover:bg-surface-elevated"
                  >
                    <span
                      aria-hidden
                      className="absolute inset-y-0 left-0"
                      style={{
                        width: `${theta * 100}%`,
                        backgroundColor: barColor,
                      }}
                    />
                    <span className="tabular relative w-7 shrink-0 text-right font-mono text-[11px] text-muted-foreground">
                      {rank}
                    </span>
                    <span
                      className="tabular relative inline-flex h-6 w-12 shrink-0 items-center justify-center rounded font-mono text-xs"
                      style={{ backgroundColor: bg, color: fg }}
                    >
                      {theta.toFixed(2)}
                    </span>
                    <span
                      className="relative truncate"
                      title={skill.label_english ?? skill.label}
                    >
                      {displaySkillLabel(skill.label_english ?? skill.label)}
                    </span>
                    <span className="relative ml-auto shrink-0 text-xs text-muted-foreground">
                      {skill.primary_benchmark}
                    </span>
                    <ExternalLink className="relative h-3 w-3 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
                  </Link>
                </li>
              )
            })}
          </ul>
        </div>
      </div>
    </div>
  )
}
