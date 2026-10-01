import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, Download, ExternalLink, GitCompareArrows } from 'lucide-react'
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
import type { Model, Skill } from '@/lib/types'

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


/** Save the model's 100-skill profile as a CSV file, built in the browser. */
function downloadProfile(model: Model, skills: Skill[]) {
  const cell = (v: string) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v)
  const rows = [['skill_id', 'skill_name', 'primary_benchmark', 'mastery_theta']]
  for (const s of [...skills].sort((a, b) => a.id - b.id)) {
    rows.push([String(s.id), s.label, s.primary_benchmark, String(model.theta[s.id])])
  }
  const csv = rows.map((r) => r.map(cell).join(',')).join('\n') + '\n'
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `skilleval_profile_${(model.hf_id ?? model.name).replace(/[^A-Za-z0-9._-]+/g, '_')}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

const ACTION =
  'inline-flex h-8 items-center gap-1.5 rounded-md border border-border bg-card px-3 text-xs font-medium text-foreground transition-colors hover:bg-surface-elevated'


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

  const [showAll, setShowAll] = useState(false)

  const meanTheta = useMemo(() => {
    if (!model || model.theta.length === 0) return 0
    let sum = 0
    for (const v of model.theta) sum += v
    return sum / model.theta.length
  }, [model])

  if (error) {
    return (
      <div className="page py-10">
        <div className="rounded-md border border-destructive bg-destructive/10 p-4 text-sm text-destructive">
          Error loading data: {error}
        </div>
      </div>
    )
  }
  if (loading) {
    return (
      <div className="page py-10">
        <div className="h-72 animate-pulse rounded-lg bg-muted" />
      </div>
    )
  }
  if (!model) {
    return (
      <div className="page py-16">
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

  const renderRow = ({ skill, theta, rank }: { skill: Skill; theta: number; rank: number }) => {
    const bg = getMasteryColor(theta, darkMode)
    const fg = getMasteryTextColor(theta, darkMode)
    const barColor = rgbToRgba(bg, darkMode ? 0.25 : 0.18)
    return (
      <li key={skill.id}>
        <Link
          to={`/skill/${skill.id}`}
          className="group relative flex items-center gap-2 overflow-hidden rounded-md border border-border bg-card px-2 py-1 text-sm hover:bg-surface-elevated"
        >
          <span aria-hidden className="absolute inset-y-0 left-0" style={{ width: `${theta * 100}%`, backgroundColor: barColor }} />
          <span className="tabular relative w-7 shrink-0 text-right font-mono text-[11px] text-muted-foreground">{rank}</span>
          <span
            className="tabular relative inline-flex h-6 w-12 shrink-0 items-center justify-center rounded font-mono text-xs"
            style={{ backgroundColor: bg, color: fg }}
          >
            {theta.toFixed(2)}
          </span>
          <span className="relative truncate" title={skill.label_english ?? skill.label}>
            {displaySkillLabel(skill.label_english ?? skill.label)}
          </span>
          <span className="relative ml-auto shrink-0 text-xs text-muted-foreground">{skill.primary_benchmark}</span>
        </Link>
      </li>
    )
  }
  const hfUrl = model.hf_id
    ? `https://huggingface.co/${model.hf_id.replace('__', '/')}`
    : null

  return (
    <div className="page py-8">
      <nav aria-label="Breadcrumb" className="mb-6 text-sm text-muted-foreground">
        <Link to="/leaderboard" className="hover:text-foreground">
          Leaderboard
        </Link>
        <span className="mx-2">/</span>
        <span className="break-all">{model.name}</span>
      </nav>

      {/* header */}
      <header className="border-b border-border pb-6">
        <h1 className="break-words text-2xl font-semibold tracking-tight sm:text-3xl">
          {model.name}
        </h1>
        <p className="tabular mt-2 flex flex-wrap items-center gap-x-2 text-sm text-muted-foreground">
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: familyColor }} aria-hidden />
          <span>{model.family} family</span>
          <span aria-hidden>·</span>
          <span>{model.params != null ? formatParams(model.params) : 'size unknown'}</span>
          <span aria-hidden>·</span>
          <span>
            {model.accuracy != null ? `${(model.accuracy * 100).toFixed(1)}% of items correct` : 'accuracy unknown'}
          </span>
          <span aria-hidden>·</span>
          <span>
            mean mastery <span className="font-semibold text-foreground">{meanTheta.toFixed(3)}</span>
          </span>
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Link to={`/compare?m=${model.id}`} className={ACTION}>
            <GitCompareArrows className="h-3.5 w-3.5" />
            Compare with other models
          </Link>
          <button type="button" onClick={() => downloadProfile(model, skills)} className={ACTION}>
            <Download className="h-3.5 w-3.5" />
            Download profile (CSV)
          </button>
          {hfUrl ? (
            <a href={hfUrl} target="_blank" rel="noopener noreferrer" className={ACTION}>
              <ExternalLink className="h-3.5 w-3.5" />
              Hugging Face
            </a>
          ) : null}
        </div>
      </header>

      {/* fingerprint + ranked skills */}
      <div className="mt-8 flex flex-col gap-8 lg:flex-row">
        <div className="shrink-0 lg:sticky lg:top-20 lg:self-start">
          <div className="flex flex-col items-center rounded-lg border border-border bg-card p-6">
            <ModelFingerprint
              model={model}
              orderedSkills={orderedSkills}
              darkMode={darkMode}
              size={240}
            />
            <p className="mt-3 max-w-[240px] text-center text-[11px] leading-4 text-muted-foreground">
              One spoke per skill; longer and darker means higher mastery. The
              outer ring marks each skill's benchmark.
            </p>
          </div>
        </div>

        <div className="min-w-0 flex-1">
          <h2 className="text-lg font-semibold tracking-tight">Skills by mastery</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Mastery runs from 0 to 1. It is the model's estimated level on a
            skill, not the share of that skill's items it answered correctly.
          </p>
          {showAll ? (
            <ul className="mt-3 space-y-1">{ranked.map(renderRow)}</ul>
          ) : (
            <>
              <h3 className="mt-4 text-sm font-semibold">Strongest 10</h3>
              <ul className="mt-2 space-y-1">{ranked.slice(0, 10).map(renderRow)}</ul>
              <h3 className="mt-5 text-sm font-semibold">Weakest 10</h3>
              <ul className="mt-2 space-y-1">{ranked.slice(-10).map(renderRow)}</ul>
            </>
          )}
          <button
            type="button"
            onClick={() => setShowAll((v) => !v)}
            className="mt-4 text-sm font-medium text-brand hover:underline"
          >
            {showAll ? 'Show the strongest and weakest 10 only' : `Show all ${ranked.length} skills`}
          </button>
        </div>
      </div>
    </div>
  )
}
