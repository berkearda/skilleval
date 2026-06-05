import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { GitCompareArrows, X } from 'lucide-react'
import type { Model, Skill } from '@/lib/types'
import { displaySkillLabel } from '@/lib/labels'
import { ModelFingerprint } from '@/components/ModelFingerprint'

// Compare 2-3 selected models: side-by-side fingerprints, a lead summary,
// and the skills where the selection diverges most, as 0-1 dot tracks.

export const COMPARE_COLORS = ['hsl(221 70% 50%)', '#D55E00', '#009E73']

export function CompareTray({
  models,
  onRemove,
  onClear,
  onOpen,
}: {
  models: Model[]
  onRemove: (id: number) => void
  onClear: () => void
  onOpen: () => void
}) {
  if (models.length === 0) return null
  return (
    <div className="fixed bottom-5 left-1/2 z-50 flex -translate-x-1/2 items-center gap-2 rounded-xl border border-border bg-card px-3 py-2 shadow-lg">
      <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
        Compare
      </span>
      {models.map((m, i) => (
        <span
          key={m.id}
          className="flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-1 text-xs"
        >
          <span
            className="h-2 w-2 rounded-full"
            style={{ backgroundColor: COMPARE_COLORS[i] }}
          />
          <span className="max-w-[18ch] truncate">{m.name}</span>
          <button
            type="button"
            onClick={() => onRemove(m.id)}
            className="text-muted-foreground hover:text-foreground"
            aria-label={`Remove ${m.name} from comparison`}
          >
            <X className="h-3 w-3" />
          </button>
        </span>
      ))}
      <button
        type="button"
        onClick={onOpen}
        disabled={models.length < 2}
        className="ml-1 inline-flex items-center gap-1.5 rounded-lg bg-brand px-3 py-1.5 text-xs font-medium text-brand-foreground shadow-sm transition-colors hover:bg-brand/90 disabled:cursor-not-allowed disabled:opacity-50"
      >
        <GitCompareArrows className="h-3.5 w-3.5" />
        {models.length < 2 ? 'Pick one more' : 'Compare'}
      </button>
      <button
        type="button"
        onClick={onClear}
        className="text-xs text-muted-foreground hover:text-foreground"
      >
        clear
      </button>
    </div>
  )
}

/** The comparison body, shared by the leaderboard modal and the /compare
 * page: lead summary, side-by-side fingerprints, divergence dot tracks. */
export function CompareView({
  models,
  orderedSkills,
  darkMode,
  onSkillClick,
}: {
  models: Model[]
  orderedSkills: Skill[]
  darkMode: boolean
  onSkillClick?: () => void
}) {
  // Skills ranked by spread across the selection.
  const diverging = useMemo(() => {
    return orderedSkills
      .map((s) => {
        const vals = models.map((m) => m.theta[s.id] ?? 0)
        return { skill: s, vals, spread: Math.max(...vals) - Math.min(...vals) }
      })
      .sort((a, b) => b.spread - a.spread)
      .slice(0, 10)
  }, [orderedSkills, models])

  // Pairwise lead summary (first two models).
  const lead = useMemo(() => {
    if (models.length < 2) return null
    const [a, b] = models
    let aWins = 0
    let bWins = 0
    for (const s of orderedSkills) {
      const av = a.theta[s.id] ?? 0
      const bv = b.theta[s.id] ?? 0
      if (av > bv) aWins++
      else if (bv > av) bWins++
    }
    return { aWins, bWins }
  }, [models, orderedSkills])

  return (
    <div>
      {lead ? (
        <p className="text-sm text-muted-foreground">
          {models[0].name} leads on {lead.aWins} of {orderedSkills.length}{' '}
          skills, {models[1].name} on {lead.bWins}.
        </p>
      ) : null}

      {/* fingerprints */}
      <div
        className="mt-4 grid gap-4"
        style={{ gridTemplateColumns: `repeat(${models.length}, 1fr)` }}
      >
        {models.map((m, i) => (
          <div
            key={m.id}
            className="flex flex-col items-center rounded-xl border border-border bg-surface p-4"
          >
            <ModelFingerprint
              model={m}
              orderedSkills={orderedSkills}
              darkMode={darkMode}
              size={170}
            />
            <div className="mt-2 flex items-center gap-1.5 text-sm font-medium">
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ backgroundColor: COMPARE_COLORS[i] }}
              />
              <span className="max-w-[20ch] truncate" title={m.name}>
                {m.name}
              </span>
            </div>
            <div className="tabular mt-0.5 text-xs text-muted-foreground">
              {m.accuracy != null
                ? `${(m.accuracy * 100).toFixed(1)}% accuracy`
                : ''}
            </div>
          </div>
        ))}
      </div>

      {/* divergence dot tracks */}
      <h3 className="mt-6 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
        Where they differ most
      </h3>
      <ul className="mt-3 space-y-2.5">
        {diverging.map(({ skill, vals }) => (
          <li key={skill.id} className="flex items-center gap-3">
            <Link
              to={`/skill/${skill.id}`}
              onClick={onSkillClick}
              className="w-[38%] truncate text-[13px] text-foreground/85 hover:text-brand"
              title={skill.label_english ?? skill.label}
            >
              {displaySkillLabel(skill.label_english ?? skill.label)}
            </Link>
            <div className="relative h-5 flex-1 rounded-full bg-surface-elevated/70">
              {/* range bar between min and max */}
              <span
                className="absolute inset-y-[7px] rounded-full bg-muted-foreground/25"
                style={{
                  left: `${Math.min(...vals) * 100}%`,
                  width: `${(Math.max(...vals) - Math.min(...vals)) * 100}%`,
                }}
              />
              {vals.map((v, i) => (
                <span
                  key={i}
                  className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-card"
                  style={{
                    left: `${v * 100}%`,
                    backgroundColor: COMPARE_COLORS[i],
                  }}
                  title={`${models[i].name}: ${v.toFixed(2)}`}
                />
              ))}
            </div>
            <span className="tabular w-20 shrink-0 text-right font-mono text-[11px] text-muted-foreground">
              Δ {(Math.max(...vals) - Math.min(...vals)).toFixed(2)}
            </span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-[11px] text-muted-foreground">
        Dots mark each model's mastery on the 0 to 1 scale; Δ is the spread
        across the selection. Small gaps sit inside estimation noise.
      </p>
    </div>
  )
}

export function CompareModal({
  models,
  orderedSkills,
  darkMode,
  onClose,
}: {
  models: Model[]
  orderedSkills: Skill[]
  darkMode: boolean
  onClose: () => void
}) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => {
    setMounted(true)
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Model comparison"
    >
      <div
        className="absolute inset-0 bg-black/45 backdrop-blur-[2px] transition-opacity"
        style={{ opacity: mounted ? 1 : 0 }}
        onClick={onClose}
      />
      <div
        className="relative max-h-[88vh] w-full max-w-3xl overflow-y-auto rounded-2xl border border-border bg-card p-6 shadow-2xl transition-all duration-200"
        style={{
          opacity: mounted ? 1 : 0,
          transform: mounted ? 'translateY(0)' : 'translateY(10px)',
        }}
      >
        <div className="flex items-start justify-between">
          <div>
            <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Skill profiles
            </div>
            <h2 className="mt-1 text-xl font-semibold tracking-tight">
              {models.map((m) => m.name).join('  vs  ')}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            aria-label="Close comparison"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="mt-3">
          <CompareView
            models={models}
            orderedSkills={orderedSkills}
            darkMode={darkMode}
            onSkillClick={onClose}
          />
        </div>
      </div>
    </div>
  )
}
