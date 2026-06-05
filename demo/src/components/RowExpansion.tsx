import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { Check, ExternalLink, GitCompareArrows } from 'lucide-react'
import type { Model, Skill } from '@/lib/types'
import { displaySkillLabel } from '@/lib/labels'
import { ModelFingerprint } from '@/components/ModelFingerprint'
import {
  getFamilyColor,
  getMasteryColor,
  getMasteryTextColor,
  rgbToRgba,
} from '@/lib/colors'

interface RowExpansionProps {
  model: Model
  skills: Skill[]
  darkMode: boolean
  /** Fixed panel height (px); the ranked skill list scrolls inside it. */
  height: number
  inCompare: boolean
  /** True when the compare set is full and this model is not in it. */
  compareFull: boolean
  onToggleCompare: (id: number) => void
}

interface RankedSkill {
  skill: Skill
  theta: number
  rank: number
}

function formatParams(p: number | null): string {
  if (p == null) return '—'
  if (p >= 1) return `${p.toFixed(p >= 10 ? 0 : 1)}B`
  return `${(p * 1000).toFixed(0)}M`
}

const kicker = 'text-xs font-medium uppercase tracking-wider text-muted-foreground'

function MetaStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className={kicker}>{label}</span>
      <span className="font-mono text-sm tabular text-foreground">{value}</span>
    </div>
  )
}

export function RowExpansion({
  model,
  skills,
  darkMode,
  height,
  inCompare,
  compareFull,
  onToggleCompare,
}: RowExpansionProps) {
  const ranked = useMemo<RankedSkill[]>(() => {
    return skills
      .map((s) => ({ skill: s, theta: model.theta[s.id] ?? 0 }))
      .sort((a, b) => b.theta - a.theta)
      .map((r, i) => ({ ...r, rank: i + 1 }))
  }, [model, skills])

  const meanTheta = useMemo(() => {
    if (model.theta.length === 0) return 0
    let sum = 0
    for (const v of model.theta) sum += v
    return sum / model.theta.length
  }, [model])

  const familyColor = getFamilyColor(model.family)
  const hfUrl = model.hf_id
    ? `https://huggingface.co/${model.hf_id.replace('__', '/')}`
    : null

  return (
    <div
      className="flex flex-col border-t border-border bg-surface"
      style={{ height, boxShadow: 'inset 2px 0 0 0 hsl(var(--brand))' }}
    >
      {/* Metadata header strip */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-border px-6 py-3">
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
      </div>

      <div className="flex min-h-0 flex-1">
        <div className="flex min-w-0 flex-1 flex-col">
      {/* Full ranked list */}
      <div className="flex items-baseline justify-between px-6 pb-1 pt-2">
        <span className={kicker}>All {ranked.length} skills, ranked by mastery</span>
        <span className="text-[11px] text-muted-foreground">
          scroll for more · click a skill to open its page
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-2">
        <ul className="space-y-1 pb-2">
          {ranked.map(({ skill, theta, rank }) => {
            const bg = getMasteryColor(theta, darkMode)
            const fg = getMasteryTextColor(theta, darkMode)
            const barColor = rgbToRgba(bg, darkMode ? 0.25 : 0.18)
            return (
              <li key={skill.id}>
                <Link
                  to={`/skill/${skill.id}`}
                  className="group relative flex items-center gap-2 overflow-hidden rounded-md border border-border bg-background px-2 py-1 text-sm hover:bg-surface-elevated"
                >
                  <span
                    aria-hidden
                    className="absolute inset-y-0 left-0"
                    style={{ width: `${theta * 100}%`, backgroundColor: barColor }}
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
                  <span className="relative truncate" title={skill.label}>
                    {displaySkillLabel(skill.label)}
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

        {/* Fingerprint rail */}
        <div className="hidden w-60 shrink-0 flex-col items-center justify-center gap-2.5 border-l border-border px-4 md:flex">
          <ModelFingerprint
            model={model}
            orderedSkills={skills}
            darkMode={darkMode}
            size={170}
          />
          <button
            type="button"
            onClick={() => onToggleCompare(model.id)}
            disabled={compareFull}
            title={
              compareFull
                ? 'Comparison is full (3 models); remove one first'
                : undefined
            }
            className={
              inCompare
                ? 'inline-flex items-center gap-1.5 rounded-lg border border-brand bg-brand/10 px-3 py-1.5 text-xs font-medium text-brand transition-colors hover:bg-brand/15'
                : 'inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-medium text-foreground shadow-sm transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50'
            }
          >
            {inCompare ? (
              <>
                <Check className="h-3.5 w-3.5" />
                In comparison
              </>
            ) : (
              <>
                <GitCompareArrows className="h-3.5 w-3.5" />
                Add to compare
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
