import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { ExternalLink } from 'lucide-react'
import type { Model, Skill } from '@/lib/types'
import { getMasteryColor, getMasteryTextColor } from '@/lib/colors'

interface RowExpansionProps {
  model: Model
  skills: Skill[]
  darkMode: boolean
  /** Fixed panel height (px); the ranked skill list scrolls inside it. */
  height: number
}

interface RankedSkill {
  skill: Skill
  theta: number
  rank: number
}

export function RowExpansion({
  model,
  skills,
  darkMode,
  height,
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

  return (
    <div
      className="flex flex-col border-t border-border bg-muted/30"
      style={{ height }}
    >
      {/* Summary strip */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-1 border-b border-border px-6 py-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          All {ranked.length} skills, ranked by mastery
        </span>
        <span className="font-mono text-xs text-foreground">
          mean θ {meanTheta.toFixed(3)}
        </span>
        {model.accuracy != null ? (
          <span className="font-mono text-xs text-foreground">
            accuracy {(model.accuracy * 100).toFixed(1)}%
          </span>
        ) : null}
        <span className="ml-auto text-[11px] text-muted-foreground">
          scroll for more · click a skill to open its page
        </span>
      </div>

      {/* Scrollable ranked list */}
      <div className="min-h-0 flex-1 overflow-y-auto px-6 py-2">
        <ul className="space-y-1 pb-2">
          {ranked.map(({ skill, theta, rank }) => {
            const bg = getMasteryColor(theta, darkMode)
            const fg = getMasteryTextColor(theta, darkMode)
            return (
              <li key={skill.id}>
                <Link
                  to={`/skill/${skill.id}`}
                  className="group flex items-center gap-2 rounded-md border border-border bg-background px-2 py-1 text-sm hover:bg-accent"
                >
                  <span className="w-7 shrink-0 text-right font-mono text-[11px] text-muted-foreground">
                    {rank}
                  </span>
                  <span
                    className="inline-flex h-6 w-12 shrink-0 items-center justify-center rounded font-mono text-xs"
                    style={{ backgroundColor: bg, color: fg }}
                  >
                    {theta.toFixed(2)}
                  </span>
                  <span className="truncate" title={skill.label}>
                    {skill.label}
                  </span>
                  <span className="ml-auto shrink-0 text-xs text-muted-foreground">
                    {skill.primary_benchmark}
                  </span>
                  <ExternalLink className="h-3 w-3 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
                </Link>
              </li>
            )
          })}
        </ul>
      </div>
    </div>
  )
}
