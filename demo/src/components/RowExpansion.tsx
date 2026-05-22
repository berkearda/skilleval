import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { ExternalLink } from 'lucide-react'
import type { Model, Skill } from '@/lib/types'
import { getMasteryColor, getMasteryTextColor } from '@/lib/colors'

interface RowExpansionProps {
  model: Model
  skills: Skill[]
  darkMode: boolean
}

interface RankedSkill {
  skill: Skill
  theta: number
}

export function RowExpansion({ model, skills, darkMode }: RowExpansionProps) {
  const { top5, bottom5 } = useMemo(() => {
    const ranked: RankedSkill[] = skills.map((s) => ({
      skill: s,
      theta: model.theta[s.id] ?? 0,
    }))
    const sorted = [...ranked].sort((a, b) => b.theta - a.theta)
    return {
      top5: sorted.slice(0, 5),
      bottom5: sorted.slice(-5).reverse(),
    }
  }, [model, skills])

  return (
    <div className="grid grid-cols-1 gap-6 border-t border-border bg-muted/30 px-6 py-4 md:grid-cols-2">
      <SkillList title="Top 5 strongest skills" items={top5} darkMode={darkMode} />
      <SkillList title="Weakest 5 skills" items={bottom5} darkMode={darkMode} />
    </div>
  )
}

function SkillList({
  title,
  items,
  darkMode,
}: {
  title: string
  items: RankedSkill[]
  darkMode: boolean
}) {
  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </h3>
      <ul className="space-y-1">
        {items.map(({ skill, theta }) => {
          const bg = getMasteryColor(theta, darkMode)
          const fg = getMasteryTextColor(theta, darkMode)
          return (
            <li key={skill.id}>
              <Link
                to={`/skill/${skill.id}`}
                className="group flex items-center gap-2 rounded-md border border-border bg-background px-2 py-1.5 text-sm hover:bg-accent"
              >
                <span
                  className="inline-flex h-6 w-12 shrink-0 items-center justify-center rounded font-mono text-xs"
                  style={{ backgroundColor: bg, color: fg }}
                >
                  {theta.toFixed(2)}
                </span>
                <span className="truncate" title={skill.label}>
                  {skill.label}
                </span>
                <span className="ml-auto text-xs text-muted-foreground">
                  {skill.primary_benchmark}
                </span>
                <ExternalLink className="h-3 w-3 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
              </Link>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
