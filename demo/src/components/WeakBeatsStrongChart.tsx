import { Link } from 'react-router-dom'
import { getBenchmarkColor } from '@/lib/colors'
import type { Skill, WeakBeatsStrong } from '@/lib/types'

// The paper's weak-beats-strong figure, redrawn with the site's 100 skill names
// so every row links to its skill page.

export function WeakBeatsStrongChart({
  wbs,
  skills,
  top = 12,
}: {
  wbs: WeakBeatsStrong
  skills: Skill[]
  top?: number
}) {
  const byId = new Map(skills.map((s) => [s.id, s]))
  const rows = wbs.skills.slice(0, top).filter((r) => byId.has(r.id))
  const benches = [...new Set(rows.map((r) => byId.get(r.id)!.primary_benchmark))]

  return (
    <div className="rounded-lg border border-border bg-card p-4 sm:p-5">
      <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {benches.map((b) => (
          <span key={b} className="inline-flex items-center gap-1.5">
            <span
              className="h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: getBenchmarkColor(b) }}
            />
            {b}
          </span>
        ))}
      </div>
      <ul className="space-y-2.5 sm:space-y-1.5">
        {rows.map((r) => {
          const s = byId.get(r.id)!
          const color = getBenchmarkColor(s.primary_benchmark)
          return (
            <li
              key={r.id}
              className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 text-sm sm:grid-cols-[minmax(0,18rem)_minmax(0,1fr)_7.5rem]"
            >
              <Link
                to={`/skill/${r.id}`}
                className="leading-snug text-foreground transition-colors hover:text-brand"
                title={s.label}
              >
                {s.label}
              </Link>
              <div className="relative order-3 col-span-2 h-2.5 rounded-full bg-muted sm:order-none sm:col-span-1">
                <div
                  className="absolute inset-y-0 left-0 rounded-full"
                  style={{ width: `${r.pct}%`, backgroundColor: color }}
                />
              </div>
              <span className="tabular text-right font-mono text-xs text-muted-foreground">
                <span className="font-semibold text-foreground">
                  {Math.round(r.pct)}%
                </span>{' '}
                ({r.wbs} of {r.n})
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
