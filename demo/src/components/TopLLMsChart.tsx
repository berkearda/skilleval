import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import type { Model } from '@/lib/types'
import { getMasteryColor } from '@/lib/colors'

interface TopLLMsChartProps {
  models: Model[]
  skillId: number
  darkMode: boolean
}

function formatParams(p: number | null): string {
  if (p == null) return ''
  if (p >= 1) return `${p.toFixed(p >= 10 ? 0 : 1)}B`
  return `${(p * 1000).toFixed(0)}M`
}

/** The ten models with the highest mastery on one skill, as bars on the full
 * 0-1 scale: near-identical values should look near-identical. Plain HTML, so
 * it stays readable on a phone. */
export function TopLLMsChart({ models, skillId, darkMode }: TopLLMsChartProps) {
  const rows = useMemo(
    () =>
      models
        .map((m) => ({ m, theta: m.theta?.[skillId] ?? 0 }))
        .sort((a, b) => b.theta - a.theta)
        .slice(0, 10),
    [models, skillId]
  )

  return (
    <ol className="space-y-2.5 sm:space-y-1.5">
      {rows.map(({ m, theta }, i) => (
        <li
          key={m.id}
          className="grid grid-cols-[1.5rem_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 text-sm sm:grid-cols-[1.5rem_minmax(0,17rem)_minmax(0,1fr)_3.5rem]"
        >
          <span className="tabular text-right font-mono text-xs text-muted-foreground sm:order-1">
            {i + 1}
          </span>
          <Link
            to={`/model/${m.id}`}
            className="truncate text-foreground transition-colors hover:text-brand sm:order-2"
            title={`${m.name}${m.params != null ? ` · ${formatParams(m.params)}` : ''} · ${m.family}`}
          >
            {m.name}
          </Link>
          <span className="tabular text-right font-mono text-xs text-foreground sm:order-4">
            {theta.toFixed(2)}
          </span>
          <div className="relative order-last col-span-2 col-start-2 h-2.5 rounded-full bg-muted sm:order-3 sm:col-span-1 sm:col-start-auto">
            <div
              className="absolute inset-y-0 left-0 rounded-full"
              style={{ width: `${theta * 100}%`, backgroundColor: getMasteryColor(theta, darkMode) }}
            />
          </div>
        </li>
      ))}
    </ol>
  )
}
