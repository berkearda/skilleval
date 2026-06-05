import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Model, Skill } from '@/lib/types'
import { displaySkillLabel } from '@/lib/labels'
import { getBenchmarkColor } from '@/lib/colors'

// The capability space: every skill positioned by PCA over its mastery
// pattern across all 3,811 models (computed offline, scripts/ in the repo).
// Skills mastered by the same models sit near each other. Benchmark identity
// is NOT an input to the layout, which is why the clusters are a finding.

interface MapPoint {
  id: number
  x: number
  y: number
}

const W = 1000
const H = 600
const PAD = 52

function radiusFor(nItems: number): number {
  // sqrt-area scaling, clamped so tiny clusters stay visible
  return Math.max(5, Math.min(17, Math.sqrt(nItems) * 0.42))
}

export function SkillMap({
  skills,
  models,
  darkMode,
}: {
  skills: Skill[]
  models: Model[]
  darkMode: boolean
}) {
  const navigate = useNavigate()
  const [points, setPoints] = useState<MapPoint[] | null>(null)
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    let cancelled = false
    void fetch(`${import.meta.env.BASE_URL}data/skill_map.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: MapPoint[] | null) => {
        if (!cancelled && d) setPoints(d)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const skillById = useMemo(() => {
    const m = new Map<number, Skill>()
    for (const s of skills) m.set(s.id, s)
    return m
  }, [skills])

  // Best model per skill, computed once from the loaded matrix.
  const topModel = useMemo(() => {
    const out = new Map<number, { name: string; theta: number }>()
    if (models.length === 0) return out
    for (const s of skills) {
      let best = -1
      let bestName = ''
      for (const m of models) {
        const v = m.theta[s.id] ?? 0
        if (v > best) {
          best = v
          bestName = m.name
        }
      }
      out.set(s.id, { name: bestName, theta: best })
    }
    return out
  }, [models, skills])

  const benches = ['MATH', 'BBH', 'GPQA', 'IFEval', 'MuSR']

  if (!points || skills.length === 0) {
    return <div className="h-[420px] w-full animate-pulse rounded-xl bg-muted" />
  }

  const px = (x: number) => PAD + x * (W - 2 * PAD)
  const py = (y: number) => PAD + (1 - y) * (H - 2 * PAD)

  const hovered = hover != null ? skillById.get(hover) : null
  const hoveredPt = hover != null ? points.find((p) => p.id === hover) : null

  return (
    <div className="relative overflow-hidden rounded-xl border border-border bg-card">
      {/* legend */}
      <div className="pointer-events-none absolute right-3 top-3 z-10 flex flex-wrap gap-x-3 gap-y-1 rounded-md bg-card/85 px-2 py-1 backdrop-blur-sm">
        {benches.map((b) => (
          <span key={b} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <span
              className="h-2 w-2 rounded-full"
              style={{ backgroundColor: getBenchmarkColor(b) }}
            />
            {b}
          </span>
        ))}
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="block h-auto w-full"
        role="img"
        aria-label="Map of 100 skills positioned by mastery similarity"
        onMouseLeave={() => setHover(null)}
      >
        {/* faint grid for map feel */}
        {Array.from({ length: 5 }, (_, i) => (
          <line
            key={`v${i}`}
            x1={(W / 5) * (i + 0.5)}
            x2={(W / 5) * (i + 0.5)}
            y1={0}
            y2={H}
            stroke="hsl(var(--border) / 0.5)"
            strokeDasharray="2 6"
          />
        ))}
        {Array.from({ length: 3 }, (_, i) => (
          <line
            key={`h${i}`}
            y1={(H / 3) * (i + 0.5)}
            y2={(H / 3) * (i + 0.5)}
            x1={0}
            x2={W}
            stroke="hsl(var(--border) / 0.5)"
            strokeDasharray="2 6"
          />
        ))}

        {points.map((p, i) => {
          const s = skillById.get(p.id)
          if (!s) return null
          const r = radiusFor(s.n_items)
          const active = hover === p.id
          return (
            <circle
              key={p.id}
              cx={px(p.x)}
              cy={py(p.y)}
              r={active ? r + 3 : r}
              fill={getBenchmarkColor(s.primary_benchmark)}
              fillOpacity={hover == null ? 0.82 : active ? 0.95 : 0.25}
              stroke={darkMode ? 'hsl(222 26% 10%)' : '#ffffff'}
              strokeWidth={1.5}
              className="cursor-pointer"
              style={{
                transition: 'fill-opacity 150ms ease, r 150ms ease',
                animation: `mapdot 420ms ${i * 14}ms both ease-out`,
                transformOrigin: `${px(p.x)}px ${py(p.y)}px`,
              }}
              onMouseEnter={() => setHover(p.id)}
              onClick={() => navigate(`/skill/${p.id}`)}
            />
          )
        })}
      </svg>

      {/* tooltip */}
      {hovered && hoveredPt ? (
        <div
          className="pointer-events-none absolute z-20 w-60 rounded-lg border border-border bg-popover p-3 shadow-lg"
          style={{
            left: `calc(${(px(hoveredPt.x) / W) * 100}% ${
              hoveredPt.x > 0.62 ? '- 15.5rem' : '+ 1rem'
            })`,
            top: `${(py(hoveredPt.y) / H) * 100}%`,
            transform: 'translateY(-50%)',
          }}
        >
          <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{
                backgroundColor: getBenchmarkColor(hovered.primary_benchmark),
              }}
            />
            {hovered.primary_benchmark} · {hovered.n_items} items
          </div>
          <div className="mt-1 text-sm font-medium leading-5 text-foreground">
            {displaySkillLabel(hovered.label_english ?? hovered.label)}
          </div>
          {topModel.get(hovered.id) ? (
            <div className="mt-1.5 truncate text-xs text-muted-foreground">
              best: {topModel.get(hovered.id)!.name} (
              {topModel.get(hovered.id)!.theta.toFixed(2)})
            </div>
          ) : null}
          <div className="mt-1 text-[11px] text-brand">click to open</div>
        </div>
      ) : null}
    </div>
  )
}
