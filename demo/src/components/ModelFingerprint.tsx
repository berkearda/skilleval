import { useMemo } from 'react'
import type { Model, Skill } from '@/lib/types'
import { getBenchmarkColor, getMasteryColor } from '@/lib/colors'

// A model's 100-skill profile as a radial fingerprint: one spoke per skill
// in the grid's benchmark-grouped order, spoke length and color by mastery,
// with thin outer arcs marking the benchmark sectors. Two models are
// distinguishable at a glance the way two irises are.

export function ModelFingerprint({
  model,
  orderedSkills,
  darkMode,
  size = 180,
}: {
  model: Model
  orderedSkills: Skill[]
  darkMode: boolean
  size?: number
}) {
  const cx = size / 2
  const cy = size / 2
  const r0 = size * 0.16
  const rMax = size * 0.46
  const arcR = size * 0.485

  const spokes = useMemo(() => {
    const n = orderedSkills.length
    return orderedSkills.map((s, i) => {
      const theta = model.theta[s.id] ?? 0
      const a = (i / n) * Math.PI * 2 - Math.PI / 2
      const r1 = r0 + theta * (rMax - r0)
      return {
        x1: cx + r0 * Math.cos(a),
        y1: cy + r0 * Math.sin(a),
        x2: cx + r1 * Math.cos(a),
        y2: cy + r1 * Math.sin(a),
        color: getMasteryColor(theta, darkMode),
        key: s.id,
      }
    })
  }, [orderedSkills, model, darkMode, cx, cy, r0, rMax])

  // Benchmark sector arcs in grid order.
  const arcs = useMemo(() => {
    const n = orderedSkills.length
    const out: { d: string; color: string; key: string }[] = []
    let start = 0
    for (let i = 1; i <= n; i++) {
      const prev = orderedSkills[i - 1].primary_benchmark
      if (i === n || orderedSkills[i].primary_benchmark !== prev) {
        const a0 = (start / n) * Math.PI * 2 - Math.PI / 2 + 0.02
        const a1 = (i / n) * Math.PI * 2 - Math.PI / 2 - 0.02
        const large = a1 - a0 > Math.PI ? 1 : 0
        out.push({
          d: `M ${cx + arcR * Math.cos(a0)} ${cy + arcR * Math.sin(a0)} A ${arcR} ${arcR} 0 ${large} 1 ${cx + arcR * Math.cos(a1)} ${cy + arcR * Math.sin(a1)}`,
          color: getBenchmarkColor(prev),
          key: `${prev}-${start}`,
        })
        start = i
      }
    }
    return out
  }, [orderedSkills, cx, cy, arcR])

  const meanTheta = useMemo(() => {
    if (model.theta.length === 0) return 0
    let sum = 0
    for (const v of model.theta) sum += v
    return sum / model.theta.length
  }, [model])

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label={`Skill fingerprint of ${model.name}`}
      className="block"
    >
      {/* faint reference rings */}
      {[0.5, 1].map((f) => (
        <circle
          key={f}
          cx={cx}
          cy={cy}
          r={r0 + f * (rMax - r0)}
          fill="none"
          stroke="hsl(var(--border) / 0.6)"
          strokeWidth={1}
          strokeDasharray="2 4"
        />
      ))}
      {spokes.map((s) => (
        <line
          key={s.key}
          x1={s.x1}
          y1={s.y1}
          x2={s.x2}
          y2={s.y2}
          stroke={s.color}
          strokeWidth={size / 80}
          strokeLinecap="round"
        />
      ))}
      {arcs.map((a) => (
        <path
          key={a.key}
          d={a.d}
          fill="none"
          stroke={a.color}
          strokeWidth={2.5}
          strokeLinecap="round"
        />
      ))}
      <text
        x={cx}
        y={cy + size * 0.015}
        textAnchor="middle"
        className="tabular fill-foreground font-mono"
        style={{ fontSize: size * 0.085, fontWeight: 600 }}
      >
        {meanTheta.toFixed(3)}
      </text>
      <text
        x={cx}
        y={cy + size * 0.085}
        textAnchor="middle"
        className="fill-muted-foreground"
        style={{ fontSize: size * 0.05 }}
      >
        mean θ
      </text>
    </svg>
  )
}
