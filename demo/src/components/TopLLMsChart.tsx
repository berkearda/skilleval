import { useMemo } from 'react'
import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { Model } from '@/lib/types'
import { getMasteryColor } from '@/lib/colors'

interface TopLLMsChartProps {
  models: Model[]
  skillId: number
  darkMode: boolean
}

interface ChartRow {
  name: string
  displayName: string
  theta: number
  family: string
  tier: string
  params: number | null
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s
  return `${s.slice(0, n - 1)}…`
}

function formatParams(p: number | null): string {
  if (p == null) return 'n/a'
  if (p >= 1_000) return `${(p / 1_000).toFixed(1)}B params`
  return `${p}M params`
}

export function TopLLMsChart({ models, skillId, darkMode }: TopLLMsChartProps) {
  const data = useMemo<ChartRow[]>(() => {
    const scored = models
      .map((m) => ({
        name: m.name,
        displayName: truncate(m.name, 36),
        theta: m.theta?.[skillId] ?? 0,
        family: m.family,
        tier: m.tier,
        params: m.params,
      }))
      .sort((a, b) => b.theta - a.theta)
      .slice(0, 10)
    // Recharts vertical bar charts render top-to-bottom in array order.
    // We want highest at top, so keep descending order.
    return scored
  }, [models, skillId])

  // Normalize bar color to the visible range so a narrow [0.86, 0.90] band
  // still shows visible variation, instead of all bars looking identical
  // on the absolute [0, 1] scale.
  const { minTheta, maxTheta } = useMemo(() => {
    if (data.length === 0) return { minTheta: 0, maxTheta: 1 }
    let mn = data[0].theta
    let mx = data[0].theta
    for (const r of data) {
      if (r.theta < mn) mn = r.theta
      if (r.theta > mx) mx = r.theta
    }
    return { minTheta: mn, maxTheta: mx }
  }, [data])

  const axisColor = darkMode ? '#94a3b8' : '#64748b'
  const gridColor = darkMode ? '#1f2937' : '#e2e8f0'

  return (
    <div className="h-[420px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={data}
          margin={{ top: 12, right: 56, left: 8, bottom: 12 }}
          barCategoryGap={6}
        >
          <XAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={(v: number) => v.toFixed(1)}
            stroke={axisColor}
            tick={{ fill: axisColor, fontSize: 11 }}
          />
          <YAxis
            type="category"
            dataKey="displayName"
            width={260}
            stroke={axisColor}
            tick={{ fill: axisColor, fontSize: 11 }}
            interval={0}
          />
          <Tooltip
            cursor={{ fill: gridColor, opacity: 0.4 }}
            contentStyle={{
              backgroundColor: darkMode ? '#0f172a' : '#ffffff',
              border: `1px solid ${gridColor}`,
              borderRadius: 6,
              fontSize: 12,
              color: darkMode ? '#f8fafc' : '#0f172a',
            }}
            labelStyle={{ color: darkMode ? '#f8fafc' : '#0f172a' }}
            formatter={((value: unknown, _name: unknown, item: unknown) => {
              const row = (item as { payload?: ChartRow } | undefined)?.payload
              const num = typeof value === 'number' ? value : Number(value)
              return [
                `θ = ${Number.isFinite(num) ? num.toFixed(3) : '—'}`,
                row
                  ? `${row.family} · ${row.tier} · ${formatParams(row.params)}`
                  : '',
              ] as [string, string]
            }) as never}
            labelFormatter={((_label: unknown, payload: unknown) => {
              const arr = payload as Array<{ payload?: ChartRow }> | undefined
              return arr?.[0]?.payload?.name ?? ''
            }) as never}
          />
          <Bar dataKey="theta" isAnimationActive={false}>
            {data.map((row, i) => {
              const range = maxTheta - minTheta
              // Avoid divide-by-zero when all 10 LLMs tie exactly.
              const relative =
                range > 1e-9 ? (row.theta - minTheta) / range : 0.5
              return (
                <Cell key={i} fill={getMasteryColor(relative, darkMode)} />
              )
            })}
            <LabelList
              dataKey="theta"
              position="right"
              formatter={(v: unknown) => {
                if (typeof v !== 'number') return ''
                return v.toFixed(3)
              }}
              style={{
                fill: darkMode ? '#e2e8f0' : '#0f172a',
                fontSize: 11,
                fontVariantNumeric: 'tabular-nums',
              }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
