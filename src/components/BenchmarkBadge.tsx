import { getBenchmarkColor } from '@/lib/colors'

interface BenchmarkBadgeProps {
  benchmark: string
  count?: number
}

export function BenchmarkBadge({ benchmark, count }: BenchmarkBadgeProps) {
  const color = getBenchmarkColor(benchmark)
  return (
    <div
      className="flex h-7 items-center justify-center gap-1.5 px-2 text-xs font-semibold tracking-wide"
      style={{
        color,
        backgroundColor: `${color}14`,
        boxShadow: `inset 0 -2px 0 0 ${color}`,
      }}
    >
      <span
        aria-hidden
        className="h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: color }}
      />
      {benchmark}
      {count !== undefined ? (
        <span className="font-normal opacity-80">({count})</span>
      ) : null}
    </div>
  )
}
