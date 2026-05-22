import { getBenchmarkColor } from '@/lib/colors'

interface BenchmarkBadgeProps {
  benchmark: string
  count?: number
}

export function BenchmarkBadge({ benchmark, count }: BenchmarkBadgeProps) {
  const color = getBenchmarkColor(benchmark)
  return (
    <div
      className="flex h-7 items-center justify-center px-2 text-xs font-semibold tracking-wide text-white"
      style={{ backgroundColor: color }}
    >
      {benchmark}
      {count !== undefined ? (
        <span className="ml-1 font-normal opacity-90">({count})</span>
      ) : null}
    </div>
  )
}
