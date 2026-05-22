import type { ExampleItem } from '@/lib/types'
import { BenchmarkBadge } from '@/components/BenchmarkBadge'

interface ExampleItemCardProps {
  item: ExampleItem
}

export function ExampleItemCard({ item }: ExampleItemCardProps) {
  return (
    <div className="rounded-md border border-border bg-card p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          {item.subtask ? (
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              {item.subtask}
            </div>
          ) : null}
          <div className="mt-1 text-xs text-muted-foreground">
            item #{item.item_idx}
          </div>
        </div>
        <div className="shrink-0">
          <BenchmarkBadge benchmark={item.benchmark} />
        </div>
      </div>
      <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-foreground">
        {item.text}
      </p>
    </div>
  )
}
