import { useState } from 'react'
import type { ExampleItem } from '@/lib/types'
import { getBenchmarkColor } from '@/lib/colors'
import { cn } from '@/lib/utils'

interface ExampleItemCardProps {
  item: ExampleItem
}

// Questions longer than this fold behind "Show more".
const FOLD_LINES = 7
const FOLD_CHARS = 450

export function ExampleItemCard({ item }: ExampleItemCardProps) {
  const [open, setOpen] = useState(false)
  const long = item.text.split('\n').length > FOLD_LINES || item.text.length > FOLD_CHARS
  const subtask = item.subtask && item.subtask !== item.benchmark ? item.subtask.replace(/_/g, ' ') : null
  return (
    <div className="rounded-md border border-border bg-card px-4 py-3">
      {/* one info line: benchmark, subtask, item number */}
      <div className="flex flex-wrap items-center gap-x-2 text-xs text-muted-foreground">
        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: getBenchmarkColor(item.benchmark) }} />
        <span className="font-medium text-foreground/80">{item.benchmark}</span>
        {subtask ? (
          <>
            <span aria-hidden>·</span>
            <span>{subtask}</span>
          </>
        ) : null}
        <span aria-hidden>·</span>
        <span className="tabular">item #{item.item_idx}</span>
      </div>
      <div className="relative mt-2">
        <p
          className={cn(
            'whitespace-pre-wrap text-sm leading-relaxed text-foreground',
            long && !open && 'max-h-[8.75rem] overflow-hidden'
          )}
        >
          {item.text}
        </p>
        {long && !open ? (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-card to-transparent" />
        ) : null}
      </div>
      {long ? (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="mt-1 text-xs font-medium text-brand hover:underline"
        >
          {open ? 'Show less' : 'Show more'}
        </button>
      ) : null}
    </div>
  )
}
