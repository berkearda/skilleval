import { useEffect, useRef, useState } from 'react'
import { Check, ChevronDown, Search } from 'lucide-react'
import { cn } from '@/lib/utils'
import { FAMILY_LIST, TIER_LIST, getFamilyColor } from '@/lib/colors'

type ValueMode = 'abs' | 'rel'

interface ModelFiltersProps {
  search: string
  onSearchChange: (s: string) => void
  selectedFamilies: Set<string>
  onToggleFamily: (family: string) => void
  selectedTiers: Set<string>
  onToggleTier: (tier: string) => void
  visibleCount: number
  totalCount: number
  valueMode: ValueMode
  onValueModeChange: (m: ValueMode) => void
  onClearAll: () => void
}

// The size tiers as they occur in the data (parameters in billions).
const TIER_HINT: Record<string, string> = {
  Small: 'under 4B',
  Mid: '7B to 14B',
  Large: '27B and up',
  Other: 'size unknown or in between',
}

/** A button that opens a short checklist; several options can be ticked. */
function MultiSelect({
  label,
  options,
  selected,
  onToggle,
  hint,
  dot,
}: {
  label: string
  options: readonly string[]
  selected: Set<string>
  onToggle: (o: string) => void
  hint?: Record<string, string>
  dot?: (o: string) => string
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const summary =
    selected.size === 0 ? 'All' : selected.size === 1 ? [...selected][0] : `${selected.size} selected`

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className={cn(
          'inline-flex h-9 items-center gap-1.5 rounded-md border border-input bg-background px-3 text-sm transition-colors hover:bg-surface-elevated',
          selected.size > 0 && 'border-brand/60'
        )}
      >
        <span className="text-muted-foreground">{label}:</span>
        <span className="text-foreground">{summary}</span>
        <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
      </button>
      {open ? (
        <div className="absolute left-0 top-full z-40 mt-1 min-w-[13rem] rounded-md border border-border bg-popover py-1 shadow-md">
          {options.map((o) => {
            const on = selected.has(o)
            return (
              <button
                key={o}
                type="button"
                onClick={() => onToggle(o)}
                className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm hover:bg-accent"
              >
                <span
                  className={cn(
                    'flex h-4 w-4 shrink-0 items-center justify-center rounded-sm border',
                    on ? 'border-brand bg-brand text-brand-foreground' : 'border-input'
                  )}
                >
                  {on ? <Check className="h-3 w-3" /> : null}
                </span>
                {dot ? (
                  <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: dot(o) }} />
                ) : null}
                <span className="text-foreground">{o}</span>
                {hint?.[o] ? <span className="ml-auto pl-3 text-xs text-muted-foreground">{hint[o]}</span> : null}
              </button>
            )
          })}
        </div>
      ) : null}
    </div>
  )
}

function ValueModeControl({ value, onChange }: { value: ValueMode; onChange: (m: ValueMode) => void }) {
  const opts: Array<[ValueMode, string, string]> = [
    ['abs', 'Mastery', 'Mastery on each skill, from 0 to 1'],
    ['rel', 'vs. average', 'Mastery minus the average of all models on that skill'],
  ]
  return (
    <div className="inline-flex h-9 items-center rounded-md border border-input p-0.5">
      {opts.map(([v, label, hint]) => (
        <button
          key={v}
          type="button"
          onClick={() => onChange(v)}
          aria-pressed={value === v}
          title={hint}
          className={cn(
            'h-full rounded px-2.5 text-sm transition-colors duration-150',
            value === v ? 'bg-surface-elevated font-medium text-foreground' : 'text-muted-foreground hover:text-foreground'
          )}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

/** One row of controls: search, family, size, value mode, and the count. */
export function ModelFilters({
  search,
  onSearchChange,
  selectedFamilies,
  onToggleFamily,
  selectedTiers,
  onToggleTier,
  visibleCount,
  totalCount,
  valueMode,
  onValueModeChange,
  onClearAll,
}: ModelFiltersProps) {
  const hasAnyFilter = search.trim() !== '' || selectedFamilies.size > 0 || selectedTiers.size > 0

  return (
    <div className="flex flex-wrap items-center gap-2 py-3">
      <div className="relative w-full sm:w-64">
        <Search
          className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
          aria-hidden
        />
        <input
          type="text"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search models"
          className="h-9 w-full rounded-md border border-input bg-background pl-8 pr-3 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        />
      </div>
      <MultiSelect
        label="Family"
        options={FAMILY_LIST}
        selected={selectedFamilies}
        onToggle={onToggleFamily}
        dot={getFamilyColor}
      />
      <MultiSelect label="Size" options={TIER_LIST} selected={selectedTiers} onToggle={onToggleTier} hint={TIER_HINT} />
      <ValueModeControl value={valueMode} onChange={onValueModeChange} />
      <div className="ml-auto flex items-center gap-3 text-sm text-muted-foreground">
        {hasAnyFilter ? (
          <button type="button" onClick={onClearAll} className="text-brand hover:underline">
            Clear filters
          </button>
        ) : null}
        <span className="tabular">
          {visibleCount === totalCount
            ? `${totalCount.toLocaleString()} models`
            : `${visibleCount.toLocaleString()} of ${totalCount.toLocaleString()} models`}
        </span>
      </div>
    </div>
  )
}
