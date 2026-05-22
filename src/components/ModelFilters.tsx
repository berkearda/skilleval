import { Search } from 'lucide-react'
import { cn } from '@/lib/utils'
import {
  FAMILY_LIST,
  TIER_LIST,
  getFamilyColor,
} from '@/lib/colors'

interface ModelFiltersProps {
  search: string
  onSearchChange: (s: string) => void
  selectedFamilies: Set<string>
  onToggleFamily: (family: string) => void
  selectedTiers: Set<string>
  onToggleTier: (tier: string) => void
  visibleCount: number
  totalCount: number
}

export function ModelFilters({
  search,
  onSearchChange,
  selectedFamilies,
  onToggleFamily,
  selectedTiers,
  onToggleTier,
  visibleCount,
  totalCount,
}: ModelFiltersProps) {
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-border bg-background px-6 py-3">
      {/* Search */}
      <div className="relative w-64">
        <Search
          className="pointer-events-none absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
          aria-hidden
        />
        <input
          type="text"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search model..."
          className="h-9 w-full rounded-md border border-input bg-background pl-8 pr-3 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        />
      </div>

      {/* Family chips */}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Family
        </span>
        {FAMILY_LIST.map((fam) => {
          const active = selectedFamilies.has(fam)
          const color = getFamilyColor(fam)
          return (
            <button
              key={fam}
              type="button"
              onClick={() => onToggleFamily(fam)}
              className={cn(
                'h-7 rounded-full border px-2.5 text-xs font-medium transition-colors',
                active
                  ? 'text-white shadow-sm'
                  : 'border-border bg-background text-foreground hover:bg-accent'
              )}
              style={
                active
                  ? { backgroundColor: color, borderColor: color }
                  : undefined
              }
              aria-pressed={active}
            >
              {fam}
            </button>
          )
        })}
      </div>

      {/* Tier checkboxes */}
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Tier
        </span>
        {TIER_LIST.map((tier) => {
          const checked = selectedTiers.has(tier)
          return (
            <label
              key={tier}
              className="flex cursor-pointer select-none items-center gap-1.5 text-sm"
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => onToggleTier(tier)}
                className="h-4 w-4 rounded border-input accent-foreground"
              />
              <span>{tier}</span>
            </label>
          )
        })}
      </div>

      {/* Count display */}
      <div className="ml-auto text-sm text-muted-foreground">
        Showing{' '}
        <span className="font-semibold text-foreground">
          {visibleCount.toLocaleString()}
        </span>{' '}
        of {totalCount.toLocaleString()} models
      </div>
    </div>
  )
}
