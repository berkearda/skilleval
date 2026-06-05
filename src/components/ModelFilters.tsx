import { Search, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import {
  FAMILY_LIST,
  TIER_LIST,
  getFamilyColor,
} from '@/lib/colors'

type Density = 'compact' | 'regular' | 'relaxed'
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
  density: Density
  onDensityChange: (d: Density) => void
  valueMode: ValueMode
  onValueModeChange: (m: ValueMode) => void
  onClearAll: () => void
}

function DensityControl({
  value,
  onChange,
}: {
  value: Density
  onChange: (d: Density) => void
}) {
  const opts: Array<[Density, string]> = [
    ['compact', 'Compact'],
    ['regular', 'Regular'],
    ['relaxed', 'Relaxed'],
  ]
  return (
    <div className="inline-flex items-center rounded-md border border-border p-0.5">
      {opts.map(([v, label]) => (
        <button
          key={v}
          type="button"
          onClick={() => onChange(v)}
          aria-pressed={value === v}
          className={cn(
            'rounded px-2.5 py-1 text-xs font-medium transition-colors duration-150',
            value === v
              ? 'bg-surface-elevated text-foreground'
              : 'text-muted-foreground hover:text-foreground'
          )}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

function ValueModeControl({
  value,
  onChange,
}: {
  value: ValueMode
  onChange: (m: ValueMode) => void
}) {
  const opts: Array<[ValueMode, string, string]> = [
    ['abs', 'Absolute', 'Raw mastery theta in [0, 1]'],
    ['rel', 'Relative', 'Theta minus the population mean on each skill'],
  ]
  return (
    <div className="inline-flex items-center rounded-md border border-border p-0.5">
      {opts.map(([v, label, hint]) => (
        <button
          key={v}
          type="button"
          onClick={() => onChange(v)}
          aria-pressed={value === v}
          title={hint}
          className={cn(
            'rounded px-2.5 py-1 text-xs font-medium transition-colors duration-150',
            value === v
              ? 'bg-brand text-brand-foreground'
              : 'text-muted-foreground hover:text-foreground'
          )}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

function FilterChip({
  label,
  color,
  onRemove,
}: {
  label: string
  color?: string
  onRemove: () => void
}) {
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full border border-border bg-surface px-2 py-0.5"
      style={color ? { borderColor: color } : undefined}
    >
      {color && (
        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      )}
      <span className="text-foreground">{label}</span>
      <button
        type="button"
        onClick={onRemove}
        aria-label={`Remove ${label} filter`}
        className="text-muted-foreground hover:text-foreground"
      >
        <X className="h-3 w-3" />
      </button>
    </span>
  )
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
  density,
  onDensityChange,
  valueMode,
  onValueModeChange,
  onClearAll,
}: ModelFiltersProps) {
  const hasAnyFilter =
    search.trim() !== '' || selectedFamilies.size > 0 || selectedTiers.size > 0

  return (
    <div className="bg-background">
      {/* Row 1: controls */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 px-6 py-3">
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
          <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
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
          <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
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

        {/* Value mode + density controls */}
        <div className="ml-auto flex items-center gap-2">
          <ValueModeControl value={valueMode} onChange={onValueModeChange} />
          <DensityControl value={density} onChange={onDensityChange} />
        </div>
      </div>

      {/* Row 2: applied-filter chip strip + count */}
      <div className="flex flex-wrap items-center gap-1.5 border-t border-border bg-background px-6 py-2 text-xs">
        <span className="text-muted-foreground tabular">
          {visibleCount.toLocaleString()} of {totalCount.toLocaleString()} models
        </span>
        {hasAnyFilter && <span className="text-border">·</span>}
        {search.trim() && (
          <FilterChip
            label={`Search: ${search.trim()}`}
            onRemove={() => onSearchChange('')}
          />
        )}
        {[...selectedFamilies].map((f) => (
          <FilterChip
            key={f}
            label={f}
            color={getFamilyColor(f)}
            onRemove={() => onToggleFamily(f)}
          />
        ))}
        {[...selectedTiers].map((t) => (
          <FilterChip key={t} label={t} onRemove={() => onToggleTier(t)} />
        ))}
        {hasAnyFilter && (
          <button
            type="button"
            onClick={onClearAll}
            className="ml-1 rounded px-1.5 py-0.5 text-muted-foreground hover:text-foreground"
          >
            Clear all
          </button>
        )}
      </div>
    </div>
  )
}
