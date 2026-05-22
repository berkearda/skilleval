import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { Link } from 'react-router-dom'
import { ArrowDown, ArrowUp, ChevronsUpDown, ExternalLink } from 'lucide-react'
import { useVirtualizer } from '@tanstack/react-virtual'
import type { Model, Skill } from '@/lib/types'
import { cn } from '@/lib/utils'
import { getFamilyColor, getMasteryColor, getMasteryTextColor } from '@/lib/colors'
import { RowExpansion } from '@/components/RowExpansion'
import { ModelFilters } from '@/components/ModelFilters'
import { SkillCell } from '@/components/SkillCell'
import { BenchmarkBadge } from '@/components/BenchmarkBadge'

interface ModelSkillTableProps {
  models: Model[]
  skills: Skill[]
  darkMode: boolean
}

type SortKey =
  | { kind: 'index' }
  | { kind: 'name' }
  | { kind: 'family' }
  | { kind: 'tier' }
  | { kind: 'params' }
  | { kind: 'mean' }
  | { kind: 'skill'; skillId: number }

type SortDir = 'asc' | 'desc'

interface SortState {
  key: SortKey
  dir: SortDir
}

// Column widths (px). Adjust here if visual tuning is needed.
const COL_W = {
  rank: 48,
  name: 280,
  family: 96,
  tier: 72,
  params: 72,
  mean: 80,
  skill: 72,
} as const

// Right-edge inset shadow on the pinned cluster — provides a visual hint
// that more columns exist to the right.
const PINNED_RIGHT_SHADOW = '2px 0 0 0 hsl(var(--border)), inset -8px 0 8px -8px rgba(0,0,0,0.18)'

const PINNED_WIDTHS = [
  COL_W.rank,
  COL_W.name,
  COL_W.family,
  COL_W.tier,
  COL_W.params,
  COL_W.mean,
]

const PINNED_TOTAL_W = PINNED_WIDTHS.reduce((a, b) => a + b, 0)

const ROW_HEIGHT = 32
const EXPANDED_PANEL_HEIGHT = 220
const GROUP_HEADER_H = 28
const SKILL_HEADER_H = 56
const COL_HEADER_H = GROUP_HEADER_H + SKILL_HEADER_H

interface ProcessedModel extends Model {
  meanTheta: number
}

interface SortableSkillGroup {
  benchmark: string
  skills: Skill[]
  startIndex: number // index into ordered skill list
}

// Convert an `rgb(r, g, b)` string to `rgba(r, g, b, a)` for translucency.
function rgbToRgba(rgb: string, alpha: number): string {
  const m = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/.exec(rgb)
  if (!m) return rgb
  return `rgba(${m[1]}, ${m[2]}, ${m[3]}, ${alpha})`
}

function formatParams(p: number | null): string {
  if (p == null) return '—'
  if (p >= 1) return `${p.toFixed(p >= 10 ? 0 : 1)}B`
  return `${(p * 1000).toFixed(0)}M`
}

function sortKeysEqual(a: SortKey, b: SortKey): boolean {
  if (a.kind !== b.kind) return false
  if (a.kind === 'skill' && b.kind === 'skill') {
    return a.skillId === b.skillId
  }
  return true
}

export function ModelSkillTable({
  models,
  skills,
  darkMode,
}: ModelSkillTableProps) {
  // Pre-compute mean theta per model once.
  const processedModels = useMemo<ProcessedModel[]>(() => {
    return models.map((m) => {
      let sum = 0
      for (let i = 0; i < m.theta.length; i++) sum += m.theta[i]
      return { ...m, meanTheta: m.theta.length ? sum / m.theta.length : 0 }
    })
  }, [models])

  // Group skills by primary_benchmark while preserving an ordered skill list
  // that drives the column order.
  const { orderedSkills, groups } = useMemo(() => {
    const byBench = new Map<string, Skill[]>()
    for (const s of skills) {
      const b = s.primary_benchmark
      if (!byBench.has(b)) byBench.set(b, [])
      byBench.get(b)!.push(s)
    }
    // Stable benchmark order: MATH, BBH, GPQA, IFEval, MuSR, then anything else.
    const order = ['MATH', 'BBH', 'GPQA', 'IFEval', 'MuSR']
    const allBenches = [
      ...order.filter((b) => byBench.has(b)),
      ...[...byBench.keys()].filter((b) => !order.includes(b)),
    ]
    const ordered: Skill[] = []
    const groupArr: SortableSkillGroup[] = []
    for (const b of allBenches) {
      const list = byBench.get(b)!
      groupArr.push({ benchmark: b, skills: list, startIndex: ordered.length })
      ordered.push(...list)
    }
    return { orderedSkills: ordered, groups: groupArr }
  }, [skills])

  // Filter state
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search), 80)
    return () => clearTimeout(t)
  }, [search])

  const [selectedFamilies, setSelectedFamilies] = useState<Set<string>>(
    new Set()
  )
  const [selectedTiers, setSelectedTiers] = useState<Set<string>>(new Set())

  // Sort state
  const [sort, setSort] = useState<SortState>({
    key: { kind: 'mean' },
    dir: 'desc',
  })

  // Expanded row id (model.id)
  const [expandedId, setExpandedId] = useState<number | null>(null)

  // Filter models.
  const filteredModels = useMemo(() => {
    const q = debouncedSearch.trim().toLowerCase()
    return processedModels.filter((m) => {
      if (q && !m.name.toLowerCase().includes(q)) return false
      if (selectedFamilies.size > 0 && !selectedFamilies.has(m.family))
        return false
      if (selectedTiers.size > 0 && !selectedTiers.has(m.tier)) return false
      return true
    })
  }, [processedModels, debouncedSearch, selectedFamilies, selectedTiers])

  // Sort filtered models.
  const sortedModels = useMemo(() => {
    const arr = filteredModels.slice()
    const dirMul = sort.dir === 'asc' ? 1 : -1
    const key = sort.key
    const cmpStr = (a: string, b: string) =>
      a < b ? -1 : a > b ? 1 : 0
    arr.sort((a, b) => {
      switch (key.kind) {
        case 'index':
          return (a.id - b.id) * dirMul
        case 'name':
          return cmpStr(a.name.toLowerCase(), b.name.toLowerCase()) * dirMul
        case 'family':
          return cmpStr(a.family, b.family) * dirMul
        case 'tier':
          return cmpStr(a.tier, b.tier) * dirMul
        case 'params': {
          const av = a.params ?? -Infinity
          const bv = b.params ?? -Infinity
          return (av - bv) * dirMul
        }
        case 'mean':
          return (a.meanTheta - b.meanTheta) * dirMul
        case 'skill': {
          const av = a.theta[key.skillId] ?? 0
          const bv = b.theta[key.skillId] ?? 0
          return (av - bv) * dirMul
        }
      }
    })
    return arr
  }, [filteredModels, sort])

  // When a row is sorted/filtered, drop the expansion so we don't show a
  // stale expanded row that's no longer visible.
  useEffect(() => {
    if (expandedId == null) return
    const stillVisible = sortedModels.some((m) => m.id === expandedId)
    if (!stillVisible) setExpandedId(null)
  }, [sortedModels, expandedId])

  // Filter / sort handlers
  const toggleFamily = useCallback((fam: string) => {
    setSelectedFamilies((prev) => {
      const next = new Set(prev)
      if (next.has(fam)) next.delete(fam)
      else next.add(fam)
      return next
    })
  }, [])
  const toggleTier = useCallback((tier: string) => {
    setSelectedTiers((prev) => {
      const next = new Set(prev)
      if (next.has(tier)) next.delete(tier)
      else next.add(tier)
      return next
    })
  }, [])

  const toggleSort = useCallback((key: SortKey) => {
    setSort((prev) => {
      if (sortKeysEqual(prev.key, key)) {
        return { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' }
      }
      // Sensible defaults: numeric / mastery columns default to descending,
      // string columns default to ascending.
      const numericKinds = ['mean', 'skill', 'params', 'index']
      const dir: SortDir = numericKinds.includes(key.kind) ? 'desc' : 'asc'
      return { key, dir }
    })
  }, [])

  // Virtualization
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const rowVirtualizer = useVirtualizer({
    count: sortedModels.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: (i) => {
      const id = sortedModels[i]?.id
      return id === expandedId ? ROW_HEIGHT + EXPANDED_PANEL_HEIGHT : ROW_HEIGHT
    },
    overscan: 12,
    measureElement:
      typeof window !== 'undefined' && 'ResizeObserver' in window
        ? (el) => el.getBoundingClientRect().height
        : undefined,
  })

  // Force the virtualizer to re-measure when the expanded id changes (since
  // estimateSize depends on expandedId).
  useEffect(() => {
    rowVirtualizer.measure()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expandedId])

  const virtualItems = rowVirtualizer.getVirtualItems()
  const totalHeight = rowVirtualizer.getTotalSize()

  // Pre-compute the full grid width.
  const skillColsWidth = orderedSkills.length * COL_W.skill
  const gridTotalW = PINNED_TOTAL_W + skillColsWidth

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col">
      <ModelFilters
        search={search}
        onSearchChange={setSearch}
        selectedFamilies={selectedFamilies}
        onToggleFamily={toggleFamily}
        selectedTiers={selectedTiers}
        onToggleTier={toggleTier}
        visibleCount={sortedModels.length}
        totalCount={processedModels.length}
      />

      <div
        ref={scrollRef}
        className="relative flex-1 overflow-auto border-t border-border bg-background"
      >
        {/* Sticky header (positioned at top via sticky positioning on inner elements) */}
        <div
          style={{
            width: gridTotalW,
            minWidth: '100%',
          }}
        >
          <TableHeader
            groups={groups}
            sort={sort}
            onToggleSort={toggleSort}
            pinnedTotalW={PINNED_TOTAL_W}
          />

          {/* Body */}
          <div
            style={{
              height: totalHeight,
              position: 'relative',
              width: '100%',
            }}
          >
            {virtualItems.map((vi) => {
              const m = sortedModels[vi.index]
              const isExpanded = m.id === expandedId
              return (
                <div
                  key={m.id}
                  data-index={vi.index}
                  ref={rowVirtualizer.measureElement}
                  style={{
                    position: 'absolute',
                    top: vi.start,
                    left: 0,
                    width: gridTotalW,
                  }}
                >
                  <TableRow
                    model={m}
                    displayRank={vi.index + 1}
                    skills={orderedSkills}
                    darkMode={darkMode}
                    expanded={isExpanded}
                    onToggleExpand={() =>
                      setExpandedId((cur) => (cur === m.id ? null : m.id))
                    }
                  />
                  {isExpanded ? (
                    <RowExpansion
                      model={m}
                      skills={orderedSkills}
                      darkMode={darkMode}
                    />
                  ) : null}
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------- Header ----------

interface TableHeaderProps {
  groups: SortableSkillGroup[]
  sort: SortState
  onToggleSort: (k: SortKey) => void
  pinnedTotalW: number
}

function TableHeader({
  groups,
  sort,
  onToggleSort,
  pinnedTotalW,
}: TableHeaderProps) {
  return (
    <div
      className="sticky top-0 z-30 flex border-b border-border bg-background"
      style={{ height: COL_HEADER_H }}
    >
      {/* Pinned header cluster (also sticky to the left) */}
      <div
        className="sticky left-0 z-30 flex bg-background"
        style={{
          width: pinnedTotalW,
          boxShadow: PINNED_RIGHT_SHADOW,
        }}
      >
        <PinnedHeaderCell
          label="#"
          k={{ kind: 'index' }}
          sort={sort}
          onToggle={onToggleSort}
          width={COL_W.rank}
          align="center"
        />
        <PinnedHeaderCell
          label="Model"
          k={{ kind: 'name' }}
          sort={sort}
          onToggle={onToggleSort}
          width={COL_W.name}
        />
        <PinnedHeaderCell
          label="Family"
          k={{ kind: 'family' }}
          sort={sort}
          onToggle={onToggleSort}
          width={COL_W.family}
        />
        <PinnedHeaderCell
          label="Tier"
          k={{ kind: 'tier' }}
          sort={sort}
          onToggle={onToggleSort}
          width={COL_W.tier}
        />
        <PinnedHeaderCell
          label="Params"
          k={{ kind: 'params' }}
          sort={sort}
          onToggle={onToggleSort}
          width={COL_W.params}
          align="right"
        />
        <PinnedHeaderCell
          label="Mean θ"
          k={{ kind: 'mean' }}
          sort={sort}
          onToggle={onToggleSort}
          width={COL_W.mean}
          align="right"
        />
      </div>

      {/* Benchmark groups */}
      <div className="flex flex-col">
        {/* Top: group bands */}
        <div className="flex" style={{ height: GROUP_HEADER_H }}>
          {groups.map((g) => (
            <div
              key={g.benchmark}
              style={{ width: g.skills.length * COL_W.skill }}
              title={`${g.benchmark} — ${g.skills.length} skills`}
            >
              <BenchmarkBadge
                benchmark={g.benchmark}
                count={g.skills.length}
              />
            </div>
          ))}
        </div>
        {/* Bottom: individual skill columns */}
        <div className="flex" style={{ height: SKILL_HEADER_H }}>
          {groups.map((g) =>
            g.skills.map((s) => (
              <SkillHeaderCell
                key={s.id}
                skill={s}
                sort={sort}
                onToggle={onToggleSort}
                width={COL_W.skill}
              />
            ))
          )}
        </div>
      </div>
    </div>
  )
}

interface PinnedHeaderCellProps {
  label: string
  k: SortKey
  sort: SortState
  onToggle: (k: SortKey) => void
  width: number
  align?: 'left' | 'right' | 'center'
}

function PinnedHeaderCell({
  label,
  k,
  sort,
  onToggle,
  width,
  align = 'left',
}: PinnedHeaderCellProps) {
  const active = sortKeysEqual(sort.key, k)
  return (
    <button
      type="button"
      onClick={() => onToggle(k)}
      className={cn(
        'flex h-full items-center gap-1 border-r border-b border-border px-2 text-xs font-semibold uppercase tracking-wide text-foreground transition-colors hover:bg-accent',
        align === 'right' && 'justify-end',
        align === 'center' && 'justify-center'
      )}
      style={{ width, height: COL_HEADER_H }}
    >
      <span>{label}</span>
      <SortIndicator active={active} dir={active ? sort.dir : undefined} />
    </button>
  )
}

interface SkillHeaderCellProps {
  skill: Skill
  sort: SortState
  onToggle: (k: SortKey) => void
  width: number
}

function SkillHeaderCell({
  skill,
  sort,
  onToggle,
  width,
}: SkillHeaderCellProps) {
  const k: SortKey = { kind: 'skill', skillId: skill.id }
  const active = sortKeysEqual(sort.key, k)
  return (
    <div
      className="relative flex flex-col items-center justify-end border-r border-b border-border px-1 pb-1 pt-1 hover:bg-accent"
      style={{ width, height: SKILL_HEADER_H }}
      title={skill.label}
    >
      <button
        type="button"
        onClick={() => onToggle(k)}
        className="flex w-full flex-1 flex-col items-center justify-end gap-0.5 truncate text-[10px] font-medium leading-tight text-foreground"
      >
        <span
          className="line-clamp-3 w-full px-0.5 text-center"
          style={{ wordBreak: 'break-word' }}
        >
          {skill.label}
        </span>
        <SortIndicator active={active} dir={active ? sort.dir : undefined} />
      </button>
      <Link
        to={`/skill/${skill.id}`}
        onClick={(e) => e.stopPropagation()}
        className="absolute right-0.5 top-0.5 text-muted-foreground hover:text-foreground"
        aria-label={`Open skill ${skill.label}`}
        title={`Open skill page: ${skill.label}`}
      >
        <ExternalLink className="h-3 w-3" />
      </Link>
    </div>
  )
}

function SortIndicator({
  active,
  dir,
}: {
  active: boolean
  dir?: SortDir
}) {
  if (!active) {
    return (
      <ChevronsUpDown className="h-3 w-3 text-muted-foreground opacity-60" />
    )
  }
  return dir === 'asc' ? (
    <ArrowUp className="h-3 w-3 text-foreground" />
  ) : (
    <ArrowDown className="h-3 w-3 text-foreground" />
  )
}

// ---------- Row ----------

interface TableRowProps {
  model: ProcessedModel
  displayRank: number
  skills: Skill[]
  darkMode: boolean
  expanded: boolean
  onToggleExpand: () => void
}

function TableRow({
  model,
  displayRank,
  skills,
  darkMode,
  expanded,
  onToggleExpand,
}: TableRowProps) {
  const familyColor = getFamilyColor(model.family)
  return (
    <div
      onClick={onToggleExpand}
      className={cn(
        'group relative flex w-full cursor-pointer border-b border-border transition-colors hover:bg-accent/50',
        expanded && 'bg-accent/40'
      )}
      style={{ height: ROW_HEIGHT }}
    >
      {/* Pinned cluster */}
      <div
        className={cn(
          'sticky left-0 z-10 flex bg-background group-hover:bg-accent/50',
          expanded && 'bg-accent/40'
        )}
        style={{
          width: PINNED_TOTAL_W,
          boxShadow: PINNED_RIGHT_SHADOW,
        }}
      >
        <div
          className="flex h-full items-center justify-center border-r border-border text-xs text-muted-foreground"
          style={{ width: COL_W.rank }}
        >
          {displayRank}
        </div>
        <div
          className="flex h-full items-center border-r border-border px-2 text-sm"
          style={{ width: COL_W.name }}
          title={model.name}
        >
          <span className="truncate">{model.name}</span>
        </div>
        <div
          className="flex h-full items-center border-r border-border px-2"
          style={{ width: COL_W.family }}
        >
          <span
            className="inline-flex h-5 items-center rounded-full px-2 text-[10px] font-semibold text-white"
            style={{ backgroundColor: familyColor }}
          >
            {model.family}
          </span>
        </div>
        <div
          className="flex h-full items-center border-r border-border px-2 text-xs text-muted-foreground"
          style={{ width: COL_W.tier }}
        >
          {model.tier}
        </div>
        <div
          className="flex h-full items-center justify-end border-r border-border px-2 font-mono text-xs"
          style={{ width: COL_W.params }}
        >
          {formatParams(model.params)}
        </div>
        <div
          className="flex h-full items-center justify-end border-r border-border px-2 font-mono text-xs font-semibold"
          style={{
            width: COL_W.mean,
            // Subtle gradient — about 70% intensity of the per-skill cells.
            backgroundColor: rgbToRgba(
              getMasteryColor(model.meanTheta, darkMode),
              0.7
            ),
            color: getMasteryTextColor(model.meanTheta, darkMode),
          }}
        >
          {model.meanTheta.toFixed(3)}
        </div>
      </div>

      {/* Skill cells */}
      <div className="flex h-full">
        {skills.map((s) => {
          const v = model.theta[s.id] ?? 0
          return (
            <div
              key={s.id}
              className="h-full border-r border-border"
              style={{ width: COL_W.skill }}
            >
              <SkillCell theta={v} darkMode={darkMode} skillLabel={s.label} />
            </div>
          )
        })}
      </div>
    </div>
  )
}
