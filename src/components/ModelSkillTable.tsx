import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  ArrowDown,
  ArrowUp,
  ChevronRight,
  ChevronsUpDown,
  ExternalLink,
  GitCompareArrows,
  SearchX,
} from 'lucide-react'
import { useVirtualizer } from '@tanstack/react-virtual'
import type { Model, Skill } from '@/lib/types'
import { displaySkillLabel } from '@/lib/labels'
import { cn } from '@/lib/utils'
import {
  getBenchmarkColor,
  getFamilyColor,
  getMasteryColor,
  getMasteryTextColor,
  rgbToRgba,
} from '@/lib/colors'
import { RowExpansion } from '@/components/RowExpansion'
import { CompareModal, CompareTray } from '@/components/CompareOverlay'
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
  | { kind: 'acc' }
  | { kind: 'mean' }
  | { kind: 'skill'; skillId: number }

type SortDir = 'asc' | 'desc'

interface SortState {
  key: SortKey
  dir: SortDir
}

type Density = 'compact' | 'regular' | 'relaxed'
type ValueMode = 'abs' | 'rel'

// Column widths (px). Adjust here if visual tuning is needed.
const COL_W = {
  expander: 28,
  rank: 44,
  name: 264,
  acc: 76,
  mean: 84,
  skill: 72,
} as const

const PINNED_WIDTHS = [
  COL_W.expander,
  COL_W.rank,
  COL_W.name,
  COL_W.acc,
  COL_W.mean,
]
const PINNED_TOTAL_W = PINNED_WIDTHS.reduce((a, b) => a + b, 0) // 496

const ROW_HEIGHTS: Record<Density, number> = {
  compact: 32,
  regular: 40,
  relaxed: 48,
}

// Fixed height of the expansion panel; its skill list scrolls internally.
// Must match the height RowExpansion renders at.
const EXPANDED_PANEL_HEIGHT = 340
const GROUP_HEADER_H = 28
const SKILL_HEADER_H = 84
const COL_HEADER_H = GROUP_HEADER_H + SKILL_HEADER_H

interface ProcessedModel extends Model {
  meanTheta: number
}

interface SortableSkillGroup {
  benchmark: string
  skills: Skill[]
  startIndex: number // index into ordered skill list
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
  const { orderedSkills, groups, groupStartIds } = useMemo(() => {
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
    // Skill ids that begin a benchmark group (for the 1px group dividers).
    const startIds = new Set<number>()
    for (const g of groupArr) {
      const first = ordered[g.startIndex]
      if (first) startIds.add(first.id)
    }
    return { orderedSkills: ordered, groups: groupArr, groupStartIds: startIds }
  }, [skills])

  // Per-skill population histogram (16 bins over [0,1], max-normalized),
  // computed once: it powers the micro-distribution under each column header.
  const skillHists = useMemo(() => {
    const BINS = 16
    const hists = new Map<number, number[]>()
    if (models.length === 0) return hists
    for (const s of skills) {
      const bins = new Array<number>(BINS).fill(0)
      for (const m of models) {
        const v = m.theta[s.id] ?? 0
        bins[Math.min(BINS - 1, Math.floor(v * BINS))]++
      }
      const max = Math.max(...bins, 1)
      hists.set(
        s.id,
        bins.map((b) => b / max)
      )
    }
    return hists
  }, [models, skills])

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
  // Stable toggle so memoized rows don't re-render from a fresh closure.
  const toggleExpand = useCallback((id: number) => {
    setExpandedId((cur) => (cur === id ? null : id))
  }, [])

  // Compare selection (max 3 models).
  const [compareIds, setCompareIds] = useState<number[]>([])
  const [compareOpen, setCompareOpen] = useState(false)
  const toggleCompare = useCallback((id: number) => {
    setCompareIds((cur) =>
      cur.includes(id)
        ? cur.filter((x) => x !== id)
        : cur.length >= 3
          ? cur
          : [...cur, id]
    )
  }, [])
  const compareModels = useMemo(
    () =>
      compareIds
        .map((id) => processedModels.find((m) => m.id === id))
        .filter((m): m is ProcessedModel => m != null),
    [compareIds, processedModels]
  )

  // Density (persisted)
  const [density, setDensity] = useState<Density>(() => {
    if (typeof window === 'undefined') return 'compact'
    const v = window.localStorage.getItem('skilleval-density')
    return v === 'regular' || v === 'relaxed' ? v : 'compact'
  })
  useEffect(() => {
    window.localStorage.setItem('skilleval-density', density)
  }, [density])
  const rowHeight = ROW_HEIGHTS[density]
  const cellFontSize = density === 'compact' ? 10 : 12

  // Absolute theta vs relative-to-population-mean display (persisted).
  const [valueMode, setValueMode] = useState<ValueMode>(() => {
    if (typeof window === 'undefined') return 'abs'
    return window.localStorage.getItem('skilleval-valuemode') === 'rel'
      ? 'rel'
      : 'abs'
  })
  useEffect(() => {
    window.localStorage.setItem('skilleval-valuemode', valueMode)
  }, [valueMode])

  // Population mean per skill, for relative mode.
  const skillMeans = useMemo(() => {
    const means = new Map<number, number>()
    if (models.length === 0) return means
    for (const s of skills) {
      let sum = 0
      for (const m of models) sum += m.theta[s.id] ?? 0
      means.set(s.id, sum / models.length)
    }
    return means
  }, [models, skills])

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
    const cmpStr = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)
    arr.sort((a, b) => {
      switch (key.kind) {
        case 'index':
          return (a.id - b.id) * dirMul
        case 'name':
          return cmpStr(a.name.toLowerCase(), b.name.toLowerCase()) * dirMul
        case 'acc': {
          const av = a.accuracy ?? -Infinity
          const bv = b.accuracy ?? -Infinity
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
  const clearAll = useCallback(() => {
    setSearch('')
    setSelectedFamilies(new Set())
    setSelectedTiers(new Set())
  }, [])

  const toggleSort = useCallback((key: SortKey) => {
    setSort((prev) => {
      if (sortKeysEqual(prev.key, key)) {
        return { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' }
      }
      // Numeric / mastery columns default to descending; name defaults asc.
      const numericKinds = ['mean', 'acc', 'skill', 'index']
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
      return id === expandedId ? rowHeight + EXPANDED_PANEL_HEIGHT : rowHeight
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

  // Re-measure when density changes (estimateSize depends on rowHeight).
  useEffect(() => {
    rowVirtualizer.measure()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [density])

  // Pre-compute the full grid width.
  const skillColsWidth = orderedSkills.length * COL_W.skill
  const gridTotalW = PINNED_TOTAL_W + skillColsWidth

  // Horizontal-scroll affordance state. Scroll events fire continuously
  // (vertical scrolling included), so store only the two booleans the UI
  // needs and update state only when one actually flips; otherwise every
  // scroll tick re-renders the whole virtualized grid and scrolling lags.
  const [edges, setEdges] = useState({ left: false, right: false })
  const syncEdges = useCallback((el: HTMLDivElement) => {
    const left = el.scrollLeft > 1
    const right = el.scrollLeft < el.scrollWidth - el.clientWidth - 1
    setEdges((prev) =>
      prev.left === left && prev.right === right ? prev : { left, right }
    )
  }, [])
  const onScroll = useCallback(
    (e: React.UIEvent<HTMLDivElement>) => syncEdges(e.currentTarget),
    [syncEdges]
  )
  // Visible scroller width; used to pin the expansion panel on screen.
  const [viewportW, setViewportW] = useState(0)
  // Measure scroll extent on mount and when the grid width changes so the
  // right-edge fade shows on first paint (the grid is wider than the viewport).
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    syncEdges(el)
    setViewportW(el.clientWidth)
    const onResize = () => {
      if (scrollRef.current) {
        syncEdges(scrollRef.current)
        setViewportW(scrollRef.current.clientWidth)
      }
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [gridTotalW, sortedModels.length, syncEdges])
  const showLeftShadow = edges.left
  const showRightShadow = edges.right
  const pinnedShadow = showLeftShadow
    ? '2px 0 0 0 hsl(var(--border)), 8px 0 8px -6px rgba(0,0,0,0.18)'
    : '1px 0 0 0 hsl(var(--border))'

  const jumpToGroup = useCallback((startIndex: number) => {
    scrollRef.current?.scrollTo({
      left: PINNED_TOTAL_W + startIndex * COL_W.skill,
      behavior: 'smooth',
    })
  }, [])

  const virtualItems = rowVirtualizer.getVirtualItems()
  const totalHeight = rowVirtualizer.getTotalSize()

  const isEmpty = sortedModels.length === 0

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col">
      <div className="border-b border-border">
        <ModelFilters
          search={search}
          onSearchChange={setSearch}
          selectedFamilies={selectedFamilies}
          onToggleFamily={toggleFamily}
          selectedTiers={selectedTiers}
          onToggleTier={toggleTier}
          visibleCount={sortedModels.length}
          totalCount={processedModels.length}
          density={density}
          onDensityChange={setDensity}
          valueMode={valueMode}
          onValueModeChange={setValueMode}
          onClearAll={clearAll}
        />
      </div>

      {/* Benchmark jump bar */}
      <div className="flex flex-wrap items-center gap-2 border-b border-border bg-background px-6 py-1.5">
        <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Jump to
        </span>
        {groups.map((g) => (
          <button
            key={g.benchmark}
            type="button"
            onClick={() => jumpToGroup(g.startIndex)}
            className="inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-xs text-muted-foreground transition-colors hover:bg-surface-elevated hover:text-foreground"
          >
            <span
              className="h-2 w-2 rounded-full"
              style={{ backgroundColor: getBenchmarkColor(g.benchmark) }}
            />
            {g.benchmark}
          </button>
        ))}
      </div>

      <div className="relative flex-1 border-t border-border">
      {/* contain:strict isolates the scroller's layout/paint from the page;
          without it Chrome repaints the full grid every scroll frame and
          scrolling visibly lags. Size comes from the wrapper, so size
          containment is safe here. */}
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="h-full overflow-auto bg-background [contain:strict]"
      >
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
            pinnedShadow={pinnedShadow}
            groupStartIds={groupStartIds}
            skillHists={skillHists}
          />

          {/* Body */}
          {isEmpty ? (
            <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
              <SearchX className="h-8 w-8 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                No models match these filters.
              </p>
              <button
                type="button"
                onClick={clearAll}
                className="rounded-md border border-border px-3 py-1.5 text-sm text-foreground hover:bg-surface-elevated"
              >
                Clear filters
              </button>
            </div>
          ) : (
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
                      rowHeight={rowHeight}
                      cellFontSize={cellFontSize}
                      pinnedShadow={pinnedShadow}
                      sortedSkillId={
                        sort.key.kind === 'skill' ? sort.key.skillId : null
                      }
                      groupStartIds={groupStartIds}
                      onToggleExpand={toggleExpand}
                      valueMode={valueMode}
                      skillMeans={skillMeans}
                      inCompare={compareIds.includes(m.id)}
                      compareFull={
                        compareIds.length >= 3 && !compareIds.includes(m.id)
                      }
                      onToggleCompare={toggleCompare}
                    />
                    {isExpanded ? (
                      // Pin the panel to the visible viewport: the row wrapper
                      // is grid-wide (thousands of px), so without this the
                      // panel content sits far off-screen when scrolled right.
                      <div
                        className="sticky left-0"
                        style={{ width: viewportW || '100%' }}
                      >
                        <RowExpansion
                          model={m}
                          skills={orderedSkills}
                          darkMode={darkMode}
                          height={EXPANDED_PANEL_HEIGHT}
                          inCompare={compareIds.includes(m.id)}
                          compareFull={
                            compareIds.length >= 3 &&
                            !compareIds.includes(m.id)
                          }
                          onToggleCompare={toggleCompare}
                        />
                      </div>
                    ) : null}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

        {/* Right-edge fade overlay: anchored to the viewport edge of the
            scroll area, indicating more grid columns off-screen. */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-y-0 right-0 transition-opacity duration-150"
          style={{
            width: 24,
            opacity: showRightShadow ? 1 : 0,
            background:
              'linear-gradient(to left, hsl(var(--background)), transparent)',
          }}
        />
      </div>

      {/* Compare tray + modal */}
      <CompareTray
        models={compareModels}
        onRemove={toggleCompare}
        onClear={() => setCompareIds([])}
        onOpen={() => setCompareOpen(true)}
      />
      {compareOpen && compareModels.length >= 2 ? (
        <CompareModal
          models={compareModels}
          orderedSkills={orderedSkills}
          darkMode={darkMode}
          onClose={() => setCompareOpen(false)}
        />
      ) : null}
    </div>
  )
}

// ---------- Header ----------

interface TableHeaderProps {
  groups: SortableSkillGroup[]
  sort: SortState
  onToggleSort: (k: SortKey) => void
  pinnedTotalW: number
  pinnedShadow: string
  groupStartIds: Set<number>
  skillHists: Map<number, number[]>
}

// Memoized: the header holds ~100 cells with histograms, and the parent
// re-renders on every virtualizer scroll tick; props only change on sort or
// edge-shadow flips.
const TableHeader = memo(function TableHeader({
  groups,
  sort,
  onToggleSort,
  pinnedTotalW,
  pinnedShadow,
  groupStartIds,
  skillHists,
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
          boxShadow: pinnedShadow,
        }}
      >
        {/* Expander spacer */}
        <div
          className="border-b border-border"
          style={{ width: COL_W.expander, height: COL_HEADER_H }}
        />
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
          label="Acc"
          k={{ kind: 'acc' }}
          sort={sort}
          onToggle={onToggleSort}
          width={COL_W.acc}
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
              title={`${g.benchmark} (${g.skills.length} skills)`}
            >
              <BenchmarkBadge benchmark={g.benchmark} count={g.skills.length} />
            </div>
          ))}
        </div>
        {/* Bottom: individual skill columns */}
        <div className="flex" style={{ height: SKILL_HEADER_H }}>
          {groups.map((g) =>
            g.skills.map((s) => {
              const isSorted =
                sort.key.kind === 'skill' && sort.key.skillId === s.id
              return (
                <SkillHeaderCell
                  key={s.id}
                  skill={s}
                  sort={sort}
                  onToggle={onToggleSort}
                  width={COL_W.skill}
                  isSorted={isSorted}
                  isGroupStart={groupStartIds.has(s.id)}
                  hist={skillHists.get(s.id)}
                />
              )
            })
          )}
        </div>
      </div>
    </div>
  )
})

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
      {align === 'right' ? (
        <>
          <SortIndicator active={active} dir={active ? sort.dir : undefined} />
          <span>{label}</span>
        </>
      ) : (
        <>
          <span>{label}</span>
          <SortIndicator active={active} dir={active ? sort.dir : undefined} />
        </>
      )}
    </button>
  )
}

interface SkillHeaderCellProps {
  skill: Skill
  sort: SortState
  onToggle: (k: SortKey) => void
  width: number
  isSorted: boolean
  isGroupStart: boolean
  hist?: number[]
}

function SkillHeaderCell({
  skill,
  sort,
  onToggle,
  width,
  isSorted,
  isGroupStart,
  hist,
}: SkillHeaderCellProps) {
  const k: SortKey = { kind: 'skill', skillId: skill.id }
  const active = sortKeysEqual(sort.key, k)
  return (
    <div
      className={cn(
        'relative flex flex-col items-center justify-end border-b border-border px-1 pb-1 pt-1 hover:bg-accent',
        isSorted && 'bg-brand/10'
      )}
      style={{
        width,
        height: SKILL_HEADER_H,
        borderLeft: isGroupStart ? '1px solid hsl(var(--border))' : undefined,
      }}
      title={skill.label}
    >
      <button
        type="button"
        onClick={() => onToggle(k)}
        className="flex w-full flex-1 flex-col items-center justify-end gap-1 overflow-hidden text-[10px] font-medium leading-tight text-foreground"
      >
        <span
          className="line-clamp-3 w-full px-0.5 text-center"
          style={{ wordBreak: 'break-word', whiteSpace: 'normal' }}
        >
          {displaySkillLabel(skill.label)}
        </span>
        {hist ? (
          <svg
            width={64}
            height={13}
            viewBox="0 0 64 13"
            aria-hidden
            className="shrink-0"
          >
            {hist.map((h, i) => (
              <rect
                key={i}
                x={i * 4}
                y={13 - Math.max(1, h * 12)}
                width={3}
                height={Math.max(1, h * 12)}
                rx={0.5}
                fill={
                  isSorted
                    ? 'hsl(var(--brand) / 0.85)'
                    : 'hsl(var(--muted-foreground) / 0.55)'
                }
              />
            ))}
          </svg>
        ) : null}
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

function SortIndicator({ active, dir }: { active: boolean; dir?: SortDir }) {
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
  rowHeight: number
  cellFontSize: number
  pinnedShadow: string
  sortedSkillId: number | null
  groupStartIds: Set<number>
  onToggleExpand: (id: number) => void
  valueMode: ValueMode
  skillMeans: Map<number, number>
  inCompare: boolean
  compareFull: boolean
  onToggleCompare: (id: number) => void
}

// Memoized: on expand/collapse and most filter/sort updates only the rows
// whose props actually changed re-render, instead of every visible row's
// 100 cells. All props are primitives or stable references on purpose.
const TableRow = memo(function TableRow({
  model,
  displayRank,
  skills,
  darkMode,
  expanded,
  rowHeight,
  cellFontSize,
  pinnedShadow,
  sortedSkillId,
  groupStartIds,
  onToggleExpand,
  valueMode,
  skillMeans,
  inCompare,
  compareFull,
  onToggleCompare,
}: TableRowProps) {
  const familyColor = getFamilyColor(model.family)
  const navigate = useNavigate()
  return (
    <div
      onClick={() => onToggleExpand(model.id)}
      className={cn(
        'group relative flex w-full cursor-pointer border-b border-border transition-colors hover:bg-surface-elevated/60',
        expanded && 'bg-surface-elevated'
      )}
      style={{
        height: rowHeight,
        boxShadow: expanded ? 'inset 2px 0 0 0 hsl(var(--brand))' : undefined,
      }}
    >
      {/* Pinned cluster */}
      <div
        className={cn(
          'sticky left-0 z-10 flex bg-background group-hover:bg-surface-elevated',
          expanded && 'bg-surface-elevated'
        )}
        style={{
          width: PINNED_TOTAL_W,
          boxShadow: pinnedShadow,
        }}
      >
        {/* Expander */}
        <div
          className="flex h-full items-center justify-center text-muted-foreground"
          style={{ width: COL_W.expander }}
        >
          <ChevronRight
            className={cn(
              'h-3.5 w-3.5 transition-transform duration-150',
              expanded && 'rotate-90'
            )}
          />
        </div>
        {/* Rank */}
        <div
          className="tabular flex h-full items-center justify-center border-r border-border text-xs text-muted-foreground"
          style={{ width: COL_W.rank }}
        >
          {displayRank}
        </div>
        {/* Model name + family dot */}
        <div
          className="relative flex h-full items-center gap-2 border-r border-border px-3 text-sm"
          style={{ width: COL_W.name }}
          title={model.name}
        >
          <span
            className="h-2 w-2 shrink-0 rounded-full"
            style={{ backgroundColor: familyColor }}
            aria-hidden
          />
          <Link
            to={`/model/${model.id}`}
            onClick={(e) => e.stopPropagation()}
            className="truncate hover:text-brand hover:underline"
            title={`Open the model page for ${model.name}`}
          >
            {model.name}
          </Link>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              if (!compareFull || inCompare) onToggleCompare(model.id)
            }}
            disabled={compareFull && !inCompare}
            aria-label={
              inCompare
                ? `Remove ${model.name} from comparison`
                : `Add ${model.name} to comparison`
            }
            title={
              compareFull && !inCompare
                ? 'Comparison is full (3 models)'
                : inCompare
                  ? 'Remove from comparison'
                  : 'Add to comparison'
            }
            className={cn(
              'ml-auto shrink-0 rounded p-1 transition-all',
              inCompare
                ? 'text-brand opacity-100'
                : 'text-muted-foreground opacity-0 hover:text-brand group-hover:opacity-100 disabled:cursor-not-allowed'
            )}
          >
            <GitCompareArrows className="h-3.5 w-3.5" />
          </button>
        </div>
        {/* Accuracy */}
        <div
          className="tabular flex h-full items-center justify-end border-r border-border px-2 font-mono text-xs"
          style={{ width: COL_W.acc }}
          title="Fraction of all 9,523 items answered correctly"
        >
          {model.accuracy != null
            ? `${(model.accuracy * 100).toFixed(1)}%`
            : '—'}
        </div>
        {/* Mean theta */}
        <div
          className="tabular flex h-full items-center justify-end border-r border-border px-2 font-mono text-xs font-semibold"
          style={{
            width: COL_W.mean,
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
          const v = model.theta[s.id] ?? null
          const isSorted = sortedSkillId === s.id
          return (
            <div
              key={s.id}
              className="h-full"
              style={{
                width: COL_W.skill,
                borderLeft: groupStartIds.has(s.id)
                  ? '1px solid hsl(var(--border))'
                  : undefined,
                boxShadow: isSorted
                  ? 'inset 0 0 0 1px hsl(var(--brand) / 0.25)'
                  : undefined,
              }}
              onClick={(e) => {
                e.stopPropagation()
                navigate(`/skill/${s.id}`)
              }}
              role="link"
              aria-label={`Open skill ${s.label}`}
            >
              <SkillCell
                theta={v}
                darkMode={darkMode}
                skillLabel={s.label}
                fontSize={cellFontSize}
                delta={
                  valueMode === 'rel' && v != null
                    ? v - (skillMeans.get(s.id) ?? 0)
                    : null
                }
              />
            </div>
          )
        })}
      </div>
    </div>
  )
})
