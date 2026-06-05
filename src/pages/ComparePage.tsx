import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Plus, Search, X } from 'lucide-react'
import { useSkillEvalData } from '@/hooks/useSkillEvalData'
import { usePageTitle } from '@/hooks/usePageTitle'
import { COMPARE_COLORS, CompareView } from '@/components/CompareOverlay'
import { getFamilyColor } from '@/lib/colors'
import type { Model, Skill } from '@/lib/types'

function useDarkMode(): boolean {
  const [dark, setDark] = useState<boolean>(
    typeof document !== 'undefined' &&
      document.documentElement.classList.contains('dark')
  )
  useEffect(() => {
    const root = document.documentElement
    const obs = new MutationObserver(() => {
      setDark(root.classList.contains('dark'))
    })
    obs.observe(root, { attributes: true, attributeFilter: ['class'] })
    return () => obs.disconnect()
  }, [])
  return dark
}

const MAX_MODELS = 3

/** Searchable picker for one comparison slot. */
function ModelPicker({
  models,
  exclude,
  onPick,
}: {
  models: Model[]
  exclude: Set<number>
  onPick: (id: number) => void
}) {
  const [q, setQ] = useState('')
  const [open, setOpen] = useState(false)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const matches = useMemo(() => {
    const query = q.trim().toLowerCase()
    if (!query) return []
    return models
      .filter(
        (m) => !exclude.has(m.id) && m.name.toLowerCase().includes(query)
      )
      .slice(0, 8)
  }, [models, exclude, q])

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [])

  return (
    <div ref={boxRef} className="relative">
      <div className="flex h-full min-h-[120px] flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border bg-surface/60 p-4">
        <Plus className="h-4 w-4 text-muted-foreground" />
        <div className="relative w-full max-w-[240px]">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <input
            type="search"
            value={q}
            onChange={(e) => {
              setQ(e.target.value)
              setOpen(true)
            }}
            onFocus={() => setOpen(true)}
            placeholder="Search a model..."
            className="w-full rounded-lg border border-border bg-card py-1.5 pl-8 pr-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          />
          {open && matches.length > 0 ? (
            <ul className="absolute left-0 right-0 top-full z-20 mt-1 max-h-64 overflow-y-auto rounded-lg border border-border bg-popover py-1 shadow-lg">
              {matches.map((m) => (
                <li key={m.id}>
                  <button
                    type="button"
                    onClick={() => {
                      onPick(m.id)
                      setQ('')
                      setOpen(false)
                    }}
                    className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm hover:bg-accent"
                  >
                    <span
                      className="h-2 w-2 shrink-0 rounded-full"
                      style={{ backgroundColor: getFamilyColor(m.family) }}
                    />
                    <span className="truncate">{m.name}</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      </div>
    </div>
  )
}

export function ComparePage() {
  usePageTitle('Compare · SkillEval')
  const { models, skills, loading, error } = useSkillEvalData()
  const darkMode = useDarkMode()
  const [params, setParams] = useSearchParams()

  // Selection lives in the URL so comparisons are shareable links.
  const ids = useMemo(() => {
    const raw = params.get('m') ?? ''
    return raw
      .split(',')
      .map((s) => Number.parseInt(s, 10))
      .filter((n) => Number.isFinite(n))
      .slice(0, MAX_MODELS)
  }, [params])

  const selected = useMemo(
    () =>
      ids
        .map((id) => models.find((m) => m.id === id))
        .filter((m): m is Model => m != null),
    [ids, models]
  )

  const setIds = (next: number[]) => {
    if (next.length === 0) setParams({}, { replace: false })
    else setParams({ m: next.join(',') }, { replace: false })
  }

  // Same benchmark-grouped skill order as the leaderboard grid.
  const orderedSkills = useMemo(() => {
    const order = ['MATH', 'BBH', 'GPQA', 'IFEval', 'MuSR']
    const byBench = new Map<string, Skill[]>()
    for (const s of skills) {
      const b = s.primary_benchmark
      if (!byBench.has(b)) byBench.set(b, [])
      byBench.get(b)!.push(s)
    }
    const benches = [
      ...order.filter((b) => byBench.has(b)),
      ...[...byBench.keys()].filter((b) => !order.includes(b)),
    ]
    return benches.flatMap((b) => byBench.get(b)!)
  }, [skills])

  // Default suggestion: the strongest model overall vs the best small one.
  const suggestion = useMemo(() => {
    if (models.length === 0) return null
    let best: Model | null = null
    let bestSmall: Model | null = null
    let bestMean = -1
    let bestSmallMean = -1
    for (const m of models) {
      let sum = 0
      for (const v of m.theta) sum += v
      const mean = m.theta.length ? sum / m.theta.length : 0
      if (mean > bestMean) {
        bestMean = mean
        best = m
      }
      if (m.params != null && m.params <= 13 && mean > bestSmallMean) {
        bestSmallMean = mean
        bestSmall = m
      }
    }
    if (!best || !bestSmall || best.id === bestSmall.id) return null
    return { a: best, b: bestSmall }
  }, [models])

  const excludeSet = useMemo(() => new Set(ids), [ids])

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
        Head to head
      </div>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        Compare models
      </h1>
      <p className="mt-3 max-w-[68ch] text-[15px] leading-7 text-foreground/90">
        Pick two or three models to see their skill fingerprints side by side
        and the skills where they diverge most. The selection lives in the
        URL, so a comparison can be shared as a link.
      </p>

      {error ? (
        <div className="mt-8 rounded-md border border-destructive bg-destructive/10 p-4 text-sm text-destructive">
          Error loading data: {error}
        </div>
      ) : loading ? (
        <div className="mt-8 h-72 animate-pulse rounded-xl bg-muted" />
      ) : (
        <>
          {/* slots */}
          <div
            className="mt-7 grid gap-4"
            style={{
              gridTemplateColumns: `repeat(${Math.min(
                MAX_MODELS,
                selected.length + 1
              )}, 1fr)`,
            }}
          >
            {selected.map((m, i) => (
              <div
                key={m.id}
                className="flex min-h-[120px] flex-col items-center justify-center gap-1.5 rounded-xl border border-border bg-card p-4"
              >
                <span
                  className="h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: COMPARE_COLORS[i] }}
                />
                <span
                  className="max-w-full truncate text-sm font-medium"
                  title={m.name}
                >
                  {m.name}
                </span>
                <span className="tabular text-xs text-muted-foreground">
                  {m.family}
                  {m.params != null ? ` · ${m.params}B` : ''}
                </span>
                <button
                  type="button"
                  onClick={() => setIds(ids.filter((x) => x !== m.id))}
                  className="mt-1 inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
                >
                  <X className="h-3 w-3" />
                  remove
                </button>
              </div>
            ))}
            {selected.length < MAX_MODELS ? (
              <ModelPicker
                models={models}
                exclude={excludeSet}
                onPick={(id) => setIds([...ids, id])}
              />
            ) : null}
          </div>

          {/* comparison or empty state */}
          {selected.length >= 2 ? (
            <div className="mt-8">
              <CompareView
                models={selected}
                orderedSkills={orderedSkills}
                darkMode={darkMode}
              />
            </div>
          ) : suggestion ? (
            <div className="mt-8 rounded-xl border border-border bg-surface p-5 text-sm">
              <span className="text-muted-foreground">
                Not sure where to start?{' '}
              </span>
              <button
                type="button"
                onClick={() => setIds([suggestion.a.id, suggestion.b.id])}
                className="font-medium text-brand hover:underline"
              >
                Try {suggestion.a.name} vs {suggestion.b.name}
              </button>
              <span className="text-muted-foreground">
                {' '}
                (the strongest model overall against the best one at 13B or
                under).
              </span>
            </div>
          ) : null}
        </>
      )}
    </div>
  )
}
