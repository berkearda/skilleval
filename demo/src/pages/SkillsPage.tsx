import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Search } from 'lucide-react'
import { useSkillEvalData } from '@/hooks/useSkillEvalData'
import { usePageTitle } from '@/hooks/usePageTitle'
import { displaySkillLabel } from '@/lib/labels'
import { getBenchmarkColor } from '@/lib/colors'
import type { Skill } from '@/lib/types'

const BENCH_ORDER = ['MATH', 'BBH', 'GPQA', 'IFEval', 'MuSR']

const ROW = 'grid grid-cols-[minmax(0,1fr)_64px_2.75rem] items-center gap-3 px-2'

/** Column heads, shown once at the top of each column. */
function HeadRow({ className = '' }: { className?: string }) {
  return (
    <li aria-hidden className={`${ROW} pb-1 text-xs text-muted-foreground ${className}`}>
      <span>Skill</span>
      <span />
      <span className="text-right">Items</span>
    </li>
  )
}

function SkillRow({ skill, maxItems }: { skill: Skill; maxItems: number }) {
  return (
    <li>
      <Link
        to={`/skill/${skill.id}`}
        className={`${ROW} rounded-md py-1.5 text-sm transition-colors hover:bg-accent hover:text-brand`}
      >
        <span className="min-w-0">{displaySkillLabel(skill.label_english ?? skill.label)}</span>
        {/* bar length: items needing the skill, on one scale for all 100 skills */}
        <span className="h-1.5 rounded-full bg-muted">
          <span
            className="block h-full rounded-full"
            style={{
              width: `${Math.max(3, (100 * skill.n_items) / maxItems)}%`,
              backgroundColor: getBenchmarkColor(skill.primary_benchmark),
            }}
          />
        </span>
        <span className="tabular text-right font-mono text-xs text-muted-foreground">
          {skill.n_items.toLocaleString()}
        </span>
      </Link>
    </li>
  )
}

export function SkillsPage() {
  usePageTitle('Skills · SkillEval')
  const { skills, loading, error } = useSkillEvalData()
  const [query, setQuery] = useState('')

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase()
    const filtered = q
      ? skills.filter((s) =>
          (s.label_english ?? s.label).toLowerCase().includes(q)
        )
      : skills
    const byBench = new Map<string, Skill[]>()
    for (const s of filtered) {
      const b = s.primary_benchmark
      if (!byBench.has(b)) byBench.set(b, [])
      byBench.get(b)!.push(s)
    }
    const benches = [
      ...BENCH_ORDER.filter((b) => byBench.has(b)),
      ...[...byBench.keys()].filter((b) => !BENCH_ORDER.includes(b)),
    ]
    return benches.map((b) => ({
      benchmark: b,
      skills: byBench
        .get(b)!
        .slice()
        .sort((a, c) => c.n_items - a.n_items),
    }))
  }, [skills, query])

  const total = groups.reduce((n, g) => n + g.skills.length, 0)
  const maxItems = skills.reduce((m, s) => Math.max(m, s.n_items), 1)

  return (
    <div className="page py-10">
      <h1 className="text-3xl font-semibold tracking-tight">
        All 100 skills
      </h1>
      <p className="mt-3 max-w-[68ch] text-[15px] leading-7 text-foreground/90">
        Each of the 9,523 benchmark items needs one or more of these 100
        skills. They are grouped by the benchmark most of their items come
        from and sorted by item count. Every skill links to its page with the
        top models and example items.
      </p>

      {/* search */}
      <div className="relative mt-6 max-w-sm">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search skills..."
          className="w-full rounded-lg border border-border bg-card py-2 pl-9 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
        />
      </div>
      {query ? (
        <p className="mt-2 text-xs text-muted-foreground">
          {total} skill{total === 1 ? '' : 's'} match
        </p>
      ) : null}

      {error ? (
        <div className="mt-8 rounded-md border border-destructive bg-destructive/10 p-4 text-sm text-destructive">
          Error loading data: {error}
        </div>
      ) : loading ? (
        <div className="mt-8 grid gap-6 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-64 animate-pulse rounded-lg bg-muted" />
          ))}
        </div>
      ) : (
        <div className="mt-8 space-y-10">
          {groups.map((g) => (
            <section key={g.benchmark}>
              <div className="flex items-baseline gap-3 border-b border-border pb-2">
                <h2 className="flex items-center gap-2 text-lg font-semibold">
                  <span
                    className="h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ backgroundColor: getBenchmarkColor(g.benchmark) }}
                  />
                  {g.benchmark}
                </h2>
                <span className="text-sm text-muted-foreground">
                  {g.skills.length} skills
                </span>
              </div>
              <ul className="mt-3 grid gap-x-8 md:grid-cols-2">
                <HeadRow />
                <HeadRow className="hidden md:grid" />
                {g.skills.map((s) => (
                  <SkillRow key={s.id} skill={s} maxItems={maxItems} />
                ))}
              </ul>
            </section>
          ))}
          {total === 0 ? (
            <p className="text-sm text-muted-foreground">
              No skills match that search.
            </p>
          ) : null}
        </div>
      )}
    </div>
  )
}
