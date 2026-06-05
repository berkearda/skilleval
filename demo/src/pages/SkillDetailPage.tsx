import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { useSkillEvalData } from '@/hooks/useSkillEvalData'
import { BenchmarkBadge } from '@/components/BenchmarkBadge'
import { SkillStatsRow } from '@/components/SkillStatsRow'
import { TopLLMsChart } from '@/components/TopLLMsChart'
import { ExampleItemCard } from '@/components/ExampleItemCard'

// Clusters where n_items < 3 — flagged in the orchestration plan so we can
// render an explicit "small cluster" note instead of padding with empty cards.
const SMALL_CLUSTER_IDS = new Set([
  50, 53, 56, 63, 67, 69, 75, 79, 80, 82, 85, 94, 95,
])

const LANGUAGE_NOTES: Record<string, string> = {
  vi: 'Vietnamese-origin skill cluster — heading shown is the English translation.',
}

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

function normalize(s: string): string {
  return s.trim().toLowerCase().replace(/\s+/g, ' ')
}

function descriptionDuplicatesLabel(
  description: string,
  label: string,
  heading: string
): boolean {
  const d = normalize(description)
  return d === normalize(label) || d === normalize(heading)
}

function BackLink({ className = '' }: { className?: string }) {
  return (
    <Link
      to="/"
      className={`inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground ${className}`}
    >
      <ArrowLeft className="h-3.5 w-3.5" />
      Back to table
    </Link>
  )
}

function NotFound({ rawId }: { rawId: string | undefined }) {
  return (
    <div className="mx-auto max-w-3xl px-6 py-16">
      <h1 className="text-2xl font-semibold tracking-tight">Skill not found</h1>
      <p className="mt-3 text-muted-foreground">
        No skill matches the id <code className="rounded bg-muted px-1.5 py-0.5 text-sm">{rawId ?? '(missing)'}</code>.
        SkillEval has 100 named skills, indexed 0–99.
      </p>
      <div className="mt-6">
        <BackLink />
      </div>
    </div>
  )
}

function SkillDetailSkeleton() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <div className="mb-6">
        <div className="h-4 w-32 animate-pulse rounded bg-muted" />
      </div>
      {/* Heading */}
      <div className="border-b border-border pb-6">
        <div className="h-3 w-24 animate-pulse rounded bg-muted" />
        <div className="mt-3 h-9 w-2/3 animate-pulse rounded bg-muted" />
        <div className="mt-3 h-5 w-32 animate-pulse rounded bg-muted" />
      </div>
      {/* Stat cards */}
      <div className="mt-8 flex flex-col gap-3 sm:flex-row">
        {Array.from({ length: 3 }).map((_, i) => (
          <div
            key={i}
            className="h-20 flex-1 animate-pulse rounded-md bg-muted"
          />
        ))}
      </div>
      {/* Chart placeholder */}
      <div className="mt-10">
        <div className="h-5 w-56 animate-pulse rounded bg-muted" />
        <div className="mt-4 h-[420px] w-full animate-pulse rounded-md bg-muted" />
      </div>
      {/* Example items placeholder */}
      <div className="mt-10">
        <div className="h-5 w-40 animate-pulse rounded bg-muted" />
        <div className="mt-4 flex flex-col gap-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <div
              key={i}
              className="h-24 w-full animate-pulse rounded-md bg-muted"
            />
          ))}
        </div>
      </div>
    </div>
  )
}

export function SkillDetailPage() {
  const { id } = useParams<{ id: string }>()
  const { models, skills, loading, error } = useSkillEvalData()
  const darkMode = useDarkMode()

  const skillId = useMemo(() => {
    if (id == null) return null
    const n = Number.parseInt(id, 10)
    if (!Number.isFinite(n) || n < 0 || n > 99) return null
    return n
  }, [id])

  if (error) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-10">
        <div className="rounded-md border border-destructive bg-destructive/10 p-4 text-sm text-destructive">
          Error loading data: {error}
        </div>
        <div className="mt-4">
          <BackLink />
        </div>
      </div>
    )
  }

  if (loading) {
    return <SkillDetailSkeleton />
  }

  if (skillId == null) {
    return <NotFound rawId={id} />
  }

  const skill = skills.find((s) => s.id === skillId)
  if (!skill) {
    return <NotFound rawId={id} />
  }

  const heading = skill.label_english ?? skill.label
  const languageNote = skill.language ? LANGUAGE_NOTES[skill.language] : null
  const isSmallCluster = SMALL_CLUSTER_IDS.has(skillId)
  const examples = skill.example_items ?? []

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <div className="mb-6">
        <BackLink />
      </div>

      {/* Header */}
      <header className="border-b border-border pb-6">
        <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Skill {skill.id} of 100
        </div>
        <h1 className="mt-2 break-words text-2xl font-semibold tracking-tight sm:text-3xl lg:text-4xl">
          {heading}
        </h1>
        {skill.label_english && skill.label !== skill.label_english ? (
          <div className="mt-1 text-sm italic text-muted-foreground">
            Original label: {skill.label}
          </div>
        ) : null}
        <div className="mt-3 flex items-center gap-3">
          <BenchmarkBadge benchmark={skill.primary_benchmark} />
          {languageNote ? (
            <span className="text-xs text-muted-foreground">{languageNote}</span>
          ) : null}
        </div>
        {skill.description &&
        !descriptionDuplicatesLabel(skill.description, skill.label, heading) ? (
          <p className="mt-4 max-w-[68ch] text-sm leading-relaxed text-muted-foreground">
            {skill.description}
          </p>
        ) : null}
      </header>

      {/* Stats */}
      <section className="mt-8">
        <SkillStatsRow
          nItems={skill.n_items}
          alphaMean={skill.alpha_mean}
          thetaVariance={skill.theta_variance_across_LLMs}
        />
      </section>

      {/* Top-10 LLMs chart */}
      <section className="mt-10">
        <h2 className="text-lg font-semibold tracking-tight">
          Top-10 LLMs on this skill
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Ranked by per-skill mastery θ; full range [0, 1].
        </p>
        <p className="mt-1 max-w-[68ch] text-sm text-muted-foreground">
          Differences smaller than the estimation noise are not meaningful; read
          the ranking as a profile, not a precise order.
        </p>
        <div className="mt-4 rounded-md border border-border bg-surface p-3">
          <TopLLMsChart
            models={models}
            skillId={skillId}
            darkMode={darkMode}
          />
        </div>
      </section>

      {/* Example items */}
      <section className="mt-10">
        <h2 className="text-lg font-semibold tracking-tight">Example items</h2>
        {isSmallCluster ? (
          <p className="mt-1 text-sm text-muted-foreground">
            Only {examples.length} example item{examples.length === 1 ? '' : 's'}{' '}
            available — this is a small cluster ({skill.n_items.toLocaleString()}{' '}
            item{skill.n_items === 1 ? '' : 's'}).
          </p>
        ) : (
          <p className="mt-1 text-sm text-muted-foreground">
            Showing {Math.min(examples.length, 3)} of {skill.n_items.toLocaleString()} items in this cluster.
          </p>
        )}

        {examples.length === 0 ? (
          <div className="mt-4 rounded-md border border-dashed border-border p-6 text-sm text-muted-foreground">
            No example items available for this cluster.
          </div>
        ) : (
          <div className="mt-4 flex flex-col gap-3">
            {examples.slice(0, 3).map((item) => (
              <ExampleItemCard key={item.item_idx} item={item} />
            ))}
          </div>
        )}
      </section>

      {/* Footer back link */}
      <div className="mt-12 border-t border-border pt-6">
        <BackLink />
      </div>
    </div>
  )
}
