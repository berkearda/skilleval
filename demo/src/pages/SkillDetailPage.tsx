import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { useSkillEvalData } from '@/hooks/useSkillEvalData'
import { usePageTitle } from '@/hooks/usePageTitle'
import { displaySkillLabel } from '@/lib/labels'
import { BenchmarkBadge } from '@/components/BenchmarkBadge'
import { getBenchmarkColor } from '@/lib/colors'
import { SkillStatsRow } from '@/components/SkillStatsRow'
import { TopLLMsChart } from '@/components/TopLLMsChart'
import { ExampleItemCard } from '@/components/ExampleItemCard'

// Clusters where n_items < 3 — flagged in the orchestration plan so we can
// render an explicit "small cluster" note instead of padding with empty cards.
const SMALL_CLUSTER_IDS = new Set([
  50, 53, 56, 63, 67, 69, 75, 79, 80, 82, 85, 94, 95,
])

const LANGUAGE_NOTES: Record<string, string> = {
  vi: 'Vietnamese-origin skill cluster. The heading shown is the English translation.',
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
      to="/skills"
      className={`inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground ${className}`}
    >
      <ArrowLeft className="h-3.5 w-3.5" />
      All skills
    </Link>
  )
}

/** Which benchmarks the items needing this skill come from. */
function BenchmarkMix({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts)
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1])
  const total = entries.reduce((s, [, n]) => s + n, 0)
  if (total === 0) return null
  return (
    <div className="max-w-2xl">
      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-muted">
        {entries.map(([b, n]) => (
          <div
            key={b}
            style={{ width: `${(100 * n) / total}%`, backgroundColor: getBenchmarkColor(b) }}
            title={`${b}: ${n.toLocaleString()} items`}
          />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted-foreground">
        {entries.map(([b, n]) => (
          <span key={b} className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: getBenchmarkColor(b) }} />
            {b}
            <span className="tabular text-foreground">{n.toLocaleString()}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

function NotFound({ rawId }: { rawId: string | undefined }) {
  return (
    <div className="page py-16">
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
    <div className="page py-8">
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
  const titled = skillId == null ? undefined : skills.find((s) => s.id === skillId)
  usePageTitle(titled ? `${displaySkillLabel(titled.label)} · SkillEval` : 'Skill · SkillEval')

  if (error) {
    return (
      <div className="page py-10">
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

  const heading = displaySkillLabel(skill.label_english ?? skill.label)
  const languageNote = skill.language ? LANGUAGE_NOTES[skill.language] : null
  const isSmallCluster = SMALL_CLUSTER_IDS.has(skillId)
  const examples = skill.example_items ?? []

  return (
    <div className="page py-8">
      <nav aria-label="Breadcrumb" className="mb-6 text-sm text-muted-foreground">
        <Link to="/skills" className="hover:text-foreground">
          Skills
        </Link>
        <span className="mx-2">/</span>
        <span>Skill #{skill.id}</span>
      </nav>

      {/* Header */}
      <header className="border-b border-border pb-6">
        <h1 className="break-words text-2xl font-semibold tracking-tight sm:text-3xl lg:text-4xl">
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

      {/* where the items come from */}
      {skill.items_by_benchmark ? (
        <section className="mt-8">
          <h2 className="text-sm font-semibold">Where its items come from</h2>
          <div className="mt-3">
            <BenchmarkMix counts={skill.items_by_benchmark} />
          </div>
        </section>
      ) : null}

      {/* top 10 models */}
      <section className="mt-10">
        <h2 className="text-lg font-semibold tracking-tight">
          Top 10 models on this skill
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Ranked by mastery on this skill, from 0 to 1.
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
        {examples.length > 0 ? (
          <p className="mt-1 text-sm text-muted-foreground">
            {isSmallCluster
              ? `This skill has only ${skill.n_items} item${skill.n_items === 1 ? '' : 's'}.`
              : `Showing ${Math.min(examples.length, 3)} of the ${skill.n_items.toLocaleString()} items that need this skill.`}
            {examples.length < Math.min(3, skill.n_items) ? ' GPQA questions are not shown.' : ''}
          </p>
        ) : null}

        {examples.length === 0 ? (
          <div className="mt-4 rounded-md border border-dashed border-border p-6 text-sm text-muted-foreground">
            {skill.primary_benchmark === 'GPQA'
              ? 'GPQA questions are not shown: the GPQA authors ask that its questions not be posted in plain text.'
              : 'No example items can be shown for this skill.'}
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
