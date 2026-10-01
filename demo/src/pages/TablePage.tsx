import { useEffect, useState } from 'react'
import { useSkillEvalData } from '@/hooks/useSkillEvalData'
import { usePageTitle } from '@/hooks/usePageTitle'
import { ModelSkillTable } from '@/components/ModelSkillTable'

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

function TableSkeleton() {
  return (
    <div className="flex-1 overflow-hidden border-t border-border bg-background p-4">
      {/* Header strip */}
      <div className="mb-3 flex gap-3">
        <div className="h-8 w-64 animate-pulse rounded bg-muted" />
        <div className="h-8 w-32 animate-pulse rounded bg-muted" />
        <div className="h-8 w-32 animate-pulse rounded bg-muted" />
      </div>
      {/* Row skeletons */}
      <div className="flex flex-col gap-1.5">
        {Array.from({ length: 14 }).map((_, i) => (
          <div
            key={i}
            className="h-8 w-full animate-pulse rounded bg-muted"
            style={{ opacity: 1 - i * 0.04 }}
          />
        ))}
      </div>
    </div>
  )
}

export function TablePage() {
  usePageTitle('Leaderboard · SkillEval')
  const { models, skills, loading, error } = useSkillEvalData()
  const darkMode = useDarkMode()

  return (
    // Fills the space the shell leaves under the header (App.tsx); only the
    // table scrolls.
    <div className="page flex min-h-0 flex-1 flex-col pb-4">
      <div className="pt-6">
        <h1 className="text-2xl font-semibold tracking-tight">Leaderboard</h1>
          <p className="mt-1 text-sm text-muted-foreground tabular">
            {models.length > 0 && skills.length > 0
              ? `Mastery of ${models.length.toLocaleString()} models on ${skills.length} skills, from 0 to 1. A fixed snapshot of the paper's model.`
              : "Mastery of 3,811 models on 100 skills, from 0 to 1. A fixed snapshot of the paper's model."}
          </p>
      </div>

      {error ? (
        <div className="m-6 rounded-md border border-destructive bg-destructive/10 p-4 text-sm text-destructive">
          Error loading data: {error}
        </div>
      ) : loading ? (
        <TableSkeleton />
      ) : (
        <ModelSkillTable
          models={models}
          skills={skills}
          darkMode={darkMode}
        />
      )}
    </div>
  )
}
