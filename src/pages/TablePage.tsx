import { useEffect, useState } from 'react'
import { useSkillEvalData } from '@/hooks/useSkillEvalData'
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
  const { models, skills, loading, error } = useSkillEvalData()
  const darkMode = useDarkMode()

  return (
    // Subtract header (~65px) and footer (~52px) so the table fills the
    // remaining viewport without forcing page-level vertical scroll.
    <div className="flex h-[calc(100vh-117px)] flex-col overflow-hidden">
      <div className="border-b border-border bg-background px-6 py-4">
        <h1 className="text-2xl font-semibold tracking-tight">
          SkillEval Browser
        </h1>
        <p className="text-sm text-muted-foreground">
          {models.length > 0 && skills.length > 0
            ? `${models.length.toLocaleString()} LLMs × ${skills.length} named skills`
            : '3,811 LLMs × 100 named skills'}
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
