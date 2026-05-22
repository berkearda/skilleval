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

export function TablePage() {
  const { models, skills, loading, error } = useSkillEvalData()
  const darkMode = useDarkMode()

  return (
    // Subtract header height (~65px) so the table fills the remaining viewport.
    <div className="flex h-[calc(100vh-65px)] flex-col">
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
        <div className="flex flex-1 items-center justify-center text-muted-foreground">
          Loading SkillEval data...
        </div>
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
