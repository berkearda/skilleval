import { useSkillEvalData } from '@/hooks/useSkillEvalData'

export function TablePage() {
  const { models, skills, loading, error } = useSkillEvalData()

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <h1 className="text-3xl font-semibold tracking-tight">
        SkillEval table view coming soon
      </h1>
      <p className="mt-3 text-muted-foreground">
        Placeholder for the skill x model theta table.
      </p>

      <div className="mt-8 rounded-md border border-border p-4 text-sm">
        {loading && <span>Loading placeholder data...</span>}
        {error && (
          <span className="text-destructive">
            Error loading data: {error}
          </span>
        )}
        {!loading && !error && (
          <span className="text-muted-foreground">
            Loaded {models.length} models and {skills.length} skills.
          </span>
        )}
      </div>

      {/* Smoke-test for Okabe-Ito Tailwind color */}
      <div
        aria-hidden
        className="mt-6 h-2 w-24 rounded-full bg-okabe-blue"
        title="okabe-blue swatch"
      />
    </div>
  )
}
