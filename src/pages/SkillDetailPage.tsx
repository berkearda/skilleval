import { useParams } from 'react-router-dom'

export function SkillDetailPage() {
  const { id } = useParams<{ id: string }>()
  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <h1 className="text-3xl font-semibold tracking-tight">Skill {id}</h1>
      <p className="mt-3 text-muted-foreground">
        Per-skill detail view will land here next phase.
      </p>
    </div>
  )
}
