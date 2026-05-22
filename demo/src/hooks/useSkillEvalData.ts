import { useEffect, useState } from 'react'
import type { Model, Skill } from '@/lib/types'

interface SkillEvalData {
  models: Model[]
  skills: Skill[]
  loading: boolean
  error: string | null
}

// Module-level cache so repeated mounts don't refetch.
let cachedModels: Model[] | null = null
let cachedSkills: Skill[] | null = null
let inflight: Promise<{ models: Model[]; skills: Skill[] }> | null = null

async function loadAll(): Promise<{ models: Model[]; skills: Skill[] }> {
  if (cachedModels && cachedSkills) {
    return { models: cachedModels, skills: cachedSkills }
  }
  if (inflight) return inflight

  const base = import.meta.env.BASE_URL
  inflight = (async () => {
    const [thetaResp, skillsResp] = await Promise.all([
      fetch(`${base}data/theta_matrix.json`),
      fetch(`${base}data/skills.json`),
    ])
    if (!thetaResp.ok) throw new Error(`theta_matrix.json: ${thetaResp.status}`)
    if (!skillsResp.ok) throw new Error(`skills.json: ${skillsResp.status}`)
    // The JSON files are stored as bare arrays at the root.
    const thetaJson = (await thetaResp.json()) as Model[] | { models: Model[] }
    const skillsJson = (await skillsResp.json()) as Skill[] | { skills: Skill[] }
    cachedModels = Array.isArray(thetaJson) ? thetaJson : thetaJson.models
    cachedSkills = Array.isArray(skillsJson) ? skillsJson : skillsJson.skills
    return { models: cachedModels, skills: cachedSkills }
  })()

  return inflight
}

export function useSkillEvalData(): SkillEvalData {
  const [models, setModels] = useState<Model[]>(cachedModels ?? [])
  const [skills, setSkills] = useState<Skill[]>(cachedSkills ?? [])
  const [loading, setLoading] = useState(!(cachedModels && cachedSkills))
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    if (cachedModels && cachedSkills) {
      setLoading(false)
      return
    }
    loadAll()
      .then(({ models, skills }) => {
        if (cancelled) return
        setModels(models)
        setSkills(skills)
        setLoading(false)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setError(err instanceof Error ? err.message : String(err))
        setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return { models, skills, loading, error }
}
