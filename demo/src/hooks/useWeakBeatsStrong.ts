import { useEffect, useState } from 'react'
import type { Skill, WeakBeatsStrong } from '@/lib/types'

// The Methodology page needs the skill names but not the 3.6 MB mastery matrix,
// so this loads only skills.json and the chart data.
let cache: Promise<{ wbs: WeakBeatsStrong; skills: Skill[] }> | null = null

function load() {
  if (!cache) {
    const base = import.meta.env.BASE_URL
    const get = async <T,>(name: string): Promise<T> => {
      const r = await fetch(`${base}data/${name}`)
      if (!r.ok) throw new Error(`${name}: ${r.status}`)
      return (await r.json()) as T
    }
    cache = Promise.all([
      get<WeakBeatsStrong>('weak_beats_strong.json'),
      get<Skill[]>('skills.json'),
    ]).then(([wbs, skills]) => ({ wbs, skills }))
  }
  return cache
}

export function useWeakBeatsStrong() {
  const [data, setData] = useState<{ wbs: WeakBeatsStrong; skills: Skill[] } | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let alive = true
    load()
      .then((d) => alive && setData(d))
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [])
  return { data, error }
}
