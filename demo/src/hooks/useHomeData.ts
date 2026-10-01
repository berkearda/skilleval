import { useEffect, useState } from 'react'

// public/data/home.json (scripts/build_site_data.py): only what the Home page
// shows, so the first visit does not download the full 3.6 MB matrix.
export interface HomeModel {
  id: number
  name: string
  family: string
  tier: string
  params: number | null
  accuracy?: number
  theta: number[]
  meanTheta: number
}

export interface HomeData {
  top: HomeModel[]
  compare: string | null
  mosaic_skills: { id: number; label: string }[]
}

let cache: Promise<HomeData> | null = null

function load(): Promise<HomeData> {
  if (!cache) {
    cache = fetch(`${import.meta.env.BASE_URL}data/home.json`).then((r) => {
      if (!r.ok) throw new Error(`home.json: ${r.status}`)
      return r.json() as Promise<HomeData>
    })
  }
  return cache
}

export function useHomeData() {
  const [data, setData] = useState<HomeData | null>(null)
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
  return { data, error, loading: data === null && error === null }
}
