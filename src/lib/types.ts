// Type definitions matching the SkillEval JSON schemas.

export interface ExampleItem {
  item_idx: number
  benchmark: string
  subtask?: string
  primary_skill?: string
  text: string
}

export interface Model {
  id: number
  name: string
  hf_id?: string
  family: string
  tier: string
  params: number | null
  theta: number[]
}

export interface Skill {
  id: number
  label: string
  description: string
  primary_benchmark: string
  n_items: number
  alpha_mean: number
  theta_variance_across_LLMs: number
  example_items: ExampleItem[]
  // Optional fields present on non-English-origin clusters (e.g. skill 67).
  label_english?: string
  language?: string
}
