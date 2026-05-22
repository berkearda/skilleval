// Type definitions matching the SkillEval JSON schemas.

export interface ExampleItem {
  item_idx: number
  benchmark: string
  text: string
}

export interface Model {
  id: string
  name: string
  family: string
  tier: string
  params: string
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
}
