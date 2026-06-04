// Color helpers for the SkillEval table view.
// All helpers return inline-style colors (not Tailwind classes) so the
// per-cell mastery gradient can interpolate smoothly across 100 levels
// without needing 100 Tailwind utility classes.

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x)

// Viridis 9-stop, low -> high (light surface). Perceptually uniform and
// colorblind-safe (no red/green confusion).
const VIRIDIS = [
  [68, 1, 84],
  [72, 45, 123],
  [59, 82, 139],
  [44, 114, 142],
  [33, 145, 140],
  [40, 174, 128],
  [94, 201, 98],
  [173, 220, 48],
  [253, 231, 37],
]

// Cividis-like, low end lifted to ~#243044 so it stays distinct from the
// dark page background (~#0f0f10). low -> high (dark surface).
const CIVIDIS_DARK = [
  [36, 48, 68],
  [38, 62, 99],
  [55, 80, 110],
  [97, 101, 111],
  [125, 124, 120],
  [155, 148, 118],
  [188, 174, 108],
  [222, 201, 88],
  [254, 232, 56],
]

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

function rampColor(stops: number[][], t: number): string {
  const x = clamp01(t) * (stops.length - 1)
  const i = Math.min(Math.floor(x), stops.length - 2)
  const f = x - i
  const r = Math.round(lerp(stops[i][0], stops[i + 1][0], f))
  const g = Math.round(lerp(stops[i][1], stops[i + 1][1], f))
  const b = Math.round(lerp(stops[i][2], stops[i + 1][2], f))
  return `rgb(${r}, ${g}, ${b})`
}

/**
 * Mastery cell background color. theta in [0,1] -> a perceptually uniform
 * sequential ramp (viridis in light, cividis-with-lifted-floor in dark).
 */
export function getMasteryColor(theta: number, darkMode: boolean): string {
  return rampColor(darkMode ? CIVIDIS_DARK : VIRIDIS, theta)
}

function relativeLuminance(rgb: string): number {
  const m = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/.exec(rgb)
  if (!m) return 0.5
  const ch = [+m[1], +m[2], +m[3]].map((v) => {
    const s = v / 255
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4)
  })
  return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
}

/**
 * Pick text color (near-white or near-black) from the WCAG relative
 * luminance of the actual interpolated cell color, so the value stays
 * readable across the full ramp including the crossover band.
 */
export function getMasteryTextColor(theta: number, darkMode: boolean): string {
  const L = relativeLuminance(getMasteryColor(theta, darkMode))
  return L < 0.4 ? '#f8fafc' : '#0b1220'
}

/** Neutral surface for a missing (null/undefined) theta — never "low mastery". */
export const NULL_CELL_BG_LIGHT = 'hsl(240 5% 94%)'
export const NULL_CELL_BG_DARK = 'hsl(240 4% 16%)'

/** Convert an `rgb(r, g, b)` string to `rgba(r, g, b, a)` for translucency. */
export function rgbToRgba(rgb: string, alpha: number): string {
  const m = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/.exec(rgb)
  if (!m) return rgb
  return `rgba(${m[1]}, ${m[2]}, ${m[3]}, ${alpha})`
}

const FAMILY_COLORS: Record<string, string> = {
  Llama: '#E69F00', // okabe-orange
  Qwen: '#56B4E9', // okabe-skyblue
  Mistral: '#009E73', // okabe-green
  Gemma: '#CC79A7', // okabe-purple
  Phi: '#B7A100', // darker yellow for contrast on white
  Mixtral: '#D55E00', // okabe-vermillion
  Other: '#64748b', // slate-500
}

export function getFamilyColor(family: string): string {
  return FAMILY_COLORS[family] ?? FAMILY_COLORS.Other
}

const BENCHMARK_COLORS: Record<string, string> = {
  MATH: '#0072B2', // okabe-blue
  BBH: '#E69F00', // okabe-orange
  GPQA: '#009E73', // okabe-green
  MuSR: '#D55E00', // okabe-vermillion
  IFEval: '#CC79A7', // okabe-purple
  Other: '#64748b', // slate-500
}

export function getBenchmarkColor(benchmark: string): string {
  return BENCHMARK_COLORS[benchmark] ?? BENCHMARK_COLORS.Other
}

export const FAMILY_LIST = [
  'Llama',
  'Qwen',
  'Mistral',
  'Gemma',
  'Phi',
  'Mixtral',
  'Other',
] as const

export const TIER_LIST = ['Small', 'Mid', 'Large', 'Other'] as const

export const BENCHMARK_LIST = ['MATH', 'BBH', 'GPQA', 'MuSR', 'IFEval'] as const
