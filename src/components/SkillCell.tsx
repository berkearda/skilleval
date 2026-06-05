import { memo } from 'react'
import {
  getDeltaColor,
  getDeltaTextColor,
  getMasteryColor,
  getMasteryTextColor,
  NULL_CELL_BG_DARK,
  NULL_CELL_BG_LIGHT,
} from '@/lib/colors'

interface SkillCellProps {
  theta: number | null | undefined
  darkMode: boolean
  skillLabel: string
  /** Font size in px, controlled by the table's density setting. */
  fontSize?: number
  /** When set, render RELATIVE mode: this is theta minus the skill's
   * population mean, shown signed on the diverging ramp. */
  delta?: number | null
}

function SkillCellInner({
  theta,
  darkMode,
  skillLabel,
  fontSize = 10,
  delta,
}: SkillCellProps) {
  // 0 is a real low value here; only null/undefined means "no data".
  if (theta == null) {
    return (
      <div
        title={`${skillLabel}: no data`}
        className="tabular flex h-full items-center justify-center font-mono text-muted-foreground"
        style={{
          backgroundColor: darkMode ? NULL_CELL_BG_DARK : NULL_CELL_BG_LIGHT,
          fontSize: `${fontSize}px`,
        }}
      >
        —
      </div>
    )
  }
  if (delta != null) {
    const bg = getDeltaColor(delta, darkMode)
    const fg = getDeltaTextColor(delta, darkMode)
    return (
      <div
        title={`${skillLabel}: theta = ${theta.toFixed(3)}, ${
          delta >= 0 ? '+' : ''
        }${delta.toFixed(3)} vs the population mean. Click to open the skill.`}
        className="tabular flex h-full items-center justify-center font-mono"
        style={{ backgroundColor: bg, color: fg, fontSize: `${fontSize}px` }}
      >
        {delta >= 0 ? '+' : '−'}
        {Math.abs(delta).toFixed(2)}
      </div>
    )
  }
  const bg = getMasteryColor(theta, darkMode)
  const fg = getMasteryTextColor(theta, darkMode)
  return (
    <div
      title={`${skillLabel}: theta = ${theta.toFixed(3)}. Click to open the skill.`}
      className="tabular flex h-full items-center justify-center font-mono"
      style={{ backgroundColor: bg, color: fg, fontSize: `${fontSize}px` }}
    >
      {theta.toFixed(2)}
    </div>
  )
}

export const SkillCell = memo(SkillCellInner)
