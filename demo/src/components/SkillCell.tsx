import { memo } from 'react'
import {
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
}

function SkillCellInner({ theta, darkMode, skillLabel, fontSize = 10 }: SkillCellProps) {
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
  const bg = getMasteryColor(theta, darkMode)
  const fg = getMasteryTextColor(theta, darkMode)
  return (
    <div
      title={`${skillLabel}: theta = ${theta.toFixed(3)}`}
      className="tabular flex h-full items-center justify-center font-mono"
      style={{ backgroundColor: bg, color: fg, fontSize: `${fontSize}px` }}
    >
      {theta.toFixed(2)}
    </div>
  )
}

export const SkillCell = memo(SkillCellInner)
