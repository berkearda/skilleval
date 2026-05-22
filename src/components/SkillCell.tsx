import { memo } from 'react'
import { getMasteryColor, getMasteryTextColor } from '@/lib/colors'

interface SkillCellProps {
  theta: number
  darkMode: boolean
  skillLabel: string
}

function SkillCellInner({ theta, darkMode, skillLabel }: SkillCellProps) {
  const bg = getMasteryColor(theta, darkMode)
  const fg = getMasteryTextColor(theta, darkMode)
  return (
    <div
      title={`${skillLabel}: theta = ${theta.toFixed(3)}`}
      className="flex h-full items-center justify-center font-mono"
      style={{ backgroundColor: bg, color: fg, fontSize: '10px' }}
    >
      {theta.toFixed(2)}
    </div>
  )
}

export const SkillCell = memo(SkillCellInner)
