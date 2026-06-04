interface StatCardProps {
  label: string
  value: string
  hint?: string
}

function StatCard({ label, value, hint }: StatCardProps) {
  return (
    <div className="flex-1 rounded-md border border-border bg-surface px-4 py-3">
      <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="tabular mt-1 text-2xl font-semibold tabular-nums text-foreground">
        {value}
      </div>
      {hint ? (
        <div className="mt-0.5 text-xs text-muted-foreground">{hint}</div>
      ) : null}
    </div>
  )
}

interface SkillStatsRowProps {
  nItems: number
  alphaMean: number
  thetaVariance: number
}

export function SkillStatsRow({
  nItems,
  alphaMean,
  thetaVariance,
}: SkillStatsRowProps) {
  return (
    <div className="flex flex-col gap-3 sm:flex-row">
      <StatCard
        label="Items in cluster"
        value={nItems.toLocaleString()}
        hint={nItems === 1 ? 'singleton cluster' : undefined}
      />
      <StatCard
        label="Discriminability (α mean)"
        value={alphaMean.toFixed(3)}
      />
      <StatCard
        label="Mastery variance across LLMs"
        value={thetaVariance.toFixed(4)}
      />
    </div>
  )
}
