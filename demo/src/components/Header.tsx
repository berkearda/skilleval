import { Link, NavLink } from 'react-router-dom'
import { ThemeToggle } from '@/components/ThemeToggle'
import { cn } from '@/lib/utils'

const navItems = [
  { to: '/', label: 'Home', end: true },
  { to: '/leaderboard', label: 'Leaderboard' },
  { to: '/skills', label: 'Skills' },
  { to: '/compare', label: 'Compare' },
  { to: '/about', label: 'Methodology' },
]

export function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-border bg-surface/95 backdrop-blur supports-[backdrop-filter]:bg-surface/80">
      <div className="mx-auto flex max-w-screen-2xl items-center justify-between px-6 py-3">
        <Link to="/" className="flex items-baseline gap-2">
          <span className="font-display text-base font-semibold tracking-tight">
            SkillEval
          </span>
          <span className="hidden border-l border-border pl-2 text-xs text-muted-foreground sm:inline">
            ETH Zürich
          </span>
        </Link>
        <nav className="flex items-center gap-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                cn(
                  'rounded-md px-3 py-1.5 text-sm transition-colors duration-150',
                  isActive
                    ? 'font-medium text-brand'
                    : 'text-muted-foreground hover:text-foreground'
                )
              }
            >
              {item.label}
            </NavLink>
          ))}
          <Link
            to="/leaderboard"
            className="ml-2 hidden rounded-md bg-brand px-3 py-1.5 text-sm font-medium text-brand-foreground transition-colors hover:bg-brand/90 sm:inline-flex"
          >
            Explore all 3,811 models
          </Link>
          <div className="ml-2">
            <ThemeToggle />
          </div>
        </nav>
      </div>
    </header>
  )
}
