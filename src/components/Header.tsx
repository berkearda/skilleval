import { Link, NavLink } from 'react-router-dom'
import { ThemeToggle } from '@/components/ThemeToggle'
import { cn } from '@/lib/utils'

const navItems = [
  { to: '/browse', label: 'Browse' },
  { to: '/about', label: 'About' },
]

export function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <div className="mx-auto flex max-w-screen-2xl items-center justify-between px-6 py-3">
        <Link to="/" className="flex items-baseline gap-2">
          <span className="text-base font-semibold tracking-tight">SkillEval</span>
          <span className="hidden text-xs text-muted-foreground sm:inline">
            skill-level ability estimates for LLMs
          </span>
        </Link>
        <nav className="flex items-center gap-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
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
          <div className="ml-2">
            <ThemeToggle />
          </div>
        </nav>
      </div>
    </header>
  )
}
