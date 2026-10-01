import { useState } from 'react'
import { Link, NavLink } from 'react-router-dom'
import { Menu, X } from 'lucide-react'
import { ThemeToggle } from '@/components/ThemeToggle'
import { cn } from '@/lib/utils'

const navItems = [
  { to: '/', label: 'Home', end: true },
  { to: '/leaderboard', label: 'Leaderboard' },
  { to: '/skills', label: 'Skills' },
  { to: '/compare', label: 'Compare' },
  { to: '/about', label: 'Method' },
]

const linkCls = ({ isActive }: { isActive: boolean }) =>
  cn(
    'rounded-md px-3 py-1.5 text-sm transition-colors duration-150',
    isActive ? 'font-medium text-brand' : 'text-muted-foreground hover:text-foreground'
  )

export function Header() {
  // Below md the links fold into a menu, so the header never runs off a phone screen.
  const [open, setOpen] = useState(false)
  const close = () => setOpen(false)

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-surface/95 backdrop-blur supports-[backdrop-filter]:bg-surface/80">
      <div className="page flex items-center justify-between gap-3 py-3">
        <Link to="/" onClick={close} className="flex items-baseline gap-2">
          <span className="font-display text-base font-semibold tracking-tight">
            SkillEval
          </span>
          <span className="hidden border-l border-border pl-2 text-xs text-muted-foreground sm:inline">
            ETH Zürich
          </span>
        </Link>
        <div className="flex items-center gap-1">
          <nav className="hidden items-center gap-1 md:flex" aria-label="Main">
            {navItems.map((item) => (
              <NavLink key={item.to} to={item.to} end={item.end} className={linkCls}>
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-1 sm:ml-2">
            <ThemeToggle />
          </div>
          <button
            type="button"
            className="inline-flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground md:hidden"
            aria-label={open ? 'Close menu' : 'Open menu'}
            aria-expanded={open}
            aria-controls="mobile-nav"
            onClick={() => setOpen((o) => !o)}
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>
      {open ? (
        <nav
          id="mobile-nav"
          aria-label="Main"
          className="page flex flex-col border-t border-border py-2 md:hidden"
        >
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              onClick={close}
              className={({ isActive }) => cn(linkCls({ isActive }), 'py-2.5 text-base')}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      ) : null}
    </header>
  )
}
