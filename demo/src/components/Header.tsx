import { Link } from 'react-router-dom'
import { ThemeToggle } from '@/components/ThemeToggle'

export function Header() {
  return (
    <header className="border-b border-border bg-background">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link to="/" className="text-lg font-semibold tracking-tight">
          SkillEval
        </Link>
        <nav className="flex items-center gap-4">
          <Link
            to="/about"
            className="text-sm text-muted-foreground hover:text-foreground"
          >
            About
          </Link>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  )
}
