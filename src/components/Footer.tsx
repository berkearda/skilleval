export function Footer() {
  return (
    <footer className="border-t border-border bg-background">
      <div className="mx-auto flex max-w-6xl flex-col items-start gap-2 px-6 py-4 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
        <div>Last updated 2026-05-22</div>
        <div className="text-center">
          SkillEval research framework, based on Neural Cognitive Diagnostic
          Models
        </div>
        <nav className="flex items-center gap-4">
          <a
            href="#"
            className="hover:text-foreground"
            aria-label="Source code"
          >
            Code
          </a>
          <a href="#" className="hover:text-foreground" aria-label="Paper">
            Paper
          </a>
        </nav>
      </div>
    </footer>
  )
}
