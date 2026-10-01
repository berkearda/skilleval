import { lazy, Suspense } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'

// Each page is its own chunk, so a first visit downloads only the page it opens.
const HomePage = lazy(() => import('@/pages/HomePage').then((m) => ({ default: m.HomePage })))
const TablePage = lazy(() => import('@/pages/TablePage').then((m) => ({ default: m.TablePage })))
const SkillsPage = lazy(() => import('@/pages/SkillsPage').then((m) => ({ default: m.SkillsPage })))
const ComparePage = lazy(() => import('@/pages/ComparePage').then((m) => ({ default: m.ComparePage })))
const ModelPage = lazy(() => import('@/pages/ModelPage').then((m) => ({ default: m.ModelPage })))
const SkillDetailPage = lazy(() =>
  import('@/pages/SkillDetailPage').then((m) => ({ default: m.SkillDetailPage }))
)
const AboutPage = lazy(() => import('@/pages/AboutPage').then((m) => ({ default: m.AboutPage })))

function PageFallback() {
  return (
    <div className="page py-12">
      <div className="h-64 animate-pulse rounded-lg bg-muted" />
    </div>
  )
}

function App() {
  return (
    <BrowserRouter basename="/skilleval">
      <div className="flex min-h-screen flex-col bg-background text-foreground">
        <Header />
        <main className="flex-1">
          <Suspense fallback={<PageFallback />}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/leaderboard" element={<TablePage />} />
              <Route path="/about" element={<AboutPage />} />
              <Route path="/skills" element={<SkillsPage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/model/:id" element={<ModelPage />} />
              <Route path="/skill/:id" element={<SkillDetailPage />} />
              <Route path="/overview" element={<Navigate to="/" replace />} />
              <Route path="/browse" element={<Navigate to="/leaderboard" replace />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  )
}

export default App
