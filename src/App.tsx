import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'
import { HomePage } from '@/pages/HomePage'
import { TablePage } from '@/pages/TablePage'
import { SkillDetailPage } from '@/pages/SkillDetailPage'
import { AboutPage } from '@/pages/AboutPage'

function App() {
  return (
    <BrowserRouter basename="/skilleval">
      <div className="flex min-h-screen flex-col bg-background text-foreground">
        <Header />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/leaderboard" element={<TablePage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/skill/:id" element={<SkillDetailPage />} />
            <Route path="/overview" element={<Navigate to="/" replace />} />
            <Route path="/browse" element={<Navigate to="/leaderboard" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  )
}

export default App
