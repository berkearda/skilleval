import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Header } from '@/components/Header'
import { TablePage } from '@/pages/TablePage'
import { SkillDetailPage } from '@/pages/SkillDetailPage'
import { AboutPage } from '@/pages/AboutPage'

function App() {
  return (
    <BrowserRouter basename="/skilleval">
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<TablePage />} />
            <Route path="/skill/:id" element={<SkillDetailPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
