import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'
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
            <Route path="/" element={<TablePage />} />
            <Route path="/skill/:id" element={<SkillDetailPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  )
}

export default App
