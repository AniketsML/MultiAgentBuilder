import { useState } from 'react'
import ContextInput from './components/ContextInput'
import PromptReview from './components/PromptReview'
import OutputSummary from './components/OutputSummary'
import KBBrowser from './components/KBBrowser'
import { MasterChat } from './components/MasterChat'
import { Sparkles, Eye, CheckCircle, BookOpen, Plus, X } from './components/Icons'

const SCREENS = { INPUT: 'input', REVIEW: 'review', OUTPUT: 'output' }

const NAV_ITEMS = [
  { key: 'input', label: 'Input', Icon: Sparkles },
  { key: 'review', label: 'Review', Icon: Eye },
  { key: 'output', label: 'Output', Icon: CheckCircle },
]

export default function App() {
  const [screen, setScreen] = useState(SCREENS.INPUT)
  const [runId, setRunId] = useState(null)
  const [result, setResult] = useState(null)
  const [showKB, setShowKB] = useState(false)

  const handleRunComplete = (id, data) => {
    setRunId(id)
    setResult(data)
    setScreen(SCREENS.REVIEW)
  }

  const handleReviewComplete = (updatedDrafts) => {
    setResult(prev => ({ ...prev, drafts: updatedDrafts }))
    setScreen(SCREENS.OUTPUT)
  }

  const handleNewRun = () => {
    setRunId(null)
    setResult(null)
    setScreen(SCREENS.INPUT)
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-[var(--color-bg)]/80 backdrop-blur-xl border-b border-[var(--color-border)]">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center">
              <Sparkles size={14} className="text-white" />
            </div>
            <span className="text-sm font-semibold text-[var(--color-text)] tracking-tight">Flow Builder</span>
          </div>

          {/* Center nav */}
          <nav className="flex items-center bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)] p-0.5">
            {NAV_ITEMS.map((item, i) => {
              const active = screen === item.key
              return (
                <button
                  key={item.key}
                  className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                    active
                      ? 'bg-[var(--color-surface-2)] text-[var(--color-text)] shadow-sm'
                      : 'text-[var(--color-text-3)] hover:text-[var(--color-text-2)]'
                  }`}
                >
                  <item.Icon size={13} />
                  <span>{item.label}</span>
                </button>
              )
            })}
          </nav>

          <div className="flex items-center gap-2">
            <button onClick={() => setShowKB(!showKB)} className="btn btn-ghost text-xs py-1.5 px-3">
              {showKB ? <><X size={13} /> Close</> : <><BookOpen size={13} /> KB</>}
            </button>
            {screen !== SCREENS.INPUT && (
              <button onClick={handleNewRun} className="btn btn-ghost text-xs py-1.5 px-3">
                <Plus size={13} /> New
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main */}
      <div className="flex">
        <main className={`flex-1 max-w-3xl mx-auto px-6 py-8 transition-all duration-200 ${showKB ? 'mr-[380px]' : ''}`}>
          {screen === SCREENS.INPUT && <ContextInput onRunComplete={handleRunComplete} />}
          {screen === SCREENS.REVIEW && result && (
            <PromptReview runId={runId} result={result} onComplete={handleReviewComplete} />
          )}
          {screen === SCREENS.OUTPUT && result && (
            <OutputSummary result={result} onNewRun={handleNewRun} />
          )}
        </main>

        {showKB && (
          <aside className="fixed right-0 top-14 bottom-0 w-[380px] border-l border-[var(--color-border)] bg-[var(--color-bg)] overflow-y-auto z-30">
            <KBBrowser />
          </aside>
        )}
      </div>
      <MasterChat />
    </div>
  )
}
