import React, { useState, useEffect } from 'react'
import { Brain, BookOpen, Zap, Layers, GitBranch, Settings, FileText, Eye, CheckCircle } from './Icons'

const PIPELINE_STEPS = [
  { id: 'analyse', labels: ['analyse_context', 'context_analysis', 'context analysis'], name: 'Context Analysis', Icon: Brain },
  { id: 'kb', labels: ['kb_case_learner', 'kb_case_learning', 'kb learner'], name: 'KB Learner', Icon: BookOpen },
  { id: 'paradigm', labels: ['paradigm_mixer', 'paradigm_mixing', 'paradigm mixer'], name: 'Paradigm Mixer', Icon: Zap },
  { id: 'decompose', labels: ['decompose_states', 'state_decomposition', 'state decomposition'], name: 'State Decomposition', Icon: Layers },
  { id: 'prioritise', labels: ['prioritise_cases', 'case_prioritisation', 'prioritising cases'], name: 'Prioritise Cases', Icon: GitBranch },
  { id: 'handlers', labels: ['write_case_handlers', 'case_writing', 'case handlers', 'seed_kb', 'kb seeded'], name: 'Case Handlers', Icon: Settings },
  { id: 'assemble', labels: ['assemble_prompts', 'prompt_assembly', 'prompt assembly'], name: 'Prompt Assembly', Icon: FileText },
  { id: 'review', labels: ['review_consistency'], name: 'Consistency Review', Icon: Eye },
]

export default function PipelineVisualizer({ progress }) {
  const [activeIndex, setActiveIndex] = useState(0)
  const [isComplete, setIsComplete] = useState(false)

  useEffect(() => {
    const p = (progress || '').toLowerCase()
    
    if (p.includes('starting')) {
      setActiveIndex(0)
      setIsComplete(false)
      return
    }

    if (p === 'pipeline complete') {
      setActiveIndex(PIPELINE_STEPS.length)
      setIsComplete(true)
      return
    }

    let matchIdx = -1
    PIPELINE_STEPS.forEach((step, idx) => {
      if (step.labels.some(label => p.includes(label.toLowerCase()))) {
        matchIdx = Math.max(matchIdx, idx)
      }
    })

    if (matchIdx > -1) {
      setActiveIndex(prev => Math.max(prev, matchIdx))
    }
  }, [progress])

  return (
    <div className="w-full flex-col py-2 overflow-hidden">
      <div className="flex items-center justify-between mb-8 mt-2 relative">
        {/* Background Track */}
        <div className="absolute left-6 right-6 top-1/2 -translate-y-1/2 h-[2px] bg-[var(--color-surface-3)] z-0 rounded-full" />
        
        {/* Fill Track */}
        <div 
          className="absolute left-6 top-1/2 -translate-y-1/2 h-[2px] bg-gradient-to-r from-violet-500 to-indigo-500 z-0 rounded-full transition-all duration-700 ease-in-out"
          style={{ width: `calc(${Math.min(activeIndex, PIPELINE_STEPS.length - 1)} * 100% / ${PIPELINE_STEPS.length - 1} - 12px)` }}
        />

        {PIPELINE_STEPS.map((step, idx) => {
          const isActive = idx === activeIndex
          const isPast = idx < activeIndex || isComplete
          const Icon = isPast ? CheckCircle : step.Icon

          return (
            <div key={step.id} className="relative z-10 flex flex-col items-center group w-12">
              <div 
                className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-500 ${
                  isActive
                    ? 'bg-gradient-to-br from-violet-600 to-indigo-600 shadow-[0_0_20px_rgba(139,92,246,0.6)] border-2 border-white/20 transform scale-110' 
                    : isPast
                      ? 'bg-[var(--color-surface-2)] text-[var(--color-success)] border border-[var(--color-success)]/40 shadow-[0_0_10px_rgba(34,197,94,0.15)]'
                      : 'bg-[var(--color-surface)] text-[var(--color-text-3)] border border-[var(--color-border)]'
                }`}
              >
                {isActive && <div className="absolute inset-0 rounded-full bg-violet-400 opacity-20 pulse-soft"></div>}
                <Icon size={18} className={isActive ? 'text-white' : ''} />
              </div>

              {/* Tooltip / Label */}
              <div 
                className={`absolute top-12 whitespace-nowrap text-[10px] font-semibold text-center transition-all duration-300 ${
                  isActive ? 'text-[var(--color-accent-light)] translate-y-0 opacity-100' : 
                  isPast ? 'text-[var(--color-text-2)] translate-y-0 opacity-100 hidden sm:block' : 
                  'text-[var(--color-text-3)] opacity-0 group-hover:opacity-100 translate-y-2 group-hover:translate-y-0'
                }`}
              >
                {step.name}
              </div>
            </div>
          )
        })}
      </div>

      <div className="bg-[var(--color-surface-2)] rounded-lg px-4 py-3 border border-[var(--color-border)] shadow-inner text-center mt-6">
        <div className="flex items-center justify-center gap-2.5">
          {!isComplete && (
            <div className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-indigo-500"></span>
            </div>
          )}
          <span className="text-sm font-medium text-[var(--color-text)] tracking-wide">
            {progress || 'Initializing Pipeline...'}
          </span>
        </div>
      </div>
    </div>
  )
}
