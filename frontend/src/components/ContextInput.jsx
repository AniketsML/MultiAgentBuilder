import { useState, useRef } from 'react'
import StateList from './StateList'
import { startRun, streamRunProgress, uploadDocument, extractStates } from '../api'
import { FileText, Upload, Brain, Sparkles, Rocket, ChevronDown, FolderOpen, Edit, AlertTriangle, Layers, GitBranch } from './Icons'

const STEPS = { UPLOAD: 'upload', STATES: 'states', RUN: 'run' }

export default function ContextInput({ onRunComplete }) {
  const [step, setStep] = useState(STEPS.UPLOAD)
  const [contextDoc, setContextDoc] = useState('')
  const [fileName, setFileName] = useState(null)
  const [pastPrompts, setPastPrompts] = useState('')
  const [showPast, setShowPast] = useState(false)
  const [stateNames, setStateNames] = useState([])
  const [stateDetails, setStateDetails] = useState([])
  const [flowSummary, setFlowSummary] = useState('')
  const [running, setRunning] = useState(false)
  const [extracting, setExtracting] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState('')
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const handleFileSelect = async (file) => {
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      const result = await uploadDocument(file)
      setContextDoc(result.text)
      setFileName(result.filename)
      setUploading(false)
      await handleExtractStates(result.text)
    } catch (err) {
      setUploading(false)
      setError(err.response?.data?.detail || 'Failed to upload file')
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileSelect(file)
  }

  const handleExtractStates = async (text = null) => {
    const doc = text || contextDoc
    if (!doc || doc.trim().length < 20) {
      setError('Context document is too short for state extraction')
      return
    }
    setExtracting(true)
    setError(null)
    try {
      const result = await extractStates(doc)
      const states = result.states || []
      setStateNames(states.map(s => s.state_name))
      setStateDetails(states)
      setFlowSummary(result.flow_summary || '')
      setStep(STEPS.STATES)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to extract states')
    }
    setExtracting(false)
  }

  const canRun = contextDoc.trim().length > 20 && stateNames.length > 0 && !running

  const handleRun = async () => {
    setRunning(true)
    setError(null)
    setProgress('Starting pipeline...')
    setStep(STEPS.RUN)
    try {
      const { run_id } = await startRun(
        contextDoc,
        stateNames.filter(s => s.trim()),
        pastPrompts.trim() || null
      )
      streamRunProgress(run_id, (data) => {
        console.log("[SSE Data Received]:", data)
        setProgress(data.progress || data.status)

        if (data.status === 'complete' && data.result) {
          setRunning(false)
          onRunComplete(run_id, data.result)
        } else if (data.status === 'complete') {
          // It completed but sent no result
          setRunning(false)
          setError(`Fatal Error: Pipeline completed, but the result payload was missing.`)
        } else if (data.status === 'error') {
          setRunning(false)
          setError(data.error || 'Pipeline failed')
        }
      })
    } catch (err) {
      setRunning(false)
      setError(err.response?.data?.detail || err.message)
    }
  }

  return (
    <div className="space-y-5">
      {/* Sub-step indicator */}
      <div className="flex items-center gap-1.5">
        {[
          { key: STEPS.UPLOAD, label: 'Upload', Icon: Upload },
          { key: STEPS.STATES, label: 'States', Icon: Layers },
          { key: STEPS.RUN, label: 'Generate', Icon: Rocket },
        ].map((s, i) => (
          <div key={s.key} className="flex items-center gap-1.5">
            {i > 0 && <div className="w-5 h-px bg-[var(--color-surface-3)]" />}
            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
              step === s.key
                ? 'bg-[var(--color-accent-muted)] text-[var(--color-accent-light)]'
                : 'text-[var(--color-text-3)]'
            }`}>
              <s.Icon size={12} />
              <span>{s.label}</span>
            </div>
          </div>
        ))}
      </div>

      {/* ──── Upload / Paste ──── */}
      <div className="card p-5">
        <div className="flex items-center gap-2 mb-1.5">
          <FileText size={16} className="text-[var(--color-accent-light)]" />
          <h2 className="text-sm font-semibold text-[var(--color-text)]">Context Document</h2>
        </div>
        <p className="text-xs text-[var(--color-text-3)] mb-4 ml-6">
          Upload .docx, .pdf, .txt or paste text. Google Docs → export as .docx first.
        </p>

        {/* Drop zone */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onClick={() => fileInputRef.current?.click()}
          className={`relative border border-dashed rounded-xl p-10 text-center cursor-pointer transition-all ${
            dragOver
              ? 'border-[var(--color-accent)] bg-[var(--color-accent-muted)]'
              : fileName
                ? 'border-[var(--color-success)]/30 bg-[var(--color-success-muted)]'
                : 'border-[var(--color-surface-3)] hover:border-[var(--color-border-hover)] hover:bg-[var(--color-surface)]'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".docx,.pdf,.txt,.md,.doc"
            onChange={(e) => handleFileSelect(e.target.files[0])}
            className="hidden"
          />
          {uploading ? (
            <div className="flex flex-col items-center gap-2">
              <div className="spinner" />
              <span className="text-xs text-[var(--color-text-2)]">Extracting text...</span>
            </div>
          ) : fileName ? (
            <div className="flex flex-col items-center gap-1.5">
              <FileText size={24} className="text-[var(--color-success)]" />
              <p className="text-sm font-medium text-[var(--color-text)]">{fileName}</p>
              <p className="text-xs text-[var(--color-text-3)]">{contextDoc.length.toLocaleString()} characters</p>
              <p className="text-xs text-[var(--color-accent-light)] mt-1">Click to replace</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <FolderOpen size={28} className="text-[var(--color-text-3)]" />
              <p className="text-sm text-[var(--color-text-2)]">Drop file here or click to browse</p>
              <p className="text-[10px] text-[var(--color-text-3)]">.docx · .pdf · .txt · .md</p>
            </div>
          )}
        </div>

        {/* Paste toggle */}
        <details className="mt-3 group">
          <summary className="flex items-center gap-1.5 text-xs text-[var(--color-text-3)] cursor-pointer hover:text-[var(--color-text-2)] transition-colors">
            <Edit size={12} /> Paste text manually
          </summary>
          <textarea
            value={contextDoc}
            onChange={(e) => { setContextDoc(e.target.value); setFileName(null) }}
            rows={5}
            placeholder="Paste your context document..."
            className="w-full mt-2"
          />
        </details>

        {/* Extract button */}
        {contextDoc && step === STEPS.UPLOAD && !extracting && (
          <button onClick={() => handleExtractStates()} className="btn btn-primary w-full mt-4 py-2.5">
            <Brain size={15} /> Analyse & Extract States
          </button>
        )}

        {extracting && (
          <div className="flex items-center justify-center gap-2 mt-4 py-2.5">
            <div className="spinner" style={{ borderTopColor: 'var(--color-accent-light)' }} />
            <span className="text-xs text-[var(--color-accent-light)]">Analysing document...</span>
          </div>
        )}
      </div>

      {/* ──── States Review ──── */}
      {step === STEPS.STATES && (
        <>
          {flowSummary && (
            <div className="card-highlight p-4 flex items-start gap-3">
              <Brain size={16} className="text-[var(--color-accent-light)] mt-0.5 flex-shrink-0" />
              <div>
                <h3 className="text-xs font-semibold text-[var(--color-accent-light)] mb-1">Flow Summary</h3>
                <p className="text-xs text-[var(--color-text-2)] leading-relaxed">{flowSummary}</p>
              </div>
            </div>
          )}

          <div className="card p-5">
            <div className="flex items-center gap-2 mb-1">
              <GitBranch size={16} className="text-[var(--color-accent-light)]" />
              <h2 className="text-sm font-semibold text-[var(--color-text)]">
                States <span className="text-[var(--color-text-3)] font-normal">({stateDetails.length})</span>
              </h2>
            </div>
            <p className="text-xs text-[var(--color-text-3)] mb-4 ml-6">Review, edit, or remove states. Add customs ones below.</p>

            <div className="space-y-1.5 mb-4">
              {stateDetails.map((s, i) => (
                <div key={s.state_name} className="flex items-start gap-2.5 px-3 py-2.5 rounded-lg bg-[var(--color-bg)] border border-[var(--color-border)] group hover:border-[var(--color-border-hover)] transition-colors">
                  <span className="text-[10px] text-[var(--color-text-3)] font-mono mt-1 w-4 text-right">{i + 1}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-semibold text-[var(--color-text)]">{s.state_name}</div>
                    <div className="text-[11px] text-[var(--color-text-3)] mt-0.5 leading-relaxed">{s.description}</div>
                  </div>
                  <button
                    onClick={() => {
                      setStateNames(prev => prev.filter((_, idx) => idx !== i))
                      setStateDetails(prev => prev.filter((_, idx) => idx !== i))
                    }}
                    className="opacity-0 group-hover:opacity-100 text-[var(--color-text-3)] hover:text-[var(--color-danger)] transition-all p-1 rounded"
                  >
                    <span className="text-xs">✕</span>
                  </button>
                </div>
              ))}
            </div>

            <StateList stateNames={stateNames} onChange={setStateNames} />
          </div>

          {/* Past prompts */}
          <div className="card overflow-hidden">
            <button
              onClick={() => setShowPast(!showPast)}
              className="w-full px-5 py-3.5 flex items-center justify-between text-left hover:bg-[var(--color-surface-2)]/50 transition-colors"
            >
              <div className="flex items-center gap-2">
                <Sparkles size={15} className="text-[var(--color-text-3)]" />
                <div>
                  <span className="text-xs font-semibold text-[var(--color-text)]">Past Prompts</span>
                  <span className="text-xs text-[var(--color-text-3)] ml-1.5">Optional</span>
                </div>
              </div>
              <ChevronDown size={14} className={`text-[var(--color-text-3)] transition-transform duration-200 ${showPast ? 'rotate-180' : ''}`} />
            </button>
            {showPast && (
              <div className="px-5 pb-4">
                <textarea
                  value={pastPrompts}
                  onChange={(e) => setPastPrompts(e.target.value)}
                  rows={5}
                  placeholder="Paste past prompts separated by --- or double newlines..."
                  className="w-full"
                />
              </div>
            )}
          </div>

          {/* Run */}
          <button onClick={handleRun} disabled={!canRun} className="btn btn-primary w-full py-3">
            <Rocket size={15} /> Run Pipeline · {stateNames.length} states
          </button>
        </>
      )}

      {/* ──── Running ──── */}
      {step === STEPS.RUN && (
        <div className="card p-6">
          {error ? (
            <div className="flex items-start gap-2.5 p-3 rounded-lg bg-[var(--color-danger-muted)] border border-[var(--color-danger)]/20">
              <AlertTriangle size={15} className="text-[var(--color-danger)] mt-0.5" />
              <div>
                <p className="text-xs text-[var(--color-danger)]">{error}</p>
                <button onClick={() => { setStep(STEPS.STATES); setError(null) }} className="text-xs text-[var(--color-accent-light)] mt-1.5 hover:underline">
                  ← Back to states
                </button>
              </div>
            </div>
          ) : running && (
            <div className="text-center py-4">
              <div className="flex items-center justify-center gap-2 mb-3">
                <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)] pulse-soft" />
                <span className="text-sm text-[var(--color-text-2)] font-medium">{progress}</span>
              </div>
              <div className="progress-track max-w-xs mx-auto">
                <div className="progress-fill" style={{ width: '60%' }} />
              </div>
              <p className="text-[11px] text-[var(--color-text-3)] mt-3">
                Processing {stateNames.length} states through 4 agents
              </p>
            </div>
          )}
        </div>
      )}

      {/* Upload error */}
      {error && step === STEPS.UPLOAD && (
        <div className="card p-3">
          <div className="flex items-center gap-2 text-xs text-[var(--color-danger)]">
            <AlertTriangle size={14} /> {error}
          </div>
        </div>
      )}
    </div>
  )
}
