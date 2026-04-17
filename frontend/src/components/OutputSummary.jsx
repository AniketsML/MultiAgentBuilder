import { useState } from 'react'
import { CheckCircle, Copy, Download, RotateCw } from './Icons'

export default function OutputSummary({ result, onNewRun }) {
  const [copiedState, setCopiedState] = useState(null)
  const drafts = result.drafts || []
  const approved = drafts.filter(d => d.status === 'approved' || d.status === 'edited')
  const discarded = drafts.filter(d => d.status === 'discarded')

  const getPromptText = (d) => d.edit_content || d.prompt

  const copyToClipboard = async (text, name) => {
    await navigator.clipboard.writeText(text)
    setCopiedState(name)
    setTimeout(() => setCopiedState(null), 2000)
  }

  const downloadJSON = () => {
    const out = approved.map(d => ({ state_name: d.state_name, prompt: getPromptText(d), status: d.status }))
    const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' })
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
    a.download = `flow_prompts_${result.run_id?.slice(0, 8) || 'export'}.json`; a.click()
  }

  const downloadTXT = () => {
    approved.forEach(d => {
      const blob = new Blob([getPromptText(d)], { type: 'text/plain' })
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
      a.download = `${d.state_name}.txt`; a.click()
    })
  }

  return (
    <div className="space-y-5">
      {/* Stats */}
      <div className="card p-6">
        <div className="flex items-center gap-2 mb-5">
          <CheckCircle size={18} className="text-[var(--color-success)]" />
          <h2 className="text-base font-semibold text-[var(--color-text)]">Pipeline Complete</h2>
        </div>
        <div className="grid grid-cols-3 gap-3">
          {[
            { n: approved.length, label: 'Approved', color: 'success' },
            { n: discarded.length, label: 'Discarded', color: 'danger' },
            { n: drafts.length, label: 'Total', color: 'accent' },
          ].map(s => (
            <div key={s.label} className={`text-center py-3 rounded-xl bg-[var(--color-${s.color}-muted)] border border-[var(--color-${s.color})]/15`}>
              <div className={`text-2xl font-bold text-[var(--color-${s.color})]`}>{s.n}</div>
              <div className="text-[10px] text-[var(--color-text-3)] mt-0.5 font-medium uppercase tracking-wider">{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Export */}
      <div className="card p-5">
        <div className="flex items-center gap-2 mb-3">
          <Download size={15} className="text-[var(--color-text-3)]" />
          <h3 className="text-xs font-semibold text-[var(--color-text)]">Export</h3>
        </div>
        <div className="flex gap-2">
          <button onClick={downloadJSON} className="btn btn-primary text-xs flex-1 py-2">JSON</button>
          <button onClick={downloadTXT} className="btn btn-ghost text-xs flex-1 py-2">.txt per state</button>
        </div>
      </div>

      {/* Prompts */}
      {approved.length > 0 && (
        <div className="space-y-3">
          {approved.map(d => (
            <div key={d.state_name} className="card overflow-hidden">
              <div className="px-5 py-3 flex items-center justify-between border-b border-[var(--color-border)]">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-[var(--color-text)]">{d.state_name}</span>
                  <span className={`badge ${d.status === 'edited' ? 'badge-edited' : 'badge-generated'}`}>{d.status}</span>
                </div>
                <button onClick={() => copyToClipboard(getPromptText(d), d.state_name)} className="btn btn-ghost text-[10px] py-1 px-2">
                  <Copy size={12} /> {copiedState === d.state_name ? 'Copied' : 'Copy'}
                </button>
              </div>
              <div className="p-5">
                <div className="mono-box">{getPromptText(d)}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      <button onClick={onNewRun} className="btn btn-primary w-full py-3">
        <RotateCw size={15} /> New Run
      </button>
    </div>
  )
}
