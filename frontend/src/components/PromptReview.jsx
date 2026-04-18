import { useState } from 'react'
import { approveDraft, discardDraft } from '../api'
import { Check, Edit, X, AlertTriangle, Eye, ChevronDown, RotateCw, Copy } from './Icons'
import RenderPrompt from './RenderPrompt'

export default function PromptReview({ runId, result, onComplete }) {
  const [drafts, setDrafts] = useState(result.drafts || [])
  const [editingState, setEditingState] = useState(null)
  const [editText, setEditText] = useState('')
  const [discardingState, setDiscardingState] = useState(null)
  const [discardReason, setDiscardReason] = useState('')
  const [regenToggle, setRegenToggle] = useState(true)
  const [loading, setLoading] = useState(null)
  const [expandedExamples, setExpandedExamples] = useState({})
  const [copiedState, setCopiedState] = useState(null)

  const copyToClipboard = async (text, name) => {
    await navigator.clipboard.writeText(text)
    setCopiedState(name)
    setTimeout(() => setCopiedState(null), 2000)
  }

  const reviewed = drafts.filter(d => d.status !== 'pending').length
  const total = drafts.length
  const allReviewed = reviewed === total

  const handleApprove = async (stateName) => {
    setLoading(stateName)
    try {
      await approveDraft(runId, stateName)
      setDrafts(prev => prev.map(d => d.state_name === stateName ? { ...d, status: 'approved' } : d))
    } catch (err) { console.error(err) }
    setLoading(null)
  }

  const handleEditStart = (draft) => { setEditingState(draft.state_name); setEditText(draft.prompt) }

  const handleEditSave = async () => {
    setLoading(editingState)
    try {
      await approveDraft(runId, editingState, editText)
      setDrafts(prev => prev.map(d => d.state_name === editingState ? { ...d, status: 'edited', prompt: editText, edit_content: editText } : d))
    } catch (err) { console.error(err) }
    setEditingState(null); setEditText(''); setLoading(null)
  }

  const handleDiscard = async () => {
    const stateName = discardingState
    setLoading(stateName)
    try {
      const res = await discardDraft(runId, stateName, discardReason, regenToggle)
      if (res.new_draft) {
        setDrafts(prev => prev.map(d => d.state_name === stateName ? { ...res.new_draft, status: 'pending' } : d))
      } else {
        setDrafts(prev => prev.map(d => d.state_name === stateName ? { ...d, status: 'discarded' } : d))
      }
    } catch (err) { console.error(err) }
    setDiscardingState(null); setDiscardReason(''); setLoading(null)
  }

  const statusConfig = {
    pending:   { cls: 'bg-[var(--color-surface-2)] text-[var(--color-text-3)]', label: 'Pending' },
    approved:  { cls: 'bg-[var(--color-success-muted)] text-[var(--color-success)]', label: 'Approved' },
    edited:    { cls: 'bg-[var(--color-warning-muted)] text-[var(--color-warning)]', label: 'Edited' },
    discarded: { cls: 'bg-[var(--color-danger-muted)] text-[var(--color-danger)]', label: 'Discarded' },
  }

  return (
    <div className="space-y-4">
      {/* Consistency review */}
      {result.review_notes && (
        <div className="card-highlight p-4 flex items-start gap-3">
          <AlertTriangle size={15} className="text-[var(--color-warning)] mt-0.5 flex-shrink-0" />
          <div>
            <h3 className="text-xs font-semibold text-[var(--color-warning)] mb-0.5">Consistency Notes</h3>
            <p className="text-xs text-[var(--color-text-2)] leading-relaxed">{result.review_notes}</p>
          </div>
        </div>
      )}

      {/* Progress */}
      <div className="flex items-center justify-between text-xs px-1">
        <span className="text-[var(--color-text-3)]">{reviewed}/{total} reviewed</span>
        <span className="text-[var(--color-accent-light)] font-medium">{Math.round((reviewed / total) * 100)}%</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${(reviewed / total) * 100}%` }} />
      </div>

      {/* Draft cards */}
      {drafts.map((draft, idx) => {
        const spec = result.states?.find(s => s.state_name === draft.state_name)
        const isEditing = editingState === draft.state_name
        const isDiscarding = discardingState === draft.state_name
        const isLoading = loading === draft.state_name
        const { cls, label } = statusConfig[draft.status] || statusConfig.pending

        return (
          <div key={draft.state_name} className="card overflow-hidden">
            {/* Header */}
            <div className="px-5 py-3.5 flex items-center justify-between border-b border-[var(--color-border)]">
              <div className="flex items-center gap-2.5">
                <span className="w-5 h-5 rounded-md bg-[var(--color-accent-muted)] flex items-center justify-center text-[var(--color-accent-light)] text-[10px] font-bold">
                  {idx + 1}
                </span>
                <div>
                  <h3 className="text-xs font-semibold text-[var(--color-text)]">{draft.state_name}</h3>
                  {spec && <p className="text-[10px] text-[var(--color-text-3)] mt-0.5">{spec.intent}</p>}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className={`badge ${cls}`}>{label}</span>
                <button onClick={() => copyToClipboard(draft.edit_content || draft.prompt, draft.state_name)} className="btn btn-ghost text-[10px] py-1 px-2 ml-1">
                  <Copy size={12} /> {copiedState === draft.state_name ? 'Copied' : 'Copy'}
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="p-5">
              {isEditing ? (
                <div className="space-y-3">
                  <textarea value={editText} onChange={(e) => setEditText(e.target.value)} rows={8}
                    className="w-full font-mono text-xs" />
                  <div className="flex gap-2 justify-end">
                    <button onClick={() => setEditingState(null)} className="btn btn-ghost text-xs py-1.5">Cancel</button>
                    <button onClick={handleEditSave} className="btn btn-success text-xs py-1.5" disabled={isLoading}>
                      {isLoading ? <><div className="spinner" style={{width:12,height:12}}/> Saving</> : <><Check size={13}/> Approve Edit</>}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="p-4 bg-[var(--color-bg)] border border-[var(--color-border)] rounded-xl">
                  <RenderPrompt text={draft.edit_content || draft.prompt} />
                </div>
              )}

              {/* Discard dialog */}
              {isDiscarding && (
                <div className="mt-4 p-3.5 rounded-lg bg-[var(--color-danger-muted)] border border-[var(--color-danger)]/15">
                  <label className="block text-xs text-[var(--color-text-2)] mb-2">Feedback (optional)</label>
                  <input type="text" value={discardReason} onChange={(e) => setDiscardReason(e.target.value)}
                    placeholder="What should be different?" className="w-full mb-3 text-xs" />
                  <div className="flex items-center justify-between">
                    <label className="flex items-center gap-2 text-xs text-[var(--color-text-3)] cursor-pointer select-none">
                      <input type="checkbox" checked={regenToggle} onChange={(e) => setRegenToggle(e.target.checked)}
                        className="accent-[var(--color-accent)]" />
                      <RotateCw size={12} /> Regenerate
                    </label>
                    <div className="flex gap-2">
                      <button onClick={() => setDiscardingState(null)} className="btn btn-ghost text-xs py-1.5">Cancel</button>
                      <button onClick={handleDiscard} className="btn btn-danger text-xs py-1.5" disabled={isLoading}>Confirm</button>
                    </div>
                  </div>
                </div>
              )}

              {/* KB examples */}
              {draft.retrieved_examples?.length > 0 && (
                <button onClick={() => setExpandedExamples(p => ({ ...p, [draft.state_name]: !p[draft.state_name] }))}
                  className="flex items-center gap-1 text-[10px] text-[var(--color-text-3)] hover:text-[var(--color-text-2)] mt-3 transition-colors">
                  <Eye size={11} /> {draft.retrieved_examples.length} KB refs
                  <ChevronDown size={10} className={`transition-transform ${expandedExamples[draft.state_name] ? 'rotate-180' : ''}`} />
                </button>
              )}
              {expandedExamples[draft.state_name] && (
                <div className="mt-2 space-y-1.5">
                  {draft.retrieved_examples.map((ex, i) => (
                    <div key={i} className="p-2.5 rounded-md bg-[var(--color-bg)] border border-[var(--color-border)] text-[10px] font-mono text-[var(--color-text-3)] leading-relaxed">
                      {ex.substring(0, 200)}{ex.length > 200 ? '…' : ''}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Actions */}
            {draft.status === 'pending' && !isEditing && !isDiscarding && (
              <div className="px-5 py-3 border-t border-[var(--color-border)] flex gap-2 justify-end">
                <button onClick={() => handleApprove(draft.state_name)} className="btn btn-success text-xs py-1.5" disabled={isLoading}>
                  <Check size={13} /> Approve
                </button>
                <button onClick={() => handleEditStart(draft)} className="btn btn-info text-xs py-1.5">
                  <Edit size={13} /> Edit
                </button>
                <button onClick={() => setDiscardingState(draft.state_name)} className="btn btn-danger text-xs py-1.5">
                  <X size={13} /> Discard
                </button>
              </div>
            )}
          </div>
        )
      })}

      {/* Finish */}
      <button onClick={() => onComplete(drafts)} disabled={!allReviewed}
        className="btn btn-primary w-full py-3 mt-2">
        {allReviewed ? <><Check size={15} /> Finish Review</> : `${total - reviewed} remaining`}
      </button>
    </div>
  )
}
