import { useState, useEffect } from 'react'
import { listKB, deleteKBEntry } from '../api'
import { BookOpen, Search, Trash, ChevronDown } from './Icons'

export default function KBBrowser() {
  const [records, setRecords] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [domain, setDomain] = useState('')
  const [source, setSource] = useState('')
  const [expandedId, setExpandedId] = useState(null)
  const [confirmDelete, setConfirmDelete] = useState(null)
  const [loading, setLoading] = useState(false)
  
  // Add Form State
  const [showAddForm, setShowAddForm] = useState(false)
  const [submittingAdd, setSubmittingAdd] = useState(false)
  const [addPromptText, setAddPromptText] = useState('')
  
  const limit = 15

  const fetchRecords = async () => {
    setLoading(true)
    try {
      const data = await listKB(domain || null, source || null, page, limit)
      setRecords(data.records); setTotal(data.total)
    } catch (err) { console.error(err) }
    setLoading(false)
  }
  
  const handleAddSubmit = async (e) => {
    e.preventDefault()
    if (!addPromptText.trim()) return
    
    setSubmittingAdd(true)
    try {
      const payload = { prompt: addPromptText }
      await import('../api').then(m => m.addAutoKBEntry(payload))
      setAddPromptText('')
      setShowAddForm(false)
      fetchRecords()
    } catch (err) {
      console.error(err)
      alert("Failed to auto-tag prompt: " + (err.response?.data?.detail || err.message))
    }
    setSubmittingAdd(false)
  }

  useEffect(() => { fetchRecords() }, [page, domain, source])

  const handleDelete = async (id) => {
    try { await deleteKBEntry(id); setConfirmDelete(null); fetchRecords() }
    catch (err) { console.error(err) }
  }

  const totalPages = Math.ceil(total / limit)

  return (
    <div className="p-5">
      <div className="flex items-center gap-2 mb-4">
        <BookOpen size={16} className="text-[var(--color-accent-light)]" />
        <h2 className="text-sm font-semibold text-[var(--color-text)]">Knowledge Base</h2>
        <div className="ml-auto flex items-center gap-3">
          <span className="text-[10px] text-[var(--color-text-3)]">{total} records</span>
          <button 
            onClick={() => setShowAddForm(!showAddForm)}
            className="btn btn-primary text-xs py-1.5 px-3 rounded-lg"
          >
            {showAddForm ? 'Cancel' : '+ Add Prompt'}
          </button>
        </div>
      </div>

      {/* Add Form */}
      {showAddForm && (
        <form onSubmit={handleAddSubmit} className="mb-6 p-4 rounded-xl border border-[var(--color-accent)] bg-[var(--color-surface)] space-y-3">
          <div>
            <label className="block text-[10px] text-[var(--color-text-3)] mb-1">
              Paste prompt text below. Our AI will automatically classify the State and Use Case!
            </label>
            <textarea 
              required 
              value={addPromptText} 
              onChange={e => setAddPromptText(e.target.value)} 
              placeholder="Write or paste your custom bot prompt here..." 
              className="w-full text-xs p-3 bg-[var(--color-bg)] border border-[var(--color-border)] rounded-lg min-h-[120px] resize-y focus:outline-none focus:border-[var(--color-accent)] transition-colors" 
            />
          </div>
          <div className="flex items-center justify-between pt-2">
            <span className="text-[10px] text-[var(--color-text-3)] italic">
              {submittingAdd ? "🤖 AI is classifying State & Use Case..." : "Ready for automated AI classification."}
            </span>
            <button type="submit" disabled={submittingAdd || !addPromptText.trim()} className="btn btn-primary text-xs py-2 px-6 shadow-md shadow-indigo-500/20">
              {submittingAdd ? 'Classifying & Saving...' : 'Auto-Classify & Save'}
            </button>
          </div>
        </form>
      )}

      {/* Filters */}
      <div className="space-y-2 mb-4">
        <div className="relative">
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-3)]" />
          <input type="text" value={domain} onChange={(e) => { setDomain(e.target.value); setPage(1) }}
            placeholder="Filter by domain..." className="w-full text-xs pl-8" />
        </div>
        <select value={source} onChange={(e) => { setSource(e.target.value); setPage(1) }}
          className="w-full text-xs bg-[var(--color-bg)] border border-[var(--color-border)] rounded-lg py-2 px-3 text-[var(--color-text-2)]">
          <option value="">All sources</option>
          <option value="seed">Seed</option>
          <option value="generated">Generated</option>
          <option value="edited">Edited</option>
        </select>
      </div>

      {/* Records */}
      {loading ? (
        <div className="text-center text-[var(--color-text-3)] py-8 text-xs">Loading...</div>
      ) : records.length === 0 ? (
        <div className="text-center text-[var(--color-text-3)] py-10 text-xs leading-relaxed">
          No entries yet.<br />Approve prompts to populate.
        </div>
      ) : (
        <div className="space-y-1.5">
          {records.map(r => (
            <div key={r.id} className="rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] overflow-hidden hover:border-[var(--color-border-hover)] transition-colors">
              <button onClick={() => setExpandedId(expandedId === r.id ? null : r.id)} className="w-full px-3.5 py-2.5 text-left">
                <div className="flex items-center justify-between mb-0.5">
                  <span className="text-xs font-medium text-[var(--color-text)]">{r.state_name}</span>
                  <span className={`badge badge-${r.source}`}>{r.source}</span>
                </div>
                <div className="flex items-center gap-1.5 text-[10px] text-[var(--color-text-3)]">
                  <span>{r.context_domain}</span>
                  <span>·</span>
                  <span>{new Date(r.timestamp).toLocaleDateString()}</span>
                </div>
              </button>

              {expandedId === r.id && (
                <div className="px-3.5 pb-3 border-t border-[var(--color-border)] pt-2.5">
                  <pre className="text-[10px] font-mono text-[var(--color-text-3)] whitespace-pre-wrap max-h-40 overflow-y-auto mb-2 leading-relaxed">
                    {r.prompt}
                  </pre>
                  {confirmDelete === r.id ? (
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-[var(--color-danger)]">Delete?</span>
                      <button onClick={() => handleDelete(r.id)} className="btn btn-danger text-[10px] py-1 px-2">Yes</button>
                      <button onClick={() => setConfirmDelete(null)} className="btn btn-ghost text-[10px] py-1 px-2">No</button>
                    </div>
                  ) : (
                    <button onClick={() => setConfirmDelete(r.id)} className="flex items-center gap-1 text-[10px] text-[var(--color-text-3)] hover:text-[var(--color-danger)] transition-colors">
                      <Trash size={11} /> Delete
                    </button>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-4 text-xs text-[var(--color-text-3)]">
          <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} className="btn btn-ghost text-[10px] py-1">← Prev</button>
          <span>{page}/{totalPages}</span>
          <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages} className="btn btn-ghost text-[10px] py-1">Next →</button>
        </div>
      )}
    </div>
  )
}
