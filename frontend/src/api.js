import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

export async function startRun(contextDoc, stateNames, pastPrompts) {
  const { data } = await api.post('/run', {
    context_doc: contextDoc,
    state_names: stateNames,
    past_prompts: pastPrompts || null,
  })
  return data
}

export async function getRunStatus(runId) {
  const { data } = await api.get(`/run/${runId}/status`)
  return data
}

export function streamRunProgress(runId, onMessage) {
  const evtSource = new EventSource(`/api/run/${runId}/stream`)
  evtSource.onmessage = (event) => {
    const data = JSON.parse(event.data)
    onMessage(data)
    if (data.status === 'complete' || data.status === 'error') {
      evtSource.close()
    }
  }
  evtSource.onerror = () => evtSource.close()
  return evtSource
}

export async function approveDraft(runId, stateName, editedPrompt = null) {
  const { data } = await api.post('/approve', {
    run_id: runId,
    state_name: stateName,
    edited_prompt: editedPrompt,
  })
  return data
}

export async function discardDraft(runId, stateName, reason = '', regenerate = false) {
  const { data } = await api.post('/discard', {
    run_id: runId,
    state_name: stateName,
    reason,
    regenerate,
  })
  return data
}

export async function listKB(domain = null, source = null, page = 1, limit = 20) {
  const params = { page, limit }
  if (domain) params.domain = domain
  if (source) params.source = source
  const { data } = await api.get('/kb/list', { params })
  return data
}

export async function deleteKBEntry(recordId) {
  const { data } = await api.delete(`/kb/${recordId}`)
  return data
}

export async function addAutoKBEntry(payload) {
  const { data } = await api.post('/kb/add-auto', payload)
  return data
}

export async function addKBEntry(payload) {
  const { data } = await api.post('/kb/add', payload)
  return data
}

export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function extractStates(contextDoc) {
  const { data } = await api.post('/extract-states', {
    context_doc: contextDoc,
  })
  return data
}

export default api
