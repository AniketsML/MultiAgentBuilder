import { useState } from 'react'
import { Plus, ChevronUp, ChevronDown, X } from './Icons'

export default function StateList({ stateNames, onChange }) {
  const [newState, setNewState] = useState('')

  const addState = () => {
    const name = newState.trim().toLowerCase().replace(/\s+/g, '_')
    if (name && !stateNames.includes(name)) {
      onChange([...stateNames, name])
      setNewState('')
    }
  }

  const removeState = (index) => onChange(stateNames.filter((_, i) => i !== index))

  const moveState = (index, direction) => {
    const arr = [...stateNames]
    const target = index + direction
    if (target < 0 || target >= arr.length) return
    ;[arr[index], arr[target]] = [arr[target], arr[index]]
    onChange(arr)
  }

  return (
    <div>
      <div className="flex gap-2">
        <input
          type="text"
          value={newState}
          onChange={(e) => setNewState(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addState())}
          placeholder="Add custom state..."
          className="flex-1 text-xs"
        />
        <button onClick={addState} disabled={!newState.trim()} className="btn btn-ghost text-xs py-1.5 px-3">
          <Plus size={13} /> Add
        </button>
      </div>
    </div>
  )
}
