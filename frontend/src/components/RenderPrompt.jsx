export default function RenderPrompt({ text }) {
  if (!text) return null;

  // Split on:
  // 1. Markdown headers (e.g. ## ROUTING)
  // 2. Case separators (e.g. --- CASE: ... ---)
  // 3. Variables {{var_name}}
  // 4. Bold text **text**
  const regex = /(## [^\n]+|\n?--- [^\n]+ ---|\{\{[^}]+\}\}|\*\*[^*\n]+\*\*)/g;
  const segments = text.split(regex);

  return (
    <div className="font-mono text-[12.5px] leading-relaxed text-[var(--color-text-2)] whitespace-pre-wrap break-words">
      {segments.map((seg, i) => {
        if (!seg) return null;

        if (seg.startsWith('{{') && seg.endsWith('}}')) {
          return (
            <span key={i} className="mx-0.5 px-1.5 py-0.5 rounded bg-[var(--color-accent-muted)] text-[var(--color-accent-light)] font-bold text-[11px] tracking-wider border border-[var(--color-accent)]/20">
              {seg}
            </span>
          );
        }
        
        if (seg.trim().startsWith('---') && seg.trim().endsWith('---')) {
          return (
            <span key={i} className="block mt-6 mb-3 text-[var(--color-info)] font-bold border-b border-[var(--color-info-muted)] pb-1">
              {seg.trim()}
            </span>
          );
        }

        if (seg.startsWith('## ')) {
          return (
            <span key={i} className="block mt-6 mb-3 text-[var(--color-text)] font-extrabold text-[14px]">
              {seg}
            </span>
          );
        }

        if (seg.startsWith('**') && seg.endsWith('**')) {
          return (
            <span key={i} className="text-[var(--color-text)] font-semibold">
              {seg.replace(/\*\*/g, '')}
            </span>
          );
        }

        return <span key={i}>{seg}</span>;
      })}
    </div>
  );
}
