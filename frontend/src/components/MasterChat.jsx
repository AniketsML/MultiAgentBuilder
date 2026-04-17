import React, { useState, useRef, useEffect } from "react";

const API_BASE = "/api";

/* ── Minimal markdown renderer (bold, bullets, line breaks) ── */
function renderMarkdown(text) {
  if (!text) return null;
  const lines = text.split("\n");
  const elements = [];
  let key = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Bullet points
    if (/^\s*[-•]\s+/.test(line) || /^\s*\d+\.\s+/.test(line)) {
      const content = line.replace(/^\s*[-•]\s+/, "").replace(/^\s*\d+\.\s+/, "");
      elements.push(
        <div key={key++} style={{ display: "flex", gap: 6, marginLeft: 4, marginTop: 2 }}>
          <span style={{ color: "rgba(139,92,246,0.7)", flexShrink: 0 }}>•</span>
          <span>{applyInline(content)}</span>
        </div>
      );
    } else if (line.trim() === "") {
      elements.push(<div key={key++} style={{ height: 6 }} />);
    } else {
      elements.push(<div key={key++}>{applyInline(line)}</div>);
    }
  }
  return elements;
}

function applyInline(text) {
  // Bold **text** and backtick `code`
  const parts = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    // Check for **bold**
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
    // Check for `code`
    const codeMatch = remaining.match(/`([^`]+)`/);

    let firstMatch = null;
    let matchType = null;

    if (boldMatch && (!codeMatch || boldMatch.index <= codeMatch.index)) {
      firstMatch = boldMatch;
      matchType = "bold";
    } else if (codeMatch) {
      firstMatch = codeMatch;
      matchType = "code";
    }

    if (!firstMatch) {
      parts.push(<span key={key++}>{remaining}</span>);
      break;
    }

    // Text before match
    if (firstMatch.index > 0) {
      parts.push(<span key={key++}>{remaining.slice(0, firstMatch.index)}</span>);
    }

    if (matchType === "bold") {
      parts.push(
        <strong key={key++} style={{ color: "rgba(255,255,255,0.95)", fontWeight: 600 }}>
          {firstMatch[1]}
        </strong>
      );
    } else {
      parts.push(
        <code
          key={key++}
          style={{
            background: "rgba(139,92,246,0.15)",
            padding: "1px 5px",
            borderRadius: 4,
            fontSize: "0.9em",
            color: "rgba(167,139,250,0.9)",
          }}
        >
          {firstMatch[1]}
        </code>
      );
    }

    remaining = remaining.slice(firstMatch.index + firstMatch[0].length);
  }

  return parts;
}


export function MasterChat() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputMsg, setInputMsg] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [pulseColor, setPulseColor] = useState("#6366f1");
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isExpanded]);

  const send = async (e) => {
    e?.preventDefault();
    if (!inputMsg.trim()) return;

    const msg = inputMsg.trim();
    setInputMsg("");
    setMessages((p) => [...p, { role: "user", text: msg }]);
    setIsThinking(true);
    setPulseColor("#f59e0b");

    try {
      const res = await fetch(`${API_BASE}/master/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: msg,
          chat_history: messages.map((m) => ({
            role: m.role === "user" ? "user" : "assistant",
            content: m.text,
          })),
        }),
      });
      const data = await res.json();
      setMessages((p) => [...p, { role: "ai", text: data.response }]);
      setPulseColor("#22c55e");
      setTimeout(() => setPulseColor("#6366f1"), 3000);
    } catch (err) {
      setMessages((p) => [...p, { role: "ai", text: `Error: ${err.message}` }]);
      setPulseColor("#ef4444");
      setTimeout(() => setPulseColor("#6366f1"), 3000);
    } finally {
      setIsThinking(false);
    }
  };

  // --- COLLAPSED ORB ---
  if (!isExpanded) {
    return (
      <button
        onClick={() => setIsExpanded(true)}
        style={{
          position: "fixed",
          bottom: 20,
          right: 20,
          width: 44,
          height: 44,
          borderRadius: "50%",
          background: `radial-gradient(circle, ${pulseColor}44 0%, ${pulseColor}22 70%, transparent 100%)`,
          border: `1.5px solid ${pulseColor}66`,
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 9999,
          transition: "all 0.3s ease",
          boxShadow: `0 0 20px ${pulseColor}33`,
          animation: isThinking ? "masterPulse 1.5s infinite" : "none",
        }}
        title="System Brain"
      >
        <div
          style={{
            width: 14,
            height: 14,
            borderRadius: "50%",
            background: pulseColor,
            boxShadow: `0 0 8px ${pulseColor}`,
            transition: "all 0.3s ease",
          }}
        />
      </button>
    );
  }

  // --- EXPANDED CHAT ---
  return (
    <div
      style={{
        position: "fixed",
        bottom: 20,
        right: 20,
        width: 360,
        height: 440,
        borderRadius: 16,
        background: "rgba(10, 10, 20, 0.88)",
        backdropFilter: "blur(24px)",
        WebkitBackdropFilter: "blur(24px)",
        border: "1px solid rgba(139,92,246,0.12)",
        display: "flex",
        flexDirection: "column",
        zIndex: 9999,
        overflow: "hidden",
        boxShadow: "0 8px 40px rgba(0,0,0,0.5), 0 0 1px rgba(139,92,246,0.3)",
        fontFamily: "'Inter', -apple-system, sans-serif",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 14px",
          borderBottom: "1px solid rgba(139,92,246,0.08)",
          background: "rgba(139,92,246,0.04)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: pulseColor,
              boxShadow: `0 0 6px ${pulseColor}`,
              transition: "all 0.3s",
            }}
          />
          <span
            style={{
              fontSize: 10,
              fontWeight: 600,
              color: "rgba(255,255,255,0.5)",
              letterSpacing: "0.8px",
              textTransform: "uppercase",
            }}
          >
            System Brain
          </span>
        </div>
        <button
          onClick={() => setIsExpanded(false)}
          style={{
            background: "none",
            border: "none",
            color: "rgba(255,255,255,0.25)",
            cursor: "pointer",
            fontSize: 15,
            lineHeight: 1,
            padding: 2,
            transition: "color 0.2s",
          }}
          onMouseEnter={(e) => (e.target.style.color = "rgba(255,255,255,0.6)")}
          onMouseLeave={(e) => (e.target.style.color = "rgba(255,255,255,0.25)")}
        >
          ×
        </button>
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "10px 12px",
          display: "flex",
          flexDirection: "column",
          gap: 10,
        }}
      >
        {messages.length === 0 && (
          <div
            style={{
              color: "rgba(255,255,255,0.2)",
              fontSize: 11,
              textAlign: "center",
              marginTop: 50,
              lineHeight: 1.7,
            }}
          >
            <div style={{ fontSize: 18, marginBottom: 8, opacity: 0.4 }}>⬡</div>
            Observing pipeline...
            <br />
            <span style={{ fontSize: 10, opacity: 0.6 }}>
              Ask about runs, KB entries, or agents
            </span>
          </div>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              alignSelf: m.role === "user" ? "flex-end" : "flex-start",
              maxWidth: "88%",
            }}
          >
            {/* Role label */}
            <div
              style={{
                fontSize: 9,
                fontWeight: 600,
                color: m.role === "user" ? "rgba(99,102,241,0.5)" : "rgba(139,92,246,0.5)",
                letterSpacing: "0.5px",
                textTransform: "uppercase",
                marginBottom: 3,
                textAlign: m.role === "user" ? "right" : "left",
                paddingLeft: m.role === "user" ? 0 : 4,
                paddingRight: m.role === "user" ? 4 : 0,
              }}
            >
              {m.role === "user" ? "You" : "Brain"}
            </div>
            {/* Message bubble */}
            <div
              style={{
                padding: "8px 12px",
                borderRadius:
                  m.role === "user"
                    ? "12px 12px 4px 12px"
                    : "12px 12px 12px 4px",
                background:
                  m.role === "user"
                    ? "rgba(99, 102, 241, 0.15)"
                    : "rgba(255,255,255,0.04)",
                color: "rgba(255,255,255,0.82)",
                fontSize: 12,
                lineHeight: 1.6,
                border:
                  m.role === "user"
                    ? "1px solid rgba(99, 102, 241, 0.2)"
                    : "1px solid rgba(255,255,255,0.04)",
              }}
            >
              {m.role === "user" ? m.text : renderMarkdown(m.text)}
            </div>
          </div>
        ))}
        {isThinking && (
          <div
            style={{
              display: "flex",
              gap: 5,
              padding: "8px 12px",
              alignSelf: "flex-start",
            }}
          >
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                style={{
                  width: 5,
                  height: 5,
                  borderRadius: "50%",
                  background: "rgba(139,92,246,0.6)",
                  animation: `masterPulse 1s ${i * 0.15}s infinite`,
                }}
              />
            ))}
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Input */}
      <form
        onSubmit={send}
        style={{
          padding: "8px 12px",
          borderTop: "1px solid rgba(139,92,246,0.08)",
          display: "flex",
          gap: 8,
          background: "rgba(139,92,246,0.02)",
        }}
      >
        <input
          value={inputMsg}
          onChange={(e) => setInputMsg(e.target.value)}
          placeholder="Ask about the system..."
          disabled={isThinking}
          style={{
            flex: 1,
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(139,92,246,0.1)",
            borderRadius: 8,
            padding: "8px 12px",
            color: "rgba(255,255,255,0.85)",
            fontSize: 12,
            outline: "none",
            transition: "border-color 0.2s",
          }}
          onFocus={(e) => (e.target.style.borderColor = "rgba(139,92,246,0.3)")}
          onBlur={(e) => (e.target.style.borderColor = "rgba(139,92,246,0.1)")}
        />
        <button
          type="submit"
          disabled={isThinking || !inputMsg.trim()}
          style={{
            background: "rgba(139, 92, 246, 0.2)",
            border: "1px solid rgba(139, 92, 246, 0.2)",
            borderRadius: 8,
            padding: "0 12px",
            color: "rgba(255,255,255,0.7)",
            cursor: "pointer",
            fontSize: 13,
            opacity: isThinking || !inputMsg.trim() ? 0.2 : 1,
            transition: "all 0.2s",
          }}
        >
          ↑
        </button>
      </form>

      <style>{`
        @keyframes masterPulse {
          0%, 100% { opacity: 0.3; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.15); }
        }
      `}</style>
    </div>
  );
}
