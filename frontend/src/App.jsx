import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Upload, FileText, Send, Trash2, BarChart2,
  Loader2, BookOpen, X
} from "lucide-react";

const API = import.meta.env.VITE_API_URL || "/api";

// ─── Styles ──────────────────────────────────────────────────────────────────
const S = {
  app: {
    display: "flex", height: "100vh", overflow: "hidden",
    background: "#0f1117", color: "#e2e8f0",
  },
  sidebar: {
    width: 300, minWidth: 260, maxWidth: 340,
    background: "#1a1d27", borderRight: "1px solid #2d3148",
    display: "flex", flexDirection: "column", overflow: "hidden",
  },
  sidebarHeader: {
    padding: "18px 16px 12px", borderBottom: "1px solid #2d3148",
  },
  logo: {
    display: "flex", alignItems: "center", gap: 8,
    fontSize: 18, fontWeight: 700, color: "#7c6cfa",
  },
  section: { padding: "12px 16px" },
  label: { fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", color: "#64748b", textTransform: "uppercase", marginBottom: 8 },
  uploadZone: {
    border: "2px dashed #3d4163", borderRadius: 10, padding: "20px 12px",
    textAlign: "center", cursor: "pointer", transition: "all .2s",
    background: "#12151f",
  },
  uploadZoneHover: { borderColor: "#7c6cfa", background: "#1a1730" },
  uploadBtn: {
    marginTop: 10, width: "100%", padding: "8px 0",
    background: "#7c6cfa", color: "#fff", border: "none",
    borderRadius: 8, cursor: "pointer", fontWeight: 600, fontSize: 13,
    display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
  },
  docList: { flex: 1, overflowY: "auto", padding: "0 16px 12px" },
  docItem: {
    display: "flex", alignItems: "center", gap: 8,
    padding: "8px 10px", borderRadius: 8, marginBottom: 6,
    background: "#12151f", border: "1px solid #2d3148",
    fontSize: 13,
  },
  docName: { flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  iconBtn: { background: "none", border: "none", cursor: "pointer", color: "#64748b", padding: 2, display: "flex" },
  evalBtn: {
    margin: "8px 16px 16px", padding: "14px 12px",
    background: "linear-gradient(135deg, #7c3aed, #4f46e5)",
    color: "#ffffff", border: "2px solid #a78bfa",
    borderRadius: 10, cursor: "pointer", fontWeight: 700, fontSize: 14,
    display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 6,
    lineHeight: 1.4, boxShadow: "0 0 12px rgba(124,58,237,0.5)",
  },

  // Main chat area
  main: { flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" },
  chatHeader: {
    padding: "16px 24px", borderBottom: "1px solid #2d3148",
    background: "#1a1d27", fontSize: 15, fontWeight: 600, color: "#94a3b8",
  },
  messages: { flex: 1, overflowY: "auto", padding: "24px", display: "flex", flexDirection: "column", gap: 20 },
  bubble: (role) => ({
    maxWidth: "75%", padding: "14px 18px", borderRadius: 14,
    lineHeight: 1.6, fontSize: 14,
    alignSelf: role === "user" ? "flex-end" : "flex-start",
    background: role === "user" ? "#7c6cfa" : "#1e2130",
    color: role === "user" ? "#fff" : "#e2e8f0",
    border: role === "assistant" ? "1px solid #2d3148" : "none",
    whiteSpace: "pre-wrap", wordBreak: "break-word",
  }),
  sourcesBox: {
    marginTop: 10, padding: "10px 14px",
    background: "#12151f", border: "1px solid #2d3148",
    borderRadius: 10, fontSize: 12,
  },
  sourceItem: {
    display: "flex", alignItems: "flex-start", gap: 8,
    padding: "6px 0", borderBottom: "1px solid #1e2130",
  },
  sourceTag: {
    background: "#2d3148", color: "#7c6cfa",
    padding: "1px 7px", borderRadius: 99, fontSize: 11, fontWeight: 700,
    whiteSpace: "nowrap",
  },
  inputBar: {
    padding: "16px 24px", borderTop: "1px solid #2d3148",
    background: "#1a1d27", display: "flex", gap: 10,
  },
  input: {
    flex: 1, background: "#12151f", border: "1px solid #2d3148",
    borderRadius: 10, color: "#e2e8f0", padding: "12px 16px",
    fontSize: 14, outline: "none", resize: "none", fontFamily: "inherit",
  },
  sendBtn: {
    padding: "12px 18px", background: "#7c6cfa", color: "#fff",
    border: "none", borderRadius: 10, cursor: "pointer",
    display: "flex", alignItems: "center", gap: 6, fontWeight: 600,
    fontSize: 14,
  },

  // Eval panel
  evalPanel: {
    width: 480, background: "#1a1d27", borderLeft: "1px solid #2d3148",
    display: "flex", flexDirection: "column", overflow: "hidden",
  },
  evalHeader: {
    padding: "16px", borderBottom: "1px solid #2d3148",
    fontWeight: 700, fontSize: 14, color: "#60a5fa",
    display: "flex", alignItems: "center", gap: 8,
  },
  evalBody: { flex: 1, overflowY: "auto", padding: 16 },
  scoreCard: {
    background: "#12151f", border: "1px solid #2d3148",
    borderRadius: 10, padding: "12px 14px", marginBottom: 12,
  },
  scoreQ: { fontSize: 12, color: "#94a3b8", marginBottom: 8, lineHeight: 1.4 },
  scoreRow: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 },
  scorePill: (val) => ({
    padding: "2px 10px", borderRadius: 99, fontSize: 12, fontWeight: 700,
    background: val >= 0.75 ? "#14532d" : val >= 0.5 ? "#713f12" : "#3b1515",
    color: val >= 0.75 ? "#4ade80" : val >= 0.5 ? "#fbbf24" : "#f87171",
  }),
  overallBox: {
    background: "#0f1835", border: "1px solid #2563eb",
    borderRadius: 10, padding: "12px 14px", marginBottom: 16,
  },
};

// ─── Components ──────────────────────────────────────────────────────────────

function SourcesAccordion({ sources }) {
  const [open, setOpen] = useState(false);
  if (!sources || sources.length === 0) return null;
  return (
    <div style={S.sourcesBox}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{ background: "none", border: "none", cursor: "pointer", color: "#7c6cfa", fontWeight: 600, fontSize: 12, display: "flex", alignItems: "center", gap: 4 }}
      >
        <span style={{ fontSize: 10 }}>{open ? "▲" : "▼"}</span>
        {sources.length} source{sources.length !== 1 ? "s" : ""} retrieved
      </button>
      {open && (
        <div style={{ marginTop: 8 }}>
          {sources.map((src) => (
            <div key={src.rank} style={S.sourceItem}>
              <span style={S.sourceTag}>S{src.rank}</span>
              <div>
                <div style={{ color: "#94a3b8", marginBottom: 2 }}>
                  <strong style={{ color: "#e2e8f0" }}>{src.filename}</strong> · Page {src.page}
                  <span style={{ marginLeft: 8, color: "#64748b" }}>
                    sim: {src.similarity}
                  </span>
                </div>
                <div style={{ color: "#64748b", fontStyle: "italic" }}>{src.snippet}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Message({ msg }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: msg.role === "user" ? "flex-end" : "flex-start" }}>
      <div style={S.bubble(msg.role)}>
        {msg.content || <span style={{ display: "inline-block", width: 16, height: 16, border: "2px solid #7c6cfa", borderTopColor: "transparent", borderRadius: "50%", animation: "spin 1s linear infinite" }} />}
      </div>
      {msg.role === "assistant" && <SourcesAccordion sources={msg.sources} />}
    </div>
  );
}

function EvalPanel({ onClose, selectedDocs }) {
  const [questions, setQuestions] = useState([]);
  const [overall, setOverall] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [scopeAtRun, setScopeAtRun] = useState(null);

  const runEval = useCallback(() => {
    setLoading(true);
    setQuestions([]);
    setOverall(null);
    setError(null);
    setScopeAtRun(null);
    const url = `${API}/evaluate`;

    fetch(url).then(async (r) => {
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail); }
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const event = JSON.parse(line.slice(6));
          if (event.type === "question") setQuestions((prev) => [...prev, event.result]);
          if (event.type === "done") { setOverall(event.overall); setLoading(false); }
        }
      }
    }).catch((e) => { setError(String(e)); setLoading(false); });
  }, [selectedDocs]);

  useEffect(() => { runEval(); }, []);

  const done = questions.length;
  const total = questions[0]?.total ?? 10;

  return (
    <div style={S.evalPanel}>
      <div style={S.evalHeader}>
        <BarChart2 size={16} /> Retrieval Evaluation
        <button onClick={onClose} style={{ ...S.iconBtn, marginLeft: "auto", color: "#94a3b8" }}><X size={16} /></button>
      </div>
      <div style={{ padding: "6px 16px 8px", borderBottom: "1px solid #2d3148", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontSize: 11, color: "#475569" }}>
          Benchmarked on: Attention Is All You Need
        </span>
        <button
          onClick={runEval}
          disabled={loading}
          style={{ fontSize: 11, padding: "3px 10px", background: "#1e2130", border: "1px solid #2d3148", borderRadius: 6, color: loading ? "#475569" : "#7c6cfa", cursor: loading ? "default" : "pointer" }}
        >
          {loading ? `${done}/${total}…` : "Re-run"}
        </button>
      </div>
      <div style={S.evalBody}>
        {loading && done === 0 && (
          <div style={{ textAlign: "center", padding: 24, color: "#64748b" }}>
            <Loader2 size={24} style={{ animation: "spin 1s linear infinite" }} />
            <div style={{ marginTop: 10, fontSize: 12 }}>Starting evaluation…</div>
          </div>
        )}
        {error && <div style={{ color: "#f87171", fontSize: 13 }}>{error}</div>}
        {overall && (
          <div style={S.overallBox}>
            <div style={{ color: "#60a5fa", fontWeight: 700, marginBottom: 8 }}>Overall RAGAS Scores</div>
            <div style={S.scoreRow}>
              <span style={{ fontSize: 13 }}>RAGAS Score</span>
              <span style={S.scorePill(overall.avg_ragas_score)}>{(overall.avg_ragas_score * 100).toFixed(0)}%</span>
            </div>
            <div style={S.scoreRow}>
              <span style={{ fontSize: 12, color: "#94a3b8" }}>Faithfulness</span>
              <span style={{ fontSize: 12, color: "#94a3b8" }}>{(overall.avg_faithfulness * 100).toFixed(0)}%</span>
            </div>
            <div style={S.scoreRow}>
              <span style={{ fontSize: 12, color: "#94a3b8" }}>Answer Relevancy</span>
              <span style={{ fontSize: 12, color: "#94a3b8" }}>{(overall.avg_answer_relevancy * 100).toFixed(0)}%</span>
            </div>
            <div style={S.scoreRow}>
              <span style={{ fontSize: 12, color: "#94a3b8" }}>Correctness</span>
              <span style={{ fontSize: 12, color: "#94a3b8" }}>{(overall.avg_correctness * 100).toFixed(0)}%</span>
            </div>
          </div>
        )}
        {questions.map((q) => (
          <div key={q.id} style={S.scoreCard}>
            <div style={S.scoreQ}>Q{q.id}/{total}: {q.question}</div>
            <div style={{ fontSize: 12, color: "#94a3b8", margin: "8px 0", lineHeight: 1.6, borderLeft: "2px solid #2d3148", paddingLeft: 8 }}>
              {q.answer}
            </div>
            {q.ground_truth && (
              <div style={{ fontSize: 11, color: "#cbd5e1", margin: "4px 0 8px", lineHeight: 1.5, borderLeft: "2px solid #4ade80", paddingLeft: 8 }}>
                <span style={{ color: "#4ade80", fontWeight: 700 }}>Reference: </span>{q.ground_truth}
              </div>
            )}
            <div style={S.scoreRow}>
              <span style={{ fontSize: 12, color: "#64748b" }}>RAGAS Score</span>
              <span style={S.scorePill(q.ragas_score)}>{(q.ragas_score * 100).toFixed(0)}%</span>
            </div>
            <div style={S.scoreRow}>
              <span style={{ fontSize: 11, color: "#475569" }}>Faithfulness</span>
              <span style={{ fontSize: 11, color: "#94a3b8" }}>{(q.faithfulness * 100).toFixed(0)}%</span>
            </div>
            <div style={S.scoreRow}>
              <span style={{ fontSize: 11, color: "#475569" }}>Answer Relevancy</span>
              <span style={{ fontSize: 11, color: "#94a3b8" }}>{(q.answer_relevancy * 100).toFixed(0)}%</span>
            </div>
            <div style={S.scoreRow}>
              <span style={{ fontSize: 11, color: "#475569" }}>Correctness</span>
              <span style={{ fontSize: 11, color: "#94a3b8" }}>{(q.correctness * 100).toFixed(0)}%</span>
            </div>
            <div style={{ fontSize: 11, color: "#475569", marginTop: 4 }}>{q.retrieved_count} chunks retrieved</div>
          </div>
        ))}
        {loading && done > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 0", color: "#64748b", fontSize: 12 }}>
            <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} />
            Running question {done + 1}/{total}…
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [docs, setDocs] = useState([]);
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [uploading, setUploading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [showEval, setShowEval] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [selectedDocs, setSelectedDocs] = useState([]); // [] = all
  const [backendReady, setBackendReady] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(scrollToBottom, [messages]);

  const loadDocs = useCallback(async () => {
    for (let attempt = 0; attempt < 40; attempt++) {
      try {
        const r = await fetch(`${API}/documents`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        setDocs(await r.json());
        setBackendReady(true);
        return;
      } catch {
        if (attempt < 39) await new Promise((res) => setTimeout(res, 5000));
      }
    }
  }, []);

  useEffect(() => { loadDocs(); }, [loadDocs]);

  const handleUpload = async (fileList) => {
    if (!fileList || fileList.length === 0) return;
    setUploading(true);
    const form = new FormData();
    for (const f of fileList) form.append("files", f);
    try {
      const res = await fetch(`${API}/upload`, { method: "POST", body: form });
      const result = await res.json();
      const indexed = result.filter((r) => r.status === "indexed").length;
      const already = result.filter((r) => r.status === "already_indexed").length;
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content: `✅ Indexed ${indexed} new PDF(s)${already ? `, ${already} already existed` : ""}. Ready to chat!`,
          id: Date.now(),
        },
      ]);
      loadDocs();
    } catch (e) {
      setMessages((prev) => [...prev, { role: "system", content: "❌ Upload failed: " + e.message, id: Date.now() }]);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (docId) => {
    await fetch(`${API}/documents/${docId}`, { method: "DELETE" });
    loadDocs();
  };

  const handleSend = async () => {
    if (!query.trim() || streaming) return;
    const q = query.trim();
    setQuery("");
    const userMsg = { role: "user", content: q, id: Date.now() };
    const assistantMsg = { role: "assistant", content: "", sources: [], id: Date.now() + 1 };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setStreaming(true);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          doc_ids: selectedDocs.length > 0 ? selectedDocs : null,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        setMessages((prev) =>
          prev.map((m) => m.id === assistantMsg.id ? { ...m, content: "❌ " + (err.detail || "Error") } : m)
        );
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event = JSON.parse(line.slice(6));
            if (event.type === "sources") {
              setMessages((prev) =>
                prev.map((m) => m.id === assistantMsg.id ? { ...m, sources: event.sources } : m)
              );
            } else if (event.type === "token") {
              setMessages((prev) =>
                prev.map((m) => m.id === assistantMsg.id ? { ...m, content: m.content + event.content } : m)
              );
            }
          } catch (_) {}
        }
      }
    } catch (e) {
      setMessages((prev) =>
        prev.map((m) => m.id === assistantMsg.id ? { ...m, content: "❌ " + e.message } : m)
      );
    } finally {
      setStreaming(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const toggleDocFilter = (docId) => {
    setSelectedDocs((prev) =>
      prev.includes(docId) ? prev.filter((d) => d !== docId) : [...prev, docId]
    );
  };

  return (
    <div style={S.app}>
      {/* Global spinner style */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* ── Sidebar ── */}
      <aside style={S.sidebar}>
        <div style={S.sidebarHeader}>
          <div style={S.logo}><BookOpen size={20} /> RAG Chat</div>
          <div style={{ fontSize: 12, color: "#475569", marginTop: 4 }}>
            Claude · Qdrant · FastAPI
          </div>
        </div>

        <div style={S.section}>
          <div style={S.label}>Upload PDFs</div>
          <div
            style={{
              ...S.uploadZone,
              ...(dragOver && backendReady ? S.uploadZoneHover : {}),
              opacity: backendReady ? 1 : 0.45,
              cursor: backendReady ? "pointer" : "not-allowed",
            }}
            onClick={() => backendReady && fileInputRef.current.click()}
            onDragOver={(e) => { e.preventDefault(); if (backendReady) setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); if (backendReady) handleUpload(e.dataTransfer.files); }}
          >
            {backendReady
              ? <Upload size={24} color={dragOver ? "#7c6cfa" : "#475569"} />
              : <Loader2 size={24} style={{ animation: "spin 1s linear infinite", color: "#7c6cfa" }} />
            }
            <div style={{ fontSize: 12, color: "#64748b", marginTop: 8 }}>
              {backendReady ? "Drop PDFs here or click to browse" : "Backend starting up…"}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              multiple
              style={{ display: "none" }}
              onChange={(e) => handleUpload(e.target.files)}
            />
          </div>
          {uploading && (
            <div style={{
              marginTop: 12,
              padding: "14px 16px",
              background: "rgba(124,108,250,0.15)",
              border: "1px solid #7c6cfa",
              borderRadius: 10,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 10,
            }}>
              <Loader2 size={32} style={{ animation: "spin 1s linear infinite", color: "#7c6cfa" }} />
              <div style={{ fontSize: 14, fontWeight: 600, color: "#7c6cfa" }}>Uploading & Indexing…</div>
              <div style={{ fontSize: 11, color: "#a89cfc", textAlign: "center" }}>
                Processing your PDF. This may take a moment.
              </div>
            </div>
          )}
        </div>

        <div style={{ ...S.section, paddingTop: 0 }}>
          <div style={S.label}>Documents ({docs.length})</div>
        </div>
        <div style={S.docList}>
          {docs.length === 0 && (
            <div style={{ fontSize: 12, color: "#475569", textAlign: "center", padding: 16 }}>
              {backendReady
                ? "No documents yet"
                : <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
                    <Loader2 size={16} style={{ animation: "spin 1s linear infinite", color: "#7c6cfa" }} />
                    <span>Connecting to server…</span>
                  </div>
              }
            </div>
          )}
          {docs.map((doc) => {
            const active = selectedDocs.includes(doc.doc_id);
            return (
              <div
                key={doc.doc_id}
                style={{
                  ...S.docItem,
                  border: active ? "1px solid #7c6cfa" : "1px solid #2d3148",
                  cursor: "pointer",
                }}
                onClick={() => toggleDocFilter(doc.doc_id)}
                title="Click to filter chat to this document"
              >
                <FileText size={14} color={active ? "#7c6cfa" : "#64748b"} />
                <span style={S.docName}>{doc.filename}</span>
                <button
                  style={S.iconBtn}
                  onClick={(e) => { e.stopPropagation(); handleDelete(doc.doc_id); }}
                  title="Delete"
                >
                  <Trash2 size={13} color="#f87171" />
                </button>
              </div>
            );
          })}
          {selectedDocs.length > 0 && (
            <div style={{ fontSize: 11, color: "#7c6cfa", marginTop: 4, textAlign: "center" }}>
              Filtering to {selectedDocs.length} doc(s). Click doc to deselect.
            </div>
          )}
        </div>

        <button style={S.evalBtn} onClick={() => setShowEval((v) => !v)}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <BarChart2 size={18} />
            <span>{showEval ? "Hide RAG Benchmark" : "Run RAG Benchmark"}</span>
          </div>
          {!showEval && (
            <span style={{ fontSize: 14, fontWeight: 700, color: "#e9d5ff" }}>
              10 questions · Attention Is All You Need
            </span>
          )}
        </button>
      </aside>

      {/* ── Chat area ── */}
      <main style={S.main}>
        <div style={S.chatHeader}>
          {docs.length === 0
            ? "Upload a PDF to start chatting"
            : selectedDocs.length > 0
            ? `Chatting with ${selectedDocs.length} selected document(s)`
            : `Chatting with all ${docs.length} document(s)`}
        </div>

        <div style={S.messages}>
          {messages.length === 0 && (
            <div style={{ textAlign: "center", color: "#475569", marginTop: 60 }}>
              <BookOpen size={48} color="#2d3148" />
              <div style={{ marginTop: 16, fontSize: 15 }}>Upload a PDF and start asking questions</div>
              <div style={{ fontSize: 13, marginTop: 6 }}>Answers will include source citations</div>
            </div>
          )}
          {messages.map((msg) =>
            msg.role === "system" ? (
              <div key={msg.id} style={{ alignSelf: "center", fontSize: 12, color: "#64748b", background: "#1e2130", padding: "6px 14px", borderRadius: 99 }}>
                {msg.content}
              </div>
            ) : (
              <Message key={msg.id} msg={msg} />
            )
          )}
          <div ref={messagesEndRef} />
        </div>

        <div style={S.inputBar}>
          <textarea
            style={S.input}
            rows={2}
            placeholder="Ask a question about your documents… (Enter to send, Shift+Enter for newline)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={streaming}
          />
          <button style={S.sendBtn} onClick={handleSend} disabled={streaming || !query.trim()}>
            {streaming
              ? <Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} />
              : <Send size={16} />}
            Send
          </button>
        </div>
      </main>

      {/* ── Eval panel ── */}
      {showEval && <EvalPanel onClose={() => setShowEval(false)} selectedDocs={selectedDocs} />}
    </div>
  );
}
