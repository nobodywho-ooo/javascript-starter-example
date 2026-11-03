import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import { createAgent, type Agent, type AgentCommand } from "./agent";
import type { ChatMessage, CachedModel, EngineStatus } from "./engine";
import {
  loadHistory,
  loadSettings,
  saveHistory,
  saveSettings,
} from "./utils/storage";

// Hardcoded list of models to choose from for the demo.
const SUGGESTED_MODELS = [
  {
    name: "Qwen2.5-0.5B-Instruct (Q4_K_M)",
    url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
  },
  {
    name: "Llama-3.2-1B-Instruct (Q4_K_M)",
    url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  },
];

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / k ** i).toFixed(1))} ${sizes[i]}`;
}

export default function App() {
  // --- State ---
  const [status, setStatus] = useState<EngineStatus>("idle");
  const [statusMessage, setStatusMessage] = useState<string>();
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [models, setModels] = useState<CachedModel[]>([]);
  // Initialize from local storage: chats and settings persist across reloads.
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadHistory());
  const [input, setInput] = useState("");
  const [showSettings, setShowSettings] = useState(false);

  // Settings
  const [selectedUrl, setSelectedUrl] = useState(SUGGESTED_MODELS[0].url);
  const [useGpu, setUseGpu] = useState(() => loadSettings().useGpu);
  const [threadCount, setThreadCount] = useState(
    () => loadSettings().threadCount,
  );
  const [useWorker, setUseWorker] = useState(
    () => loadSettings().useWorkerThread,
  );
  const [systemPrompt, setSystemPrompt] = useState(
    () => loadSettings().systemPrompt,
  );

  // Agent lifecycle: we recreate the agent whenever the worker setting
  // changes, and tear it down when unmounting.
  const agentRef = useRef<Agent | null>(null);
  const [agentReady, setAgentReady] = useState(false);

  // Refs for scrolling the chat log.
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const logRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat and logs.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const send = useCallback((command: AgentCommand) => {
    agentRef.current?.send(command);
  }, []);

  const log = useCallback((msg: string) => {
    const time = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    setLogs((prev) => [...prev.slice(-100), `[${time}] ${msg}`]);
  }, []);

  // --- Persistence ---
  // Save the chat history whenever it changes.
  useEffect(() => {
    // Don't persist the streaming placeholder — only completed messages.
    const completed = messages.filter((m) => m.id !== "streaming");
    saveHistory(completed);
  }, [messages]);

  // Save settings whenever they change.
  useEffect(() => {
    saveSettings({
      useGpu,
      threadCount,
      useWorkerThread: useWorker,
      systemPrompt,
    });
  }, [useGpu, threadCount, useWorker, systemPrompt]);

  // --- Agent Setup ---
  useEffect(() => {
    setAgentReady(false);
    let disposed = false;
    createAgent(useWorker, {
      onLog: log,
      onStatus: (s, msg) => {
        setStatus(s);
        setStatusMessage(msg);
      },
      onProgress: setProgress,
      onToken: (token) => {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.role === "assistant" && last.id === "streaming") {
            return [
              ...prev.slice(0, -1),
              { ...last, content: last.content + token },
            ];
          }
          return prev;
        });
      },
      onError: (msg) => log(`ERROR: ${msg}`),
      onModels: setModels,
    }).then((agent) => {
      if (disposed) {
        agent.dispose?.();
        return;
      }
      agentRef.current = agent;
      setAgentReady(true);
      agent.send({ type: "list-models" });
    });
    return () => {
      disposed = true;
      agentRef.current?.dispose?.();
      agentRef.current = null;
    };
  }, [useWorker, log]);

  // --- Handlers ---

  const handleDownload = () => {
    send({ type: "download", url: selectedUrl });
  };

  const handleLoad = () => {
    // Clear the visible chat when switching models.
    setMessages([]);
    send({
      type: "load",
      options: { url: selectedUrl, useGpu, threadCount, systemPrompt },
    });
  };

  const handleUnload = () => {
    setMessages([]);
    send({ type: "unload" });
  };

  const handleAsk = () => {
    const text = input.trim();
    if (!text || status !== "ready") return;
    setInput("");
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", content: text },
      // Placeholder that gets appended to as tokens stream in.
      { id: "streaming", role: "assistant", content: "" },
    ]);
    send({ type: "ask", text });
  };

  const isBusy =
    status === "loading" || status === "downloading" || status === "generating";

  return (
    <div className="app">
      <header>
        <h1>NobodyWho Web Chat</h1>
        <div className="header-controls">
          <span className={`status-badge status-${status}`}>
            {status}
            {status === "downloading" && ` (${Math.round(progress * 100)}%)`}
          </span>
          <button
            className="icon-btn"
            onClick={() => setShowSettings((s) => !s)}
            title="Settings"
          >
            ⚙️
          </button>
        </div>
      </header>

      {showSettings && (
        <div className="settings-panel">
          <h2>Model</h2>
          <label>
            Model URL
            <select
              value={selectedUrl}
              onChange={(e) => setSelectedUrl(e.target.value)}
            >
              {SUGGESTED_MODELS.map((m) => (
                <option key={m.url} value={m.url}>
                  {m.name}
                </option>
              ))}
              {!SUGGESTED_MODELS.some((m) => m.url === selectedUrl) && (
                <option value={selectedUrl}>{selectedUrl}</option>
              )}
            </select>
          </label>
          <div className="cached-models">
            <h3>Cached models ({models.length})</h3>
            {models.length === 0 && <p>No models cached yet.</p>}
            {models.map((m) => (
              <div key={m.path} className="cached-model">
                <span className="path">{m.path.split("/").pop()}</span>
                <span className="size">{formatBytes(m.size)}</span>
                <button onClick={() => setSelectedUrl(m.path)}>Select</button>
              </div>
            ))}
          </div>
          <div className="btn-row">
            <button onClick={handleDownload} disabled={isBusy}>
              Download model
            </button>
            <button onClick={handleLoad} disabled={isBusy}>
              Load model
            </button>
            <button onClick={handleUnload} disabled={status === "idle"}>
              Unload model
            </button>
          </div>

          <h2>Settings</h2>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={useGpu}
              onChange={(e) => setUseGpu(e.target.checked)}
            />
            Use WebGPU (requires model reload)
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={useWorker}
              onChange={(e) => setUseWorker(e.target.checked)}
            />
            Run inference in Web Worker
          </label>
          <label>
            CPU threads ({threadCount === 0 ? "auto" : threadCount})
            <input
              type="range"
              min={0}
              max={16}
              value={threadCount}
              onChange={(e) => setThreadCount(Number(e.target.value))}
            />
          </label>
          <label>
            System prompt
            <textarea
              rows={2}
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
            />
          </label>
        </div>
      )}

      <main className="chat-area">
        {status === "idle" && messages.length === 0 && (
          <div className="empty-state">
            <p>Download and load a model from the settings to get started.</p>
          </div>
        )}
        <div className="messages">
          {messages.map((m) => (
            <div key={m.id} className={`message ${m.role}`}>
              <div className="bubble">{m.content || "…"}</div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <div className="input-area">
          <input
            type="text"
            placeholder={
              status === "ready"
                ? "Type a message..."
                : "Load a model to start chatting"
            }
            value={input}
            disabled={status !== "ready" && status !== "generating"}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (status === "generating") {
                  send({ type: "stop" });
                } else {
                  handleAsk();
                }
              }
            }}
          />
          {status === "generating" ? (
            <button onClick={() => send({ type: "stop" })}>Stop</button>
          ) : (
            <button onClick={handleAsk} disabled={status !== "ready"}>
              Send
            </button>
          )}
          <button
            onClick={() => {
              setMessages([]);
              send({ type: "reset" });
            }}
            disabled={status !== "ready"}
            title="Clear conversation history"
          >
            Reset
          </button>
        </div>
      </main>

      <footer className="log-panel">
        <div className="log-header">Logs {statusMessage && `— ${statusMessage}`}</div>
        <div className="log-content" ref={logRef}>
          {logs.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </div>
      </footer>
    </div>
  );
}
