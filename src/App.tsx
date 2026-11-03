import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import "./ds/styles.css";
import { Button, Eyebrow, Wordmark } from "./ds/ui";
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
  const [, setAgentReady] = useState(false);

  // Refs for scrolling the chat log.
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = useCallback((command: AgentCommand) => {
    agentRef.current?.send(command);
  }, []);

  // Engine logs go to the console with the proper level.
  const log = useCallback((msg: string, level: "info" | "warn" | "error" = "info") => {
    const line = `[nobodywho] ${msg}`;
    if (level === "error") console.error(line);
    else if (level === "warn") console.warn(line);
    else console.log(line);
  }, []);

  // --- Persistence ---
  // Save the chat history whenever it changes.
  useEffect(() => {
    // Don't persist the streaming placeholder — only completed messages.
    const completed = messages
      .filter((m) => m.id !== "streaming")
      .map((m) => ({ ...m, timestamp: Date.now() }));
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
        // Finalize the streaming placeholder with a real id once generation
        // ends, so its key stays unique and the message gets persisted.
        if (s === "ready") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === "streaming" ? { ...m, id: crypto.randomUUID() } : m,
            ),
          );
        }
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
      onError: (msg) => log(msg, "error"),
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
    setMessages((prev) => {
      // Belt and braces: finalize any lingering placeholder first.
      const finalized = prev.map((m) =>
        m.id === "streaming" ? { ...m, id: crypto.randomUUID() } : m,
      );
      return [
        ...finalized,
        { id: crypto.randomUUID(), role: "user" as const, content: text },
        // Placeholder that gets appended to as tokens stream in.
        { id: "streaming", role: "assistant" as const, content: "" },
      ];
    });
    send({ type: "ask", text });
  };

  const isBusy =
    status === "loading" || status === "downloading" || status === "generating";

  return (
    <div className="app">
      <header className="app-head">
        <Wordmark size={22} />
        <div className="header-controls">
          <span
            className={`status status-${status}`}
            title={statusMessage}
          >
            {status}
            {status === "downloading" && ` ${Math.round(progress * 100)}%`}
          </span>
          <Button
            variant="outline"
            aria-expanded={showSettings}
            onClick={() => setShowSettings((s) => !s)}
          >
            Settings
          </Button>
        </div>
      </header>

      {showSettings && (
        <>
          <div
            className="drawer-scrim"
            onClick={() => setShowSettings(false)}
          />
          <aside
            className="drawer"
            role="dialog"
            aria-modal="true"
            aria-label="Settings"
            onKeyDown={(e) => {
              if (e.key === "Escape") setShowSettings(false);
            }}
          >
            <div className="drawer-head">
              <Eyebrow>Settings</Eyebrow>
              <Button
                variant="outline"
                onClick={() => setShowSettings(false)}
              >
                Close
              </Button>
            </div>

            <div className="settings-group">
            <Eyebrow>Model</Eyebrow>
            <div className="nw-field">
              <label htmlFor="model-url">Model</label>
              <div className="sel">
                <select
                  id="model-url"
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
              </div>
            </div>
            <div className="cached-models">
              <Eyebrow>Cached models – {models.length}</Eyebrow>
              {models.length === 0 && <p className="empty">Nothing cached yet.</p>}
              {models.map((m) => (
                <div key={m.path} className="cached-model">
                  <span className="path">{m.path.split("/").pop()}</span>
                  <span className="size">{formatBytes(m.size)}</span>
                  <Button variant="outline" onClick={() => setSelectedUrl(m.path)}>
                    Select
                  </Button>
                </div>
              ))}
            </div>
            <div className="nw-actions">
              <Button onClick={handleDownload} disabled={isBusy}>
                Download model
              </Button>
              <Button onClick={handleLoad} disabled={isBusy}>
                Load model
              </Button>
              <Button variant="outline" onClick={handleUnload} disabled={status === "idle"}>
                Unload model
              </Button>
            </div>
          </div>

          <div className="settings-group">
            <Eyebrow>Inference</Eyebrow>
            <div className="settings-grid">
              <label className="settings-check">
                <input
                  type="checkbox"
                  checked={useGpu}
                  onChange={(e) => setUseGpu(e.target.checked)}
                />
                Use WebGPU (requires model reload)
              </label>
              <label className="settings-check">
                <input
                  type="checkbox"
                  checked={useWorker}
                  onChange={(e) => setUseWorker(e.target.checked)}
                />
                Run inference in Web Worker
              </label>
              <div className="nw-field">
                <label htmlFor="thread-count">
                  CPU threads ({threadCount === 0 ? "auto" : threadCount})
                </label>
                <input
                  id="thread-count"
                  type="range"
                  min={0}
                  max={16}
                  value={threadCount}
                  onChange={(e) => setThreadCount(Number(e.target.value))}
                />
              </div>
              <div className="nw-field nw-field--full">
                <label htmlFor="system-prompt">System prompt</label>
                <textarea
                  id="system-prompt"
                  rows={2}
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                />
              </div>
            </div>
          </div>
        </aside>
        </>
      )}

      <main className="chat-area">
        <div className="messages">
          {status === "idle" && messages.length === 0 && (
            <div className="empty-state">
              <Eyebrow>NobodyWho web chat</Eyebrow>
              <p>Download and load a model from the settings to get started.</p>
            </div>
          )}
          {messages.map((m) => (
            <div key={m.id} className={`msg msg-${m.role}`}>
              <span className="msg-who">
                {m.role === "user" ? "You" : "NobodyWho"}
              </span>
              <div className="msg-body">{m.content || "…"}</div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <div className="input-area">
          <input
            className="chat-input"
            type="text"
            placeholder={
              status === "ready"
                ? "Type a message"
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
            <Button variant="outline" onClick={() => send({ type: "stop" })}>
              Stop
            </Button>
          ) : (
            <Button onClick={handleAsk} disabled={status !== "ready"}>
              Send
            </Button>
          )}
          <Button
            variant="outline"
            onClick={() => {
              setMessages([]);
              send({ type: "reset" });
            }}
            disabled={status !== "ready"}
            title="Clear conversation history"
          >
            Reset
          </Button>
        </div>
      </main>
    </div>
  );
}
