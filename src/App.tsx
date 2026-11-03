import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import "./ds/styles.css";
import { Button, Eyebrow, Wordmark } from "./ds/ui";
import { createAgent, type Agent, type AgentCommand } from "./agent";
import type { CachedModel, EngineStatus } from "./engine";
import {
  loadActiveConversationId,
  loadConversations,
  loadSettings,
  saveActiveConversationId,
  saveConversations,
  saveSettings,
  type Conversation,
  type ConversationSettings,
  type StoredMessage,
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

const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / k ** i).toFixed(1))} ${sizes[i]}`;
}

function makeConversation(modelUrl: string): Conversation {
  const now = Date.now();
  return {
    id: crypto.randomUUID(),
    title: "New conversation",
    createdAt: now,
    updatedAt: now,
    messages: [],
    settings: {
      modelUrl,
      useGpu: false,
      threadCount: 0, // 0 = auto
      systemPrompt: DEFAULT_SYSTEM_PROMPT,
    },
  };
}

/** Identity of the model configuration an engine session is running. */
function settingsKey(s: ConversationSettings): string {
  return JSON.stringify([s.modelUrl, s.useGpu, s.threadCount, s.systemPrompt]);
}

export default function App() {
  // --- Conversations ---
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    const stored = loadConversations();
    return stored.length > 0
      ? stored
      : [makeConversation(SUGGESTED_MODELS[0].url)];
  });
  const [activeId, setActiveId] = useState<string>(
    () => loadActiveConversationId() ?? "",
  );
  const active =
    conversations.find((c) => c.id === activeId) ?? conversations[0];
  const settings = active.settings;

  // --- Engine state (global — one engine session) ---
  const [status, setStatus] = useState<EngineStatus>("idle");
  const [statusMessage, setStatusMessage] = useState<string>();
  const [progress, setProgress] = useState(0);
  const [models, setModels] = useState<CachedModel[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loadedKey, setLoadedKey] = useState<string | null>(null);

  // --- UI state ---
  const [input, setInput] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [useWorker, setUseWorker] = useState(
    () => loadSettings().useWorkerThread,
  );

  // The agent holds event callbacks that outlive renders, so keep a ref to
  // the active conversation id for use inside those callbacks.
  const activeIdRef = useRef(active.id);
  useEffect(() => {
    activeIdRef.current = active.id;
  }, [active.id]);

  // Refs for scrolling the chat log.
  const messagesRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const statusRef = useRef(status);
  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  // Auto-scroll chat — but never while tokens are streaming in, and only
  // when the reader is already at (or near) the bottom, so scrolling back
  // through a response is never interrupted.
  useEffect(() => {
    if (statusRef.current === "generating") return;
    const el = messagesRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distance > 48) return; // user has scrolled up
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [active.messages]);

  // --- Persistence ---
  useEffect(() => {
    // Don't persist the streaming placeholder — only completed messages.
    const completed = conversations.map((c) => ({
      ...c,
      messages: c.messages.filter((m) => m.id !== "streaming"),
    }));
    saveConversations(completed);
  }, [conversations]);

  useEffect(() => {
    if (active.id !== activeId) {
      setActiveId(active.id);
    }
    saveActiveConversationId(active.id);
  }, [active.id, activeId]);

  useEffect(() => {
    saveSettings({ useWorkerThread: useWorker });
  }, [useWorker]);

  // --- Console logging with proper levels ---
  const log = useCallback(
    (msg: string, level: "info" | "warn" | "error" = "info") => {
      const line = `[nobodywho] ${msg}`;
      if (level === "error") console.error(line);
      else if (level === "warn") console.warn(line);
      else console.log(line);
    },
    [],
  );

  // --- Agent lifecycle ---
  const agentRef = useRef<Agent | null>(null);

  useEffect(() => {
    let disposed = false;
    createAgent(useWorker, {
      onLog: log,
      onStatus: (s, msg) => {
        setStatus(s);
        setStatusMessage(msg);
        // Finalize any streaming placeholder with a real id once generation
        // ends, so keys stay unique and messages get persisted — in every
        // conversation, in case the user switched mid-generation.
        if (s === "ready") {
          setConversations((prev) =>
            prev.map((c) => ({
              ...c,
              messages: c.messages.map((m) =>
                m.id === "streaming" ? { ...m, id: crypto.randomUUID() } : m,
              ),
            })),
          );
        }
      },
      onProgress: setProgress,
      onToken: (token) => {
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== activeIdRef.current) return c;
            const last = c.messages[c.messages.length - 1];
            if (last && last.id === "streaming") {
              return {
                ...c,
                messages: [
                  ...c.messages.slice(0, -1),
                  { ...last, content: last.content + token },
                ],
              };
            }
            return c;
          }),
        );
      },
      onError: (msg) => {
        log(msg, "error");
        setError(msg);
      },
      onModels: setModels,
    }).then((agent) => {
      if (disposed) {
        agent.dispose?.();
        return;
      }
      agentRef.current = agent;
      agent.send({ type: "list-models" });
    });
    return () => {
      disposed = true;
      agentRef.current?.dispose?.();
      agentRef.current = null;
    };
  }, [useWorker, log]);

  const send = useCallback((command: AgentCommand) => {
    agentRef.current?.send(command);
  }, []);

  // --- Conversation helpers ---

  const updateConversation = (id: string, patch: Partial<Conversation>) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === id ? { ...c, ...patch, updatedAt: Date.now() } : c,
      ),
    );
  };

  const setSettings = (patch: Partial<ConversationSettings>) => {
    updateConversation(active.id, { settings: { ...settings, ...patch } });
  };

  const switchConversation = (id: string) => {
    if (id === active.id) return;
    setActiveId(id);
    setInput("");
    setError(null);
    // Fresh engine context for the conversation we switch to.
    send({ type: "reset" });
  };

  const createConversation = () => {
    const conv = makeConversation(settings.modelUrl);
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
    setInput("");
    setError(null);
    send({ type: "reset" });
  };

  const deleteConversation = (id: string) => {
    const next = conversations.filter((c) => c.id !== id);
    const result = next.length > 0 ? next : [makeConversation(settings.modelUrl)];
    setConversations(result);
    if (id === active.id) {
      const fallback = result[0];
      setActiveId(fallback.id);
      setError(null);
      send({ type: "reset" });
    }
  };

  // --- Handlers ---

  const handleDownload = () => {
    setError(null);
    send({ type: "download", url: settings.modelUrl });
  };

  const handleLoad = () => {
    setError(null);
    send({
      type: "load",
      options: {
        url: settings.modelUrl,
        useGpu: settings.useGpu,
        threadCount: settings.threadCount,
        systemPrompt: settings.systemPrompt,
      },
    });
    setLoadedKey(settingsKey(settings));
  };

  const handleUnload = () => {
    setError(null);
    setLoadedKey(null);
    send({ type: "unload" });
  };

  const handleAsk = () => {
    const text = input.trim();
    if (!text || status !== "ready") return;
    setInput("");
    setError(null);
    updateConversation(active.id, {
      title:
        active.messages.length === 0
          ? text.length > 40
            ? `${text.slice(0, 40)}…`
            : text
          : active.title,
      messages: [
        ...active.messages.map((m) =>
          m.id === "streaming"
            ? { ...m, id: crypto.randomUUID() }
            : m,
        ),
        { id: crypto.randomUUID(), role: "user", content: text, timestamp: Date.now() },
        // Placeholder that gets appended to as tokens stream in.
        { id: "streaming", role: "assistant", content: "", timestamp: Date.now() },
      ] as StoredMessage[],
    });
    send({ type: "ask", text });
  };

  const handleReset = () => {
    setError(null);
    updateConversation(active.id, { messages: [] });
    send({ type: "reset" });
  };

  const isBusy =
    status === "loading" || status === "downloading" || status === "generating";

  const modelMismatch =
    loadedKey !== null && loadedKey !== settingsKey(settings);
  const modelNotLoaded = loadedKey === null && active.messages.length > 0;

  const sortedConversations = [...conversations].sort(
    (a, b) => b.updatedAt - a.updatedAt,
  );

  return (
    <div className="app">
      {/* --- Conversations sidebar --- */}
      <aside className="conv-sidebar">
        <div className="conv-head">
          <Eyebrow>Conversations</Eyebrow>
          <Button variant="outline" onClick={createConversation}>
            New
          </Button>
        </div>
        <div className="conv-list">
          {sortedConversations.map((c) => (
            <div
              key={c.id}
              role="button"
              tabIndex={0}
              className={`conv-item${c.id === active.id ? " is-active" : ""}`}
              onClick={() => switchConversation(c.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  switchConversation(c.id);
                }
              }}
            >
              <span className="t">{c.title}</span>
              <span className="meta">
                {c.messages.length} messages ·{" "}
                {c.settings.modelUrl.split("/").pop()?.replace(/\.gguf$/i, "")}
              </span>
              <button
                className="conv-del"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteConversation(c.id);
                }}
                title="Delete conversation"
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* --- Main column --- */}
      <div className="main-col">
        <header className="app-head">
          <Wordmark size={22} />
          <div className="header-controls">
            <span className={`status status-${status}`} title={statusMessage}>
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
                <p className="drawer-note">
                  Settings apply to this conversation — “{active.title}”.
                </p>
                <div className="nw-field">
                  <label htmlFor="model-url">Model</label>
                  <div className="sel">
                    <select
                      id="model-url"
                      value={settings.modelUrl}
                      onChange={(e) => setSettings({ modelUrl: e.target.value })}
                    >
                      {SUGGESTED_MODELS.map((m) => (
                        <option key={m.url} value={m.url}>
                          {m.name}
                        </option>
                      ))}
                      {!SUGGESTED_MODELS.some((m) => m.url === settings.modelUrl) && (
                        <option value={settings.modelUrl}>
                          {settings.modelUrl}
                        </option>
                      )}
                    </select>
                  </div>
                </div>
                <div className="cached-models">
                  <Eyebrow>Cached models – {models.length}</Eyebrow>
                  {models.length === 0 && (
                    <p className="empty">Nothing cached yet.</p>
                  )}
                  {models.map((m) => (
                    <div key={m.path} className="cached-model">
                      <span className="path">{m.path.split("/").pop()}</span>
                      <span className="size">{formatBytes(m.size)}</span>
                      <Button
                        variant="outline"
                        onClick={() => setSettings({ modelUrl: m.path })}
                      >
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
                  <Button
                    variant="outline"
                    onClick={handleUnload}
                    disabled={status === "idle"}
                  >
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
                      checked={settings.useGpu}
                      onChange={(e) => setSettings({ useGpu: e.target.checked })}
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
                      CPU threads ({settings.threadCount === 0 ? "auto" : settings.threadCount})
                    </label>
                    <input
                      id="thread-count"
                      type="range"
                      min={0}
                      max={16}
                      value={settings.threadCount}
                      onChange={(e) =>
                        setSettings({ threadCount: Number(e.target.value) })
                      }
                    />
                  </div>
                  <div className="nw-field nw-field--full">
                    <label htmlFor="system-prompt">System prompt</label>
                    <textarea
                      id="system-prompt"
                      rows={2}
                      value={settings.systemPrompt}
                      onChange={(e) =>
                        setSettings({ systemPrompt: e.target.value })
                      }
                    />
                  </div>
                </div>
              </div>
            </aside>
          </>
        )}

        <main className="chat-area">
          <div className="messages" ref={messagesRef}>
            {active.messages.length === 0 && (
              <div className="empty-state">
                <Eyebrow>NobodyWho web chat</Eyebrow>
                <p>Download and load a model from the settings to get started.</p>
              </div>
            )}
            {active.messages.map((m) => (
              <div key={m.id} className={`msg msg-${m.role}`}>
                <span className="msg-who">
                  {m.role === "user" ? "You" : "NobodyWho"}
                </span>
                <div className="bubble">{m.content || "…"}</div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {error && (
            <div className="error-note" role="alert">
              <span className="label">Error</span>
              <p>{error}</p>
              <button className="error-dismiss" onClick={() => setError(null)}>
                Dismiss
              </button>
            </div>
          )}

          {(modelMismatch || modelNotLoaded) && (
            <div className="model-note mono">
              {modelNotLoaded
                ? "Model not loaded — load it from settings to continue this conversation."
                : "Loaded model differs from this conversation's settings — reload to apply them."}
            </div>
          )}

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
              onClick={handleReset}
              disabled={status !== "ready"}
              title="Clear conversation history"
            >
              Reset
            </Button>
          </div>
        </main>
      </div>
    </div>
  );
}
