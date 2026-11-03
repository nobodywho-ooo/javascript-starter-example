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
type SuggestedModel = {
  name: string;
  url: string;
  size: string;
  /** Set when the model likely exceeds the current WebGPU budget. */
  warning?: string;
};

const SUGGESTED_MODELS: SuggestedModel[] = [
  {
    name: "LFM2.5-350M (Q4_K_M)",
    url: "https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_K_M.gguf",
    size: "218.7 MB",
  },
  {
    name: "Bonsai-1.7B (Q1_0)",
    url: "https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B-Q1_0.gguf",
    size: "236.8 MB",
  },
  {
    name: "Gemma 3 270M IT (Q4_K_M)",
    url: "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf",
    size: "241.4 MB",
  },
  {
    name: "Qwen3 0.6B (Q4_K_M)",
    url: "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf",
    size: "378.3 MB",
  },
  {
    name: "Llama 3.2 1B Instruct (Q4_K_M)",
    url: "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    size: "770.3 MB",
  },
  {
    name: "Granite 4.0 h-1B (Q4_K_M)",
    url: "https://huggingface.co/ibm-granite/granite-4.0-h-1b-GGUF/resolve/main/granite-4.0-h-1b-Q4_K_M.gguf",
    size: "859.4 MB",
  },
  {
    name: "Qwen3.5 2B (Q4_K_M)",
    url: "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
    size: "1.2 GB",
  },
  {
    name: "SmolLM3 3B (Q4_K_M)",
    url: "https://huggingface.co/unsloth/SmolLM3-3B-GGUF/resolve/main/SmolLM3-3B-Q4_K_M.gguf",
    size: "1.8 GB",
    warning: "Model size exceeds the current WebGPU budget and may fail to load.",
  },
  {
    name: "Ministral 3 3B Instruct 2512 (Q4_K_M)",
    url: "https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
    size: "2.0 GB",
    warning: "Model size exceeds the current WebGPU budget and may fail to load.",
  },
  {
    name: "Gemma 4 E2B IT (Q4_K_M)",
    url: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
    size: "2.9 GB",
    warning: "Model size exceeds the current WebGPU budget and may fail to load.",
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
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  // Tokens/second: a fast 3s rolling window while generating, and the rate
  // over the entire last generation once it finishes.
  const [liveTps, setLiveTps] = useState(0);
  const [lastRunTps, setLastRunTps] = useState(0);
  const genStartRef = useRef<number | null>(null);
  const genTokensRef = useRef(0);
  const tokenTimesRef = useRef<number[]>([]);
  // A message queued while the model auto-loads; flushed on "ready".
  const pendingAskRef = useRef<{ text: string; conversationId: string } | null>(
    null,
  );

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
        if (s === "generating") {
          // Start a fresh tokens/second measurement for this generation.
          genStartRef.current = performance.now();
          genTokensRef.current = 0;
          tokenTimesRef.current = [];
          setLiveTps(0);
        } else if (s === "idle") {
          genStartRef.current = null;
          setLiveTps(0);
          setLastRunTps(0);
        }
        // Finalize any streaming placeholder with a real id once generation
        // ends, so keys stay unique and messages get persisted — in every
        // conversation, in case the user switched mid-generation.
        if (s === "ready") {
          // Final rate over the entire last generation.
          if (genStartRef.current !== null) {
            const elapsed = (performance.now() - genStartRef.current) / 1000;
            if (elapsed > 0) {
              setLastRunTps(genTokensRef.current / elapsed);
            }
            genStartRef.current = null;
          }
          setConversations((prev) =>
            prev.map((c) => ({
              ...c,
              messages: c.messages.map((m) =>
                m.id === "streaming" ? { ...m, id: crypto.randomUUID() } : m,
              ),
            })),
          );
          // Auto-load flush: send the queued message once the model is ready.
          const pending = pendingAskRef.current;
          if (pending && pending.conversationId === activeIdRef.current) {
            pendingAskRef.current = null;
            setConversations((prev) =>
              prev.map((c) =>
                c.id !== pending.conversationId
                  ? c
                  : {
                      ...c,
                      messages: [
                        ...c.messages,
                        {
                          id: "streaming",
                          role: "assistant" as const,
                          content: "",
                          timestamp: Date.now(),
                        },
                      ],
                    },
              ),
            );
            agentRef.current?.send({ type: "ask", text: pending.text });
          }
        }
      },
      onProgress: setProgress,
      onToken: (token) => {
        // Live tokens/second over a 3s rolling window (pure decode speed —
        // the window starts at the first token, so prompt processing and
        // first-token latency don't skew it).
        genTokensRef.current += 1;
        const now = performance.now();
        const times = tokenTimesRef.current;
        times.push(now);
        while (times.length > 0 && now - times[0] > 3000) times.shift();
        const span = Math.max((now - (times[0] ?? now)) / 1000, 0.5);
        setLiveTps(times.length / span);
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
        // Errors persist as part of the conversation's chat history.
        setConversations((prev) =>
          prev.map((c) =>
            c.id !== activeIdRef.current
              ? c
              : {
                  ...c,
                  messages: [
                    ...c.messages,
                    {
                      id: crypto.randomUUID(),
                      role: "error" as const,
                      content: msg,
                      timestamp: Date.now(),
                    },
                  ],
                },
          ),
        );
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

  // --- Settings popover behaviour: close on outside click or Escape ---
  useEffect(() => {
    if (!showSettings) return;
    const onPointerDown = (e: PointerEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target?.closest(".settings-popover") ||
        target?.closest(".settings-toggle")
      ) {
        return;
      }
      setShowSettings(false);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setShowSettings(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [showSettings]);

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
    // Fresh engine context for the conversation we switch to.
    send({ type: "reset" });
  };

  const createConversation = () => {
    const conv = makeConversation(settings.modelUrl);
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
    setInput("");
    send({ type: "reset" });
  };

  const deleteConversation = (id: string) => {
    const next = conversations.filter((c) => c.id !== id);
    const result =
      next.length > 0 ? next : [makeConversation(settings.modelUrl)];
    setConversations(result);
    if (id === active.id) {
      const fallback = result[0];
      setActiveId(fallback.id);
      send({ type: "reset" });
    }
  };

  // --- Handlers ---

  const handleDownload = () => {
    send({ type: "download", url: settings.modelUrl });
  };

  const handleLoad = () => {
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
    setLoadedKey(null);
    send({ type: "unload" });
  };

  const handleAsk = () => {
    const text = input.trim();
    if (!text) return;
    if (status === "loading" || status === "downloading") return;

    // Auto-load: with no model running (or one that doesn't match this
    // conversation's settings), load first and send the message once ready.
    if (status === "idle" || (status === "ready" && modelMismatch)) {
      pendingAskRef.current = { text, conversationId: active.id };
      setInput("");
      updateConversation(active.id, {
        title:
          active.messages.length === 0
            ? text.length > 40
              ? `${text.slice(0, 40)}…`
              : text
            : active.title,
        messages: [
          ...active.messages.map((m) =>
            m.id === "streaming" ? { ...m, id: crypto.randomUUID() } : m,
          ),
          {
            id: crypto.randomUUID(),
            role: "user",
            content: text,
            timestamp: Date.now(),
          },
        ] as StoredMessage[],
      });
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
      return;
    }

    if (status !== "ready") return;

    setInput("");
    updateConversation(active.id, {
      title:
        active.messages.length === 0
          ? text.length > 40
            ? `${text.slice(0, 40)}…`
            : text
          : active.title,
      messages: [
        ...active.messages.map((m) =>
          m.id === "streaming" ? { ...m, id: crypto.randomUUID() } : m,
        ),
        {
          id: crypto.randomUUID(),
          role: "user",
          content: text,
          timestamp: Date.now(),
        },
        // Placeholder that gets appended to as tokens stream in.
        {
          id: "streaming",
          role: "assistant",
          content: "",
          timestamp: Date.now(),
        },
      ] as StoredMessage[],
    });
    send({ type: "ask", text });
  };

  const handleReset = () => {
    updateConversation(active.id, { messages: [] });
    send({ type: "reset" });
  };

  const isBusy =
    status === "loading" || status === "downloading" || status === "generating";

  const modelMismatch =
    loadedKey !== null && loadedKey !== settingsKey(settings);

  const selectedModel = SUGGESTED_MODELS.find(
    (m) => m.url === settings.modelUrl,
  );

  const sortedConversations = [...conversations].sort(
    (a, b) => b.updatedAt - a.updatedAt,
  );

  // Settings popover body — rendered in the sidebar footer (desktop) and
  // the compact header (mobile).
  const settingsBody = (
    <>
      <div className="settings-group">
        <Eyebrow>Model</Eyebrow>
        <p className="popover-note">
          Settings apply to this conversation
          {active.messages.length === 0 && active.title === "New conversation"
            ? "."
            : ` – “${active.title}”.`}
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
                <option value={settings.modelUrl}>{settings.modelUrl}</option>
              )}
            </select>
          </div>
        </div>
        {selectedModel && (
          <p className="popover-note">
            {selectedModel.size}
            {selectedModel.warning && (
              <span className="warn"> – {selectedModel.warning}</span>
            )}
          </p>
        )}
        <div className="cached-models">
          <Eyebrow>Cached models – {models.length}</Eyebrow>
          {models.length === 0 && <p className="empty">Nothing cached yet.</p>}
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
              CPU threads (
              {settings.threadCount === 0 ? "auto" : settings.threadCount})
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
              onChange={(e) => setSettings({ systemPrompt: e.target.value })}
            />
          </div>
        </div>
      </div>
    </>
  );

  const settingsPopover = showSettings ? (
    <div className="settings-popover" role="dialog" aria-label="Settings">
      <div className="popover-head">
        <Eyebrow>Settings</Eyebrow>
        <Button variant="outline" onClick={() => setShowSettings(false)}>
          Close
        </Button>
      </div>
      {settingsBody}
    </div>
  ) : null;

  // One merged status readout: download/load progress while busy, tokens
  // per second otherwise (live window while generating, last run when idle).
  const statusLabel =
    status === "downloading"
      ? `downloading ${Math.round(progress * 100)}%`
      : status === "loading"
        ? "loading…"
        : status === "generating"
          ? `${liveTps.toFixed(1)} tok/s`
          : `${lastRunTps.toFixed(1)} tok/s`;

  const isMetric =
    status === "idle" || status === "ready" || status === "generating";

  const statusBadge = (
    <span
      className={`status status-${status}${isMetric ? " status-metric" : ""}`}
      title={statusMessage}
    >
      {statusLabel}
    </span>
  );

  return (
    <div className="app">
      {/* --- Conversations sidebar --- */}
      <aside className="conv-sidebar">
        <div className="conv-brand">
          <Wordmark size={20} />
        </div>
        <div className="conv-head">
          <Eyebrow>Conversations</Eyebrow>
          <Button variant="outline" onClick={createConversation}>
            New
          </Button>
        </div>
        <div className="conv-list">
          {sortedConversations.map((c) => {
            // A brand-new conversation has no title yet — show it without
            // one until the user has written something.
            const untitled = c.messages.length === 0 && c.title === "New conversation";
            return (
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
              {!untitled && <span className="t">{c.title}</span>}
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
            );
          })}
        </div>
        <div className="conv-foot">
          {settingsPopover}
          <div className="conv-foot-row">
            {statusBadge}
            <Button
              variant="outline"
              className="settings-toggle"
              aria-expanded={showSettings}
              onClick={() => setShowSettings((s) => !s)}
            >
              Settings
            </Button>
          </div>
        </div>
      </aside>

      {/* --- Main column --- */}
      <div className="main-col">
        {/* Compact header on small screens, where the sidebar is hidden */}
        <header className="m-head">
          {statusBadge}
          <Button
            variant="outline"
            className="settings-toggle"
            aria-expanded={showSettings}
            onClick={() => setShowSettings((s) => !s)}
          >
            Settings
          </Button>
          {settingsPopover}
        </header>

        <main className="chat-area">
          <div className="messages" ref={messagesRef}>
            {active.messages.length === 0 && (
              <div className="empty-state">
                <Eyebrow>NobodyWho web chat</Eyebrow>
                <p>Type a message to get started.</p>
              </div>
            )}
            {active.messages.map((m) => (
              <div key={m.id} className={`msg msg-${m.role}`}>
                <div className="bubble">{m.content || "…"}</div>
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
                  : status === "idle"
                    ? "Type a message"
                    : "Loading model…"
              }
              value={input}
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
              <Button
                onClick={handleAsk}
                disabled={status === "loading" || status === "downloading"}
              >
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
