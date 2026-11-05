import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import "./ds/styles.css";
import { Button, Eyebrow, Wordmark } from "./ds/ui";
import { createAgent, type Agent, type AgentCommand } from "./agent";
import type {
  CachedModel,
  EngineStatus,
  HistoryMessage,
  LoadOptions,
} from "./engine";
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
  },
  {
    name: "Ministral 3 3B Instruct 2512 (Q4_K_M)",
    url: "https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
    size: "2.0 GB",
  },
  {
    name: "Gemma 4 E2B IT (Q4_K_M)",
    url: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
    size: "2.9 GB",
  },
];

/** Filename part of a URL or filesystem path. */
function filenameOf(pathOrUrl: string): string {
  return pathOrUrl.split("/").pop() ?? pathOrUrl;
}

/** Parse a human size like "218.7 MB" or "1.2 GB" into bytes, for sorting. */
function parseSizeBytes(size: string): number {
  const match = size.match(/^([\d.]+)\s*(KB|MB|GB|TB)$/i);
  if (!match) return Number.POSITIVE_INFINITY;
  const mult =
    { kb: 1024, mb: 1024 ** 2, gb: 1024 ** 3, tb: 1024 ** 4 }[
      match[2].toLowerCase()
    ] ?? 1;
  return parseFloat(match[1]) * mult;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / k ** i).toFixed(1))} ${sizes[i]}`;
}

function makeConversation(): Conversation {
  const now = Date.now();
  return {
    id: crypto.randomUUID(),
    title: "New conversation",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

/** A conversation the user never used: nothing written, still untitled. */
function isEmptyConversation(c: Conversation): boolean {
  return c.messages.length === 0 && c.title === "New conversation";
}

/** Map the app's settings onto the engine's LoadOptions — the single place
 * where that mapping lives. Both load call sites build their options with
 * this, and settingsKey is derived from it, so a field can't be applied in
 * one path but not the other (or counted by the key but not actually sent).
 * When adding a ConversationSettings field, wire it in here; omitted fields
 * (nThreads, enableThinking) mean "engine/template default". */
function loadOptions(s: ConversationSettings): LoadOptions {
  return {
    url: s.modelUrl,
    useGpu: s.useGpu,
    systemPrompt: s.systemPrompt,
    enableThinking: s.enableThinking,
    nThreads: s.nThreads,
  };
}

// Compile-time guard attempt removed: LoadOptions renames modelUrl → url,
// so a keyof-level check can't hold. The guard against drift is structural:
// there is exactly one place (loadOptions above) where settings become load
// options, and both call sites plus settingsKey use it.

/** Identity of the model configuration an engine session is running. */
function settingsKey(s: ConversationSettings): string {
  // Derived from loadOptions so the key always describes exactly what gets
  // sent to the engine on load.
  return JSON.stringify(loadOptions(s));
}

export default function App() {
  // --- Conversations ---
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    // Drop conversations the user never wrote anything in — otherwise empty
    // "New conversation" entries accumulate across restarts.
    const stored = loadConversations().filter((c) => !isEmptyConversation(c));
    return stored.length > 0 ? stored : [makeConversation()];
  });
  const [activeId, setActiveId] = useState<string>(
    () => loadActiveConversationId() ?? "",
  );
  const active =
    conversations.find((c) => c.id === activeId) ?? conversations[0];

  // The agent holds event callbacks that outlive renders, so keep the active
  // conversation id in a ref for use inside those callbacks. Updated through
  // `changeActiveId` so the ref can never drift from the state.
  const activeIdRef = useRef(active.id);
  const changeActiveId = (id: string) => {
    activeIdRef.current = id;
    setActiveId(id);
  };

  // The engine handle and a stable `send`. Declared early so callbacks and
  // effects below can use them without ordering hazards.
  const agentRef = useRef<Agent | null>(null);
  // Set when the agent-lifecycle effect's cleanup runs, cleared when it
  // starts — see the effect body for why.
  const disposedRef = useRef(false);
  const send = useCallback((command: AgentCommand) => {
    agentRef.current?.send(command);
  }, []);

  const updateConversation = (id: string, patch: Partial<Conversation>) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === id ? { ...c, ...patch, updatedAt: Date.now() } : c,
      ),
    );
  };

  // Model/engine settings are global — shared by every conversation.
  const [settings, setSettingsState] = useState<ConversationSettings>(() => {
    const stored = loadSettings().model;
    const base = {
      ...stored,
      modelUrl: stored.modelUrl || SUGGESTED_MODELS[3].url,
    };
    // Older stored settings predate the field — default it on.
    base.enableThinking ??= true;
    return base;
  });
  const setSettings = (patch: Partial<ConversationSettings>) => {
    const next = { ...settings, ...patch };
    setSettingsState(next);
    // Options apply to the running session — if the change invalidates it,
    // unload now; the model reloads automatically on the next message.
    if (loadedKey !== null && settingsKey(next) !== loadedKey) {
      setLoadedKey(null);
      setLoadedModelUrl(null);
      send({ type: "unload" });
    }
  };

  // --- Engine state (global — one engine session) ---
  const [status, setStatus] = useState<EngineStatus>("idle");
  const [statusMessage, setStatusMessage] = useState<string>();
  const [progress, setProgress] = useState(0);
  const [models, setModels] = useState<CachedModel[]>([]);
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  // The model URL the engine session is actually running, so per-model rows
  // can show LOADED even if unrelated settings have changed since the load.
  const [loadedModelUrl, setLoadedModelUrl] = useState<string | null>(null);
  // The model a load command is currently fetching/initializing, if any.
  const [loadingModelUrl, setLoadingModelUrl] = useState<string | null>(null);
  // Set when a load command is sent, confirmed when the engine reports ready.
  const pendingLoadUrlRef = useRef<string | null>(null);
  const pendingLoadKeyRef = useRef<string | null>(null);
  // True from load command to load completion — used to sync conversation
  // history into the freshly-initialized engine session.
  const loadInFlightRef = useRef(false);
  // Tokens/second: a fast 3s rolling window while generating, and the rate
  // over the entire last generation once it finishes.
  const [liveTps, setLiveTps] = useState(0);
  const [lastRunTps, setLastRunTps] = useState(0);
  // Tokens generated this run, and the timestamps of the first and latest
  // token — the whole-run average uses these.
  const genTokensRef = useRef(0);
  const firstTokenTimeRef = useRef(0);
  const lastTokenTimeRef = useRef(0);
  // Token timestamps inside the rolling window (live rate only).
  const tokenTimesRef = useRef<number[]>([]);
  // When the live badge was last updated — it is throttled, since updating
  // it per token re-renders the whole app per token.
  const lastTpsUpdateRef = useRef(0);
  // A message queued while the model auto-loads; flushed on "ready".
  const pendingAskRef = useRef<{ text: string; conversationId: string } | null>(
    null,
  );
  // Streamed tokens are buffered and flushed to state on a throttle, not per
  // token: a per-token setConversations copies every conversation and every
  // message, so its cost grows with history length — which is exactly why
  // tok/s appeared to decay as conversations got longer.
  const streamBufRef = useRef("");
  const streamFlushRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const flushStreamBuffer = useCallback(() => {
    if (streamFlushRef.current !== null) {
      clearTimeout(streamFlushRef.current);
      streamFlushRef.current = null;
    }
    const chunk = streamBufRef.current;
    if (!chunk) return;
    streamBufRef.current = "";
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== activeIdRef.current) return c;
        const last = c.messages[c.messages.length - 1];
        if (last && last.id === "streaming") {
          return {
            ...c,
            messages: [
              ...c.messages.slice(0, -1),
              { ...last, content: last.content + chunk },
            ],
          };
        }
        return c;
      }),
    );
  }, []);

  // --- UI state ---
  const [input, setInput] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [useWorker, setUseWorker] = useState(
    () => loadSettings().useWorkerThread,
  );

  const conversationsRef = useRef(conversations);
  useEffect(() => {
    conversationsRef.current = conversations;
  }, [conversations]);

  /** Build engine chat history from a conversation's stored messages.
   * `excludeLastUserText` drops a just-queued user message that the
   * following ask() will add itself. */
  const historyFor = (
    conversationId: string,
    excludeLastUserText?: string,
  ): HistoryMessage[] => {
    const conv = conversationsRef.current.find((c) => c.id === conversationId);
    const msgs = (conv?.messages ?? []).filter(
      (m) =>
        (m.role === "user" || m.role === "assistant") &&
        m.id !== "streaming" &&
        m.content.trim() !== "",
    );
    const last = msgs[msgs.length - 1];
    if (
      excludeLastUserText !== undefined &&
      last &&
      last.role === "user" &&
      last.content === excludeLastUserText
    ) {
      msgs.pop();
    }
    return msgs.map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content,
    }));
  };

  // Refs for scrolling the chat log.
  const messagesRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  // Keep the chat input focused by default — and refocus when the active
  // conversation changes (new chat, switch, etc.).
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    inputRef.current?.focus();
  }, [active.id]);
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
    // Don't persist the streaming placeholder — only completed messages —
    // and don't persist conversations the user never wrote anything in.
    const completed = conversations
      .filter((c) => !isEmptyConversation(c))
      .map((c) => ({
        ...c,
        messages: c.messages.filter((m) => m.id !== "streaming"),
      }));
    saveConversations(completed);
  }, [conversations]);

  useEffect(() => {
    saveActiveConversationId(active.id);
  }, [active.id]);

  useEffect(() => {
    saveSettings({ useWorkerThread: useWorker, model: settings });
  }, [useWorker, settings]);

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
  useEffect(() => {
    // Tracks whether this effect's cleanup already ran — the agent promise
    // may resolve after teardown (StrictMode remount, worker toggle), in
    // which case the freshly created agent must be disposed immediately.
    // A ref is used instead of a closure variable because effect runs are
    // serialized (cleanup always precedes the next run).
    disposedRef.current = false;
    // A fresh engine has no model — never inherit status/loading state from
    // a previous agent instance (StrictMode remount, HMR, worker toggle),
    // or a send could race past the auto-load path.
    // oxlint-disable-next-line react/set-state-in-effect
    setStatus("idle");
    setLoadedKey(null);
    setLoadedModelUrl(null);
    setLoadingModelUrl(null);
    setProgress(0);
    createAgent(useWorker, {
      onLog: log,
      onStatus: (s, msg) => {
        setStatus(s);
        setStatusMessage(msg);
        if (s === "loading" || s === "downloading") {
          loadInFlightRef.current = true;
        }
        if (s === "generating") {
          // Start a fresh tokens/second measurement for this generation.
          genTokensRef.current = 0;
          firstTokenTimeRef.current = 0;
          lastTokenTimeRef.current = 0;
          tokenTimesRef.current = [];
          lastTpsUpdateRef.current = 0;
          setLiveTps(0);
        } else if (s === "idle") {
          setLiveTps(0);
          setLastRunTps(0);
          // Nothing is loaded (unloaded, failed load, or stopped download).
          setLoadedModelUrl(null);
          setLoadingModelUrl(null);
        }
        if (s === "ready") {
          const justLoaded = loadInFlightRef.current;
          loadInFlightRef.current = false;
          // A freshly-loaded session starts with an empty history — sync
          // the engine with the active conversation's stored messages. If a
          // queued message is about to be asked, exclude it (the ask below
          // adds it itself).
          if (justLoaded) {
            const queued = pendingAskRef.current;
            const excluded =
              queued && queued.conversationId === activeIdRef.current
                ? queued.text
                : undefined;
            send({
              type: "set-history",
              messages: historyFor(activeIdRef.current, excluded),
            });
          }
          // A load finished — confirm the model row states.
          setLoadedModelUrl(pendingLoadUrlRef.current);
          setLoadedKey(pendingLoadKeyRef.current);
          setLoadingModelUrl(null);
        }
        // Finalize any streaming placeholder with a real id once generation
        // ends, so keys stay unique and messages get persisted — in every
        // conversation, in case the user switched mid-generation.
        if (s === "ready") {
          // Apply any still-buffered tokens before the finalizer turns the
          // streaming placeholder into a persisted message.
          flushStreamBuffer();
          // Final rate over the entire last generation — decode-only, since
          // timing starts at the first token. N tokens span N-1 intervals.
          const count = genTokensRef.current;
          const span =
            (lastTokenTimeRef.current - firstTokenTimeRef.current) / 1000;
          if (count > 1 && span > 0) {
            setLastRunTps((count - 1) / span);
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
      onTokens: (count) => {
        // Attribute the session's context usage to whichever conversation
        // was active during generation.
        updateConversation(activeIdRef.current, { tokens: count });
      },
      onToken: (token) => {
        // Live tokens/second over a 3s rolling window (pure decode speed —
        // the window starts at the first token, so prompt processing and
        // first-token latency don't skew it).
        const now = performance.now();
        if (genTokensRef.current === 0) firstTokenTimeRef.current = now;
        lastTokenTimeRef.current = now;
        genTokensRef.current += 1;
        const times = tokenTimesRef.current;
        times.push(now);
        while (now - times[0] > 3000) times.shift();
        // N timestamps span N-1 intervals. Throttle the state update to
        // ~4/s — the badge doesn't need per-token precision.
        if (times.length > 1 && now - lastTpsUpdateRef.current >= 250) {
          const span = (now - times[0]) / 1000;
          if (span > 0) {
            lastTpsUpdateRef.current = now;
            setLiveTps((times.length - 1) / span);
          }
        }
        // Buffer the token; the throttled flush applies it to state.
        streamBufRef.current += token;
        if (streamFlushRef.current === null) {
          streamFlushRef.current = setTimeout(flushStreamBuffer, 50);
        }
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
      if (disposedRef.current) {
        agent.dispose?.();
        return;
      }
      agentRef.current = agent;
      agent.send({ type: "list-models" });
    });
    return () => {
      disposedRef.current = true;
      if (streamFlushRef.current !== null) {
        clearTimeout(streamFlushRef.current);
        streamFlushRef.current = null;
      }
      agentRef.current?.dispose?.();
      agentRef.current = null;
    };
  }, [useWorker, log, flushStreamBuffer, send]);

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

  const switchConversation = (id: string) => {
    if (id === active.id) return;
    changeActiveId(id);
    setInput("");
    // Load the target conversation's history into the engine session.
    send({ type: "set-history", messages: historyFor(id) });
    // Drop untouched drafts left behind — they were never visible in the
    // sidebar, so silently discarding them keeps state tidy.
    setConversations((prev) =>
      prev.filter((c) => c.id === id || !isEmptyConversation(c)),
    );
  };

  const createConversation = () => {
    // Reuse the current untouched draft instead of stacking up invisible
    // empty conversations.
    if (isEmptyConversation(active)) {
      setInput("");
      return;
    }
    const conv = makeConversation();
    setConversations((prev) => [conv, ...prev]);
    changeActiveId(conv.id);
    setInput("");
    send({ type: "reset" });
  };

  const deleteConversation = (id: string) => {
    const next = conversations.filter((c) => c.id !== id);
    const result = next.length > 0 ? next : [makeConversation()];
    setConversations(result);
    if (id === active.id) {
      const fallback = result[0];
      changeActiveId(fallback.id);
      send({ type: "reset" });
    }
  };

  // --- Handlers ---

  const handleUnload = () => {
    // Don't clear state here: the engine may defer the unload (e.g. during
    // generation), so let the status events ("idle") confirm it.
    send({ type: "unload" });
  };

  /** SELECT and DOWNLOAD both point the session at a model and load it; the
   * engine fetches it first if it isn't cached. */
  const handleModelLoad = (url: string) => {
    if (status === "loading" || status === "downloading") return;
    setSettings({ modelUrl: url });
    pendingLoadUrlRef.current = url;
    pendingLoadKeyRef.current = settingsKey({ ...settings, modelUrl: url });
    setLoadingModelUrl(url);
    send({
      type: "load",
      options: loadOptions({ ...settings, modelUrl: url }),
    });
  };

  /** Append the user message and queue it to flush once the model is
   * ready. Shared by the auto-load and load-in-flight paths. */
  const queueUserMessage = (text: string) => {
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
  };

  const handleAsk = () => {
    const text = input.trim();
    if (!text) return;

    // A load is already in flight (e.g. started while typing) — queue the
    // message if it targets these settings; it flushes once ready.
    if (status === "loading" || status === "downloading") {
      if (pendingLoadKeyRef.current === settingsKey(settings)) {
        queueUserMessage(text);
      }
      return;
    }

    // Auto-load: with no model running (or one that doesn't match the
    // current settings), load first and send the message once ready.
    if (status === "idle" || (status === "ready" && modelMismatch)) {
      queueUserMessage(text);
      send({ type: "load", options: loadOptions(settings) });
      pendingLoadUrlRef.current = settings.modelUrl;
      pendingLoadKeyRef.current = settingsKey(settings);
      setLoadingModelUrl(settings.modelUrl);
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

  // Unified model rows: the suggested models plus any cached models that
  // aren't in the suggested list, ordered smallest → largest. Each row's
  // state picks its action: LOADED (click = unload), SELECT (cached), or
  // DOWNLOAD (remote; click = download and load).
  type ModelRow = {
    key: string;
    url: string;
    name: string;
    sizeBytes: number;
    state: "loaded" | "cached" | "remote";
  };
  const modelRows: ModelRow[] = (() => {
    const rows = new Map<string, ModelRow>();
    // The cache listing returns local paths, while suggested models are
    // keyed by URL — match them up by filename instead.
    const cachedByFilename = new Map(
      models.map((m) => [filenameOf(m.path), m]),
    );
    for (const m of SUGGESTED_MODELS) {
      const cached = cachedByFilename.get(filenameOf(m.url));
      if (cached) cachedByFilename.delete(filenameOf(m.url));
      rows.set(m.url, {
        key: m.url,
        url: m.url,
        name: m.name,
        sizeBytes: cached ? cached.size : parseSizeBytes(m.size),
        state:
          loadedModelUrl === m.url ? "loaded" : cached ? "cached" : "remote",
      });
    }
    // Cached models outside the suggested list (e.g. custom URLs).
    for (const m of cachedByFilename.values()) {
      rows.set(m.path, {
        key: m.path,
        url: m.path,
        name: filenameOf(m.path),
        sizeBytes: m.size,
        state: loadedModelUrl === m.path ? "loaded" : "cached",
      });
    }
    return [...rows.values()].sort((a, b) => a.sizeBytes - b.sizeBytes);
  })();

  const isBusy =
    status === "loading" || status === "downloading" || status === "generating";

  const modelMismatch =
    loadedKey !== null && loadedKey !== settingsKey(settings);

  const sortedConversations = [...conversations].sort(
    (a, b) => b.updatedAt - a.updatedAt,
  );
  // Sidebar shows only conversations the user has actually written in — a
  // fresh draft appears there once its first message is sent.
  const visibleConversations = sortedConversations.filter(
    (c) => !isEmptyConversation(c),
  );

  // Settings popover body — rendered in the sidebar footer (desktop) and
  // the compact header (mobile).
  const settingsBody = (
    <>
      <div className="settings-group">
        <div className="cached-models">
          <Eyebrow>Models</Eyebrow>
          {modelRows.map((row) => (
            <div key={row.key} className="cached-model">
              <span className="path">{row.name}</span>
              <span className="size">{formatBytes(row.sizeBytes)}</span>
              {row.state === "loaded" ? (
                <Button variant="outline" onClick={handleUnload}>
                  Loaded
                </Button>
              ) : loadingModelUrl === row.url ? (
                // In flight: download progress on the button, then a brief
                // “loading” while the engine initializes, then LOADED.
                <Button disabled>
                  {status === "downloading"
                    ? `${Math.round(progress * 100)}%`
                    : "loading…"}
                </Button>
              ) : (
                <Button
                  variant={row.state === "cached" ? "outline" : "solid"}
                  onClick={() => handleModelLoad(row.url)}
                  disabled={isBusy}
                >
                  {row.state === "cached" ? "Select" : "Download"}
                </Button>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="settings-group">
        <Eyebrow>Inference</Eyebrow>
        <div className="settings-grid">
          <label
            className="settings-check"
            title="WebGPU is disabled in this build"
          >
            <input
              type="checkbox"
              checked={settings.useGpu}
              disabled
              onChange={(e) => setSettings({ useGpu: e.target.checked })}
            />
            Use WebGPU
          </label>
          <label className="settings-check">
            <input
              type="checkbox"
              checked={settings.enableThinking}
              onChange={(e) =>
                setSettings({ enableThinking: e.target.checked })
              }
            />
            Enable thinking
          </label>
          <div className="nw-field" title="Coming soon">
            <label htmlFor="n-threads">Threads</label>
            <input
              id="n-threads"
              type="number"
              min={1}
              placeholder="Auto"
              value={settings.nThreads ?? ""}
              disabled
              onChange={(e) => {
                const v = e.target.value;
                setSettings({
                  nThreads:
                    v === "" ? undefined : Math.max(1, Math.floor(Number(v))),
                });
              }}
            />
          </div>
          <label className="settings-check">
            <input
              type="checkbox"
              checked={useWorker}
              onChange={(e) => setUseWorker(e.target.checked)}
            />
            Run inference in Web Worker
          </label>
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
          <Wordmark size={20} href="https://nobodywho.ai" />
          <a
            className="brand-source"
            href="https://github.com/nobodywho-ooo/javascript-starter-example"
            target="_blank"
            rel="noopener"
          >
            Source
          </a>
        </div>
        <div className="conv-list">
          {visibleConversations.map((c) => {
            // A brand-new conversation has no title yet — show it without
            // one until the user has written something.
            const untitled =
              c.messages.length === 0 && c.title === "New conversation";
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
                <span className="meta">{c.tokens ?? 0} tokens</span>
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
          <div
            role="button"
            tabIndex={0}
            className={`conv-item conv-item-new${
              isEmptyConversation(active) ? " is-draft" : ""
            }`}
            onClick={createConversation}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                createConversation();
              }
            }}
          >
            + New conversation
          </div>
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
            {active.messages.map((m) => (
              <div key={m.id} className={`msg msg-${m.role}`}>
                <div className="bubble">{m.content || "…"}</div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-area">
            <input
              ref={inputRef}
              className="chat-input"
              type="text"
              placeholder="Type a message"
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
          </div>
        </main>
      </div>
    </div>
  );
}
