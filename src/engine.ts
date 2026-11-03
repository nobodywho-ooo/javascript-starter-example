/**
 * Engine — the core logic for the chat app.
 *
 * This file is intentionally UI-agnostic. It can run on the main thread
 * (via `agent.ts`) or inside a Web Worker (via `worker.ts`).
 */

import {
  Model,
  Chat,
  Tool,
  downloadModel,
  getCachedModels,
} from "react-native-nobodywho";

// --- Event-loop yielding ---

// Yield a *macrotask* between tokens. This is load-bearing for command
// processing: in the WASM build the Rust futures and their wakers run on
// microtasks (wasm-bindgen-futures), so a `for await` token loop with no
// macrotask boundary becomes a microtask chain that starves `onmessage` —
// e.g. a "stop" command would never run until generation finished.
//
// setTimeout(0) would also work, but browsers clamp nested timeouts to
// ~4ms, throttling generation to ~250 tokens/s. A MessageChannel message is
// a macrotask too, but fires immediately — same guarantee, no clamp. Falls
// back to setTimeout where MessageChannel doesn't exist (e.g. Hermes).
const yieldChannel: MessageChannel | null =
  typeof MessageChannel === "function" ? new MessageChannel() : null;
const yieldWaiters: Array<() => void> = [];
if (yieldChannel) {
  // Message order is FIFO, so waiters resolve in the order they queued.
  yieldChannel.port1.onmessage = () => yieldWaiters.shift()?.();
}
function yieldMacrotask(): Promise<void> {
  if (!yieldChannel) {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }
  return new Promise((resolve) => {
    yieldWaiters.push(resolve);
    yieldChannel.port2.postMessage(null);
  });
}

// --- Types shared with the UI ---

export type CachedModel = {
  path: string;
  size: number; // bytes (converted from bigint)
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export type EngineStatus =
  | "idle" // No model loaded
  | "downloading" // Fetching model file
  | "loading" // Loading model into memory
  | "ready" // Ready to chat
  | "generating"; // Currently streaming tokens

export type LogLevel = "info" | "warn" | "error";

export interface EngineEvents {
  onLog(message: string, level?: LogLevel): void;
  onStatus(status: EngineStatus, message?: string): void;
  onProgress(fraction: number): void;
  onToken(token: string): void;
  /** Context usage of the session, reported after each generation. */
  onTokens(count: number): void;
  onError(message: string): void;
  onModels(models: CachedModel[]): void;
}

export type LoadOptions = {  url: string;
  useGpu: boolean;
  threadCount: number; // 0 = auto
  systemPrompt: string;
  /** Give the model a web-fetch tool (browser fetch — subject to CORS). */
  enableWebTools?: boolean;
};

export type HistoryMessage = { role: "user" | "assistant"; content: string };

/**
 * Tools for the chat session. The `fetch_url` tool lets the model make HTTP
 * GET requests from the browser — subject to the same CORS rules as any
 * page script, so only CORS-enabled endpoints will work.
 */
function makeWebTools(): Tool[] {
  return [
    new Tool({
      name: "fetch_url",
      description:
        "Fetch a URL with an HTTP GET request and return the response body " +
        "as text. Use for APIs and web pages. Only URLs that allow " +
        "cross-origin requests (CORS) will work.",
      parameters: [{ name: "url", type: "string" }],
      call: async (url: string) => {
        try {
          const res = await fetch(url, {
            headers: { Accept: "text/plain, application/json, */*" },
          });
          if (!res.ok) {
            return `HTTP error ${res.status} ${res.statusText}`;
          }
          const text = await res.text();
          // Keep the context window small — models ramble on big pages.
          return text.length > 4000 ? text.slice(0, 4000) + "…" : text;
        } catch (e) {
          return `Request failed: ${e}`;
        }
      },
    }),
  ];
}

export type EngineOptions = {
  /** Yield a macrotask between tokens so the host event loop stays
   * responsive — the browser can paint on the main thread, and a worker can
   * process incoming commands (e.g. "stop") mid-generation. */
  yieldBetweenTokens?: boolean;
};

type DownloadHandle = {
  url: string;
  cancelled: boolean;
  report: ((fraction: number) => void) | null;
  /** Resolves with the cached path; rejects on failure. */
  promise: Promise<string>;
  cancel: () => void;
};

export class Engine {
  private events: EngineEvents;
  private model: Model | null = null;
  private chat: Chat | null = null;
  private yieldBetweenTokens: boolean;
  /** True while a generation task is using the model context. */
  private generating = false;
  /** Options of the most recent load, so ask() can re-load after an unload
   * or failed load instead of dropping the message. */
  private lastLoadOptions: LoadOptions | null = null;
  /** Dedupes concurrent identical load calls (e.g. a prefetch race). */
  private activeLoad: { key: string; promise: Promise<void> } | null = null;
  /** Set by stop(); checked in the token loop so tokens already buffered in
   * the Rust channel are dropped instead of streamed out after a stop. */
  private stopRequested = false;
  private activeDownload: DownloadHandle | null = null;

  constructor(events: EngineEvents, options: EngineOptions = {}) {
    this.events = events;
    this.yieldBetweenTokens = options.yieldBetweenTokens ?? true;
  }

  // --- Commands ---

  async listModels(): Promise<void> {
    try {
      const models: CachedModel[] = getCachedModels().map((m) => ({
        path: m.path,
        size: Number(m.size),
      }));
      this.events.onModels(models);
    } catch (e) {
      this.events.onError(`Failed to list cached models: ${e}`);
    }
  }

  /**
   * Start (or attach to) a model download. Concurrent requests for the same
   * URL share one download — a second fetch would download the model twice
   * and corrupt the shared partial `.part` file. A request for a different
   * URL is rejected while one is in flight.
   *
   * The returned promise resolves with the cached path, or with "" if the
   * download was stopped (the underlying fetch keeps going in the
   * background — the library exposes no abort — but no further progress is
   * reported and the status has already been reset by `stop()`).
   */
  private startDownload(
    url: string,
    report: ((fraction: number) => void) | null,
  ): Promise<string> {
    const active = this.activeDownload;
    if (active && active.url !== url) {
      throw new Error(
        "Another download is already in progress — stop it first.",
      );
    }
    if (active) {
      // Same URL: share the in-flight download, re-attach reporting.
      active.cancelled = false;
      active.report = report;
      return active.promise;
    }
    let cancelResolve: (path: string) => void = () => {};
    const cancelledPromise = new Promise<string>((resolve) => {
      cancelResolve = () => resolve("");
    });
    const handle: DownloadHandle = {
      url,
      cancelled: false,
      report,
      promise: Promise.resolve("") as Promise<string>,
      cancel: () => {
        handle.cancelled = true;
        handle.report = null;
        cancelResolve("");
      },
    };
    handle.promise = (async () => {
      try {
        const path = await downloadModel({
          modelPath: url,
          onDownloadProgress: (downloaded, total) => {
            if (handle.cancelled) return;
            handle.report?.(
              total > 0 ? Number(downloaded) / Number(total) : 0,
            );
          },
        });
        if (handle.cancelled) {
          this.events.onLog("Background download finished.");
        }
        return path;
      } catch (e) {
        if (handle.cancelled) {
          this.events.onLog(`Background download failed: ${e}`, "warn");
        }
        throw e;
      } finally {
        if (this.activeDownload === handle) this.activeDownload = null;
      }
    })();
    this.activeDownload = handle;
    return Promise.race([handle.promise, cancelledPromise]);
  }

  /**
   * Load a model into memory and initialize a chat session.
   * If the model isn't cached, this will download it first.
   */
  async load(opts: LoadOptions): Promise<void> {
    this.lastLoadOptions = opts;
    // Same settings already loading? Piggyback on the in-flight load instead
    // of starting a second one (which would leak the first model).
    const key = JSON.stringify(opts);
    if (this.activeLoad?.key === key) return this.activeLoad.promise;
    const promise = this.doLoad(opts).finally(() => {
      if (this.activeLoad?.promise === promise) this.activeLoad = null;
    });
    this.activeLoad = { key, promise };
    return promise;
  }

  private async doLoad(opts: LoadOptions): Promise<void> {
    this.events.onStatus("loading");
    let reportedDownload = false;
    try {
      this.events.onLog(`Resolving ${opts.url}...`);
      // Step 1: Explicit download step (returns local cached path). The
      // progress callback flips the status to "downloading" on the first
      // byte, so an uncached model shows real progress instead of sitting
      // on a silent "loading". A cached model emits nothing and stays on
      // "loading".
      const path = await this.startDownload(opts.url, (fraction) => {
        if (!reportedDownload) {
          reportedDownload = true;
          this.events.onStatus("downloading");
        }
        this.events.onProgress(fraction);
      });
      if (!path) {
        // Stopped during the download — status already reset by stop().
        this.events.onLog("Load cancelled during download.", "warn");
        return;
      }
      this.events.onLog(`Model is at ${path}`);
      this.events.onProgress(0);
      this.events.onStatus("loading");

      this.events.onLog(`Loading model (GPU: ${opts.useGpu})...`);
      // Step 2: Load the model.
      this.model = await Model.load({
        modelPath: path,
        useGpu: opts.useGpu,
      });

      this.events.onLog(
        `Model loaded. Context window: ${this.model.maxCtx} tokens.`,
      );

      // Step 3: Create the chat session.
      this.chat = new Chat({
        model: this.model,
        systemPrompt: opts.systemPrompt,
        // 0 means "auto" (omit the parameter).
        threadCount: opts.threadCount > 0 ? opts.threadCount : undefined,
        tools: opts.enableWebTools ? makeWebTools() : undefined,
      });

      this.events.onStatus("ready", `Max context: ${this.model.maxCtx}`);
    } catch (e) {
      this.events.onError(`Failed to load model: ${e}`);
      this.events.onProgress(0);
      this.events.onStatus("idle");
    } finally {
      // A load may have fetched the model into the cache — refresh the
      // cached-models list either way, so the UI reflects the current
      // cache state without waiting for a reload.
      this.listModels();
    }
  }

  async ask(text: string): Promise<void> {
    // A message can arrive with no model loaded — e.g. a queued message
    // flushed after an unload won a race with the load meant for it.
    // Reload the last-used configuration so the message isn't lost.
    if (!this.chat && this.lastLoadOptions) {
      // A message can arrive with no model loaded — e.g. a queued message
      // flushed after an unload won a race with the load meant for it.
      // Reload the last-used configuration so the message isn't lost.
      this.events.onLog("No model loaded — loading before answering.", "info");
      await this.load(this.lastLoadOptions);
    }
    if (!this.chat) return;
    this.events.onStatus("generating");
    this.stopRequested = false;
    this.generating = true;
    try {
      const stream = await this.chat.ask(text);
      for await (const token of stream) {
        if (this.stopRequested) break;
        this.events.onToken(token);
        if (this.yieldBetweenTokens) {
          // Macrotask boundary between tokens: lets the worker process
          // incoming commands (e.g. "stop") and lets the main thread paint,
          // without the ~4ms setTimeout clamp.
          await yieldMacrotask();
        }
      }
      this.events.onLog("Generation finished.");
    } catch (e) {
      this.events.onError(`Generation failed: ${e}`);
    } finally {
      this.generating = false;
      this.events.onStatus("ready");
      // Report how many tokens the session now occupies, so the UI can
      // track the conversation's context usage. Safe only after generation
      // has wound down — mid-generation calls block the worker.
      if (this.chat) {
        try {
          const stats = await this.chat.getStats();
          this.events.onTokens(stats.contextUsed);
        } catch {
          // Stats unavailable (e.g. chat torn down mid-flight) — skip.
        }
      }
    }
  }

  stop(): void {
    const download = this.activeDownload;
    if (download && !download.cancelled) {
      // Stop a running download: reset the status now; the fetch itself
      // keeps going in the background (no abort API) but is no longer
      // reported, and finishing it still fills the cache.
      download.cancel();
      this.events.onLog("Download stopped — finishing in background.", "warn");
      this.events.onProgress(0);
      this.events.onStatus("idle");
      return;
    }
    this.chat?.stopGeneration();
    this.stopRequested = true;
  }

  async resetHistory(): Promise<void> {
    if (this.chat && this.generating) {
      // Resetting while generating hangs until the response completes: the
      // Rust-side reset flushes the in-flight token generator, which drains
      // it to completion (core/src/chat.rs flush()). Stop the generation
      // first, so the flush only winds down an already-stopping generator.
      this.chat.stopGeneration();
      this.stopRequested = true;
    }
    await this.chat?.resetHistory();
    this.events.onLog("Chat history reset.");
  }

  /** Replace the session's chat history (e.g. when switching conversations
   * or after loading a model mid-conversation). */
  async setHistory(messages: HistoryMessage[]): Promise<void> {
    if (!this.chat) return;
    if (this.generating) {
      // Same flush-to-completion hazard as resetHistory — stop first.
      this.chat.stopGeneration();
      this.stopRequested = true;
    }
    await this.chat.setChatHistory(
      messages.map((m) =>
        m.role === "user"
          ? { role: "user" as const, content: m.content }
          : { role: "assistant" as const, content: m.content },
      ),
    );
    this.events.onLog(`Chat history loaded (${messages.length} messages).`);
  }

  unload(): void {
    if (this.generating) {
      // The generation task is still using the model context — destroying
      // it now would be a use-after-free. Stop generation; the caller can
      // unload again once it has wound down.
      this.chat?.stopGeneration();
      this.events.onLog(
        "Generation stopped — call unload again to free the model.",
        "warn",
      );
      return;
    }
    // Destroy the chat first: it owns the context/KV-cache and holds an Arc
    // to the model, so the weights are only freed once it is gone. The
    // explicit destroy matters on WASM — without it the Rust objects are
    // freed only when the JS GC happens to run their finalizers, which can
    // be long after unload (FinalizationRegistry timing is not
    // memory-pressure aware).
    this.chat?.destroy();
    this.chat = null;
    this.model?.destroy();
    this.model = null;
    this.events.onStatus("idle");
    this.events.onLog("Model unloaded.");
  }
}
