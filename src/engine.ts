/**
 * Engine — the core logic for the chat app.
 *
 * This file is intentionally UI-agnostic. It can run on the main thread
 * (via `agent.ts`) or inside a Web Worker (via `worker.ts`).
 */

import {
  Model,
  Chat,
  downloadModel,
  getCachedModels,
} from "react-native-nobodywho";

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
  onError(message: string): void;
  onModels(models: CachedModel[]): void;
}

export type LoadOptions = {
  url: string;
  useGpu: boolean;
  threadCount: number; // 0 = auto
  systemPrompt: string;
};

export type EngineOptions = {
  /** Yield a macrotask between tokens so the host event loop stays
   * responsive — the browser can paint on the main thread, and a worker can
   * process incoming commands (e.g. "stop") mid-generation. */
  yieldBetweenTokens?: boolean;
};

export class Engine {
  private events: EngineEvents;
  private model: Model | null = null;
  private chat: Chat | null = null;
  private yieldBetweenTokens: boolean;

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
   * Explicitly download a model file to the local cache.
   * Reports progress via `onProgress`.
   */
  async download(url: string): Promise<void> {
    this.events.onLog(`Downloading ${url}...`);
    this.events.onStatus("downloading");
    try {
      await downloadModel({
        modelPath: url,
        onDownloadProgress: (downloaded, total) => {
          if (total > 0n) {
            this.events.onProgress(Number(downloaded) / Number(total));
          }
        },
      });
      this.events.onLog("Download finished.");
    } catch (e) {
      this.events.onError(`Download failed: ${e}`);
    } finally {
      this.events.onProgress(0);
      this.events.onStatus("idle");
      this.listModels();
    }
  }

  /**
   * Load a model into memory and initialize a chat session.
   * If the model isn't cached, this will download it first.
   */
  async load(opts: LoadOptions): Promise<void> {
    this.events.onStatus("loading");
    let reportedDownload = false;
    try {
      this.events.onLog(`Resolving ${opts.url}...`);
      // Step 1: Explicit download step (returns local cached path). The
      // progress callback flips the status to "downloading" on the first
      // byte, so an uncached model shows real progress instead of sitting
      // on a silent "loading". A cached model emits nothing and stays on
      // "loading".
      const path = await downloadModel({
        modelPath: opts.url,
        onDownloadProgress: (downloaded, total) => {
          if (!reportedDownload) {
            reportedDownload = true;
            this.events.onStatus("downloading");
          }
          if (total > 0n) {
            this.events.onProgress(Number(downloaded) / Number(total));
          }
        },
      });
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
      });

      this.events.onStatus("ready", `Max context: ${this.model.maxCtx}`);
    } catch (e) {
      this.events.onError(`Failed to load model: ${e}`);
      this.events.onProgress(0);
      this.events.onStatus("idle");
    }
  }

  async ask(text: string): Promise<void> {
    if (!this.chat) {
      this.events.onLog("No model loaded — load a model before asking.", "warn");
      return;
    }
    this.events.onStatus("generating");
    try {
      const stream = await this.chat.ask(text);
      for await (const token of stream) {
        this.events.onToken(token);
        if (this.yieldBetweenTokens) {
          // Give the event loop a macrotask so the browser can paint and
          // React can flush renders between tokens on the main thread.
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
      }
      this.events.onLog("Generation finished.");
    } catch (e) {
      this.events.onError(`Generation failed: ${e}`);
    } finally {
      this.events.onStatus("ready");
    }
  }

  stop(): void {
    this.chat?.stopGeneration();
  }

  async resetHistory(): Promise<void> {
    await this.chat?.resetHistory();
    this.events.onLog("Chat history reset.");
  }

  unload(): void {
    this.chat = null;
    this.model?.destroy();
    this.model = null;
    this.events.onStatus("idle");
    this.events.onLog("Model unloaded.");
  }
}
