/**
 * Agent — abstracts the execution context of the Engine.
 *
 * The UI talks to an "Agent", which either:
 * 1. Spawns a Web Worker (`worker.ts`) to run the Engine off the main thread.
 * 2. Imports the Engine directly (`engine.ts`) to run it on the main thread.
 *
 * Both modes expose the exact same command/event interface, so the UI
 * doesn't need to know which one it's talking to.
 */

import type {
  EngineEvents,
  HistoryMessage,
  LoadOptions,
  LogLevel,
} from "./engine";

export type AgentCommand =
  | { type: "list-models" }
  | { type: "load"; options: LoadOptions }
  | { type: "ask"; text: string }
  | { type: "stop" }
  | { type: "reset" }
  | { type: "set-history"; messages: HistoryMessage[] }
  | { type: "unload" };

export interface Agent {
  send(command: AgentCommand): void;
  dispose?(): void;
}

// Event types posted from the worker to the main thread.
type WorkerEvent =
  | { type: "log"; payload: { message: string; level: LogLevel } }
  | { type: "status"; payload: { status: any; message?: string } }
  | { type: "progress"; payload: number }
  | { type: "token"; payload: string }
  | { type: "tokens"; payload: number }
  | { type: "error"; payload: string }
  | { type: "models"; payload: any[] };

function dispatch(events: EngineEvents, event: WorkerEvent): void {
  switch (event.type) {
    case "log":
      events.onLog(event.payload.message, event.payload.level);
      break;
    case "status":
      events.onStatus(event.payload.status, event.payload.message);
      break;
    case "progress":
      events.onProgress(event.payload);
      break;
    case "token":
      events.onToken(event.payload);
      break;
    case "tokens":
      events.onTokens(event.payload);
      break;
    case "error":
      events.onError(event.payload);
      break;
    case "models":
      events.onModels(event.payload);
      break;
  }
}

export async function createAgent(
  useWorker: boolean,
  events: EngineEvents,
): Promise<Agent> {
  if (useWorker) {
    // --- Worker Mode ---
    const worker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (event: MessageEvent<WorkerEvent>) => {
      dispatch(events, event.data);
    };
    // Surface worker-level failures (e.g. a crash while loading the WASM)
    // instead of failing silently.
    worker.onerror = (event) => {
      events.onError(`Worker error: ${event.message ?? "unknown"}`);
    };
    worker.onmessageerror = () => {
      events.onError("Worker message serialization failed.");
    };
    return {
      send: (command) => worker.postMessage(command),
      dispose: () => worker.terminate(),
    };
  } else {
    // --- Direct Mode (Main Thread) ---
    // Dynamically imported so we don't load the WASM on the main thread
    // unless we actually need it.
    const { Engine } = await import("./engine");
    // The engine yields a macrotask between tokens by default, so the UI
    // can stream and stay responsive on the main thread.
    const engine = new Engine(events);
    // Serialize command execution: direct mode doesn't have the worker's
    // natural ordering, and e.g. set-history must land before a follow-up
    // ask, not interleaved with it.
    let queue: Promise<void> = Promise.resolve();
    return {
      send: (command) => {
        queue = queue
          .then(async () => {
            try {
              switch (command.type) {
                case "list-models":
                  await engine.listModels();
                  break;
                case "load":
                  await engine.load(command.options);
                  break;
                case "ask":
                  await engine.ask(command.text);
                  break;
                case "stop":
                  engine.stop();
                  break;
                case "reset":
                  await engine.resetHistory();
                  break;
                case "set-history":
                  await engine.setHistory(command.messages);
                  break;
                case "unload":
                  engine.unload();
                  break;
              }
            } catch (e) {
              events.onError(String(e));
            }
          })
          // Keep the chain alive if a command throws.
          .catch(() => undefined);
      },
    };
  }
}
