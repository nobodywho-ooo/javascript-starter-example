/// <reference lib="webworker" />
/**
 * Worker — runs the Engine in a dedicated Web Worker.
 *
 * This keeps the main thread responsive while the model is loading or
 * generating tokens.
 */

import type { AgentCommand } from "./agent";
import { Engine } from "./engine";

// Keep yieldBetweenTokens (default): the token loop must leave a macrotask
// boundary for `stop` commands to be processed. engine.ts yields via
// MessageChannel — no setTimeout's ~4ms clamp, so it costs ~nothing per
// token.
const engine = new Engine({
  onLog: (message, level = "info") =>
    postMessage({ type: "log", payload: { message, level } }),
  onStatus: (status, message) =>
    postMessage({ type: "status", payload: { status, message } }),
  onProgress: (fraction) => postMessage({ type: "progress", payload: fraction }),
  onToken: (token) => postMessage({ type: "token", payload: token }),
  onTokens: (count) => postMessage({ type: "tokens", payload: count }),
  onError: (message) => postMessage({ type: "error", payload: message }),
  onModels: (models) => postMessage({ type: "models", payload: models }),
});

// The command is the whole message payload — same shape the agent sends.
self.onmessage = async (event: MessageEvent<AgentCommand>) => {
  const command = event.data;
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
      case "unload":
        engine.unload();
        break;
    }
  } catch (e) {
    postMessage({ type: "error", payload: String(e) });
  }
};
