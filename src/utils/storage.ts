// Local storage utilities for the chat application.

export type StoredMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
};

export type ConversationSettings = {
  modelUrl: string;
  useGpu: boolean;
  threadCount: number; // 0 = auto
  systemPrompt: string;
};

export type Conversation = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: StoredMessage[];
  settings: ConversationSettings;
};

const CONVERSATIONS_KEY = "nobodywho-conversations";
const ACTIVE_KEY = "nobodywho-active-conversation";
const SETTINGS_KEY = "nobodywho-settings";

/** App-level settings (engine-level, not per conversation). */
export type Settings = {
  useWorkerThread: boolean;
};

export const defaultSettings: Settings = {
  useWorkerThread: true,
};

function isConversation(value: unknown): value is Conversation {
  if (typeof value !== "object" || value === null) return false;
  const c = value as Conversation;
  return (
    typeof c.id === "string" &&
    typeof c.title === "string" &&
    Array.isArray(c.messages) &&
    typeof c.settings?.modelUrl === "string"
  );
}

export function loadConversations(): Conversation[] {
  try {
    const data = localStorage.getItem(CONVERSATIONS_KEY);
    if (!data) return [];
    const parsed = JSON.parse(data);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isConversation);
  } catch (e) {
    console.warn("Failed to load conversations from localStorage:", e);
    return [];
  }
}

export function saveConversations(conversations: Conversation[]): void {
  try {
    localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(conversations));
  } catch (e) {
    console.warn("Failed to save conversations to localStorage:", e);
  }
}

export function loadActiveConversationId(): string | null {
  try {
    return localStorage.getItem(ACTIVE_KEY);
  } catch {
    return null;
  }
}

export function saveActiveConversationId(id: string): void {
  try {
    localStorage.setItem(ACTIVE_KEY, id);
  } catch {
    // Non-fatal.
  }
}

export function loadSettings(): Settings {
  try {
    const data = localStorage.getItem(SETTINGS_KEY);
    if (!data) return { ...defaultSettings };
    return { ...defaultSettings, ...JSON.parse(data) };
  } catch (e) {
    console.warn("Failed to load settings from localStorage:", e);
    return { ...defaultSettings };
  }
}

export function saveSettings(settings: Settings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (e) {
    console.warn("Failed to save settings to localStorage:", e);
  }
}
