// Local storage utilities for the chat application.

export type StoredMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
};

const HISTORY_KEY = 'nobodywho-chat-history';
const SETTINGS_KEY = 'nobodywho-settings';

export type Settings = {
  useGpu: boolean;
  threadCount: number;
  useWorkerThread: boolean;
  systemPrompt: string;
};

export const defaultSettings: Settings = {
  useGpu: false,
  threadCount: 4,
  useWorkerThread: true,
  systemPrompt: 'You are a helpful assistant.',
};

export function loadHistory(): StoredMessage[] {
  try {
    const data = localStorage.getItem(HISTORY_KEY);
    if (!data) return [];
    return JSON.parse(data);
  } catch (e) {
    console.warn('Failed to load history from localStorage:', e);
    return [];
  }
}

export function saveHistory(messages: StoredMessage[]): void {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(messages));
  } catch (e) {
    console.warn('Failed to save history to localStorage:', e);
  }
}

export function clearHistory(): void {
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch (e) {
    console.warn('Failed to clear history from localStorage:', e);
  }
}

export function loadSettings(): Settings {
  try {
    const data = localStorage.getItem(SETTINGS_KEY);
    if (!data) return { ...defaultSettings };
    return { ...defaultSettings, ...JSON.parse(data) };
  } catch (e) {
    console.warn('Failed to load settings from localStorage:', e);
    return { ...defaultSettings };
  }
}

export function saveSettings(settings: Settings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (e) {
    console.warn('Failed to save settings to localStorage:', e);
  }
}
