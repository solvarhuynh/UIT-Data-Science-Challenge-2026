import { useCallback, useEffect, useState } from "react";

export function readStorage<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export function writeStorage<T>(key: string, value: T) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* quota / private mode */
  }
}

/** SSR-safe persisted state: renders the fallback on the server, hydrates after mount. */
export function usePersistentState<T>(key: string, fallback: T) {
  const [value, setValue] = useState<T>(fallback);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setValue(readStorage<T>(key, fallback));
    setHydrated(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  useEffect(() => {
    if (hydrated) writeStorage(key, value);
  }, [key, value, hydrated]);

  return [value, setValue, hydrated] as const;
}

export function useTheme() {
  const [theme, setTheme, hydrated] = usePersistentState<"light" | "dark">(
    "shipcode.theme",
    "light",
  );

  useEffect(() => {
    if (!hydrated) return;
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.style.colorScheme = theme;
  }, [theme, hydrated]);

  const toggle = useCallback(() => setTheme((t) => (t === "dark" ? "light" : "dark")), [setTheme]);

  return { theme, setTheme, toggle };
}
