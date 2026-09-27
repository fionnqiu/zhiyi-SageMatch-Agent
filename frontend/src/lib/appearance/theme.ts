export type ThemeMode = "dark" | "light";

const STORAGE_KEY = "sagematch-theme";

/** 读上次选择。没有记录或值损坏时回到深色，避免第一次打开就改掉现有外观。 */
export function readTheme(): ThemeMode {
  try {
    return localStorage.getItem(STORAGE_KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

/** 主题挂在 html 上，让 Tailwind 的 --color-* 和原生控件的 color-scheme 一起换。 */
export function applyTheme(mode: ThemeMode) {
  document.documentElement.dataset.theme = mode;
  document.documentElement.style.colorScheme = mode;
  try {
    localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    // 隐私模式写不进存储时，本次会话仍然切过去。
  }
}
