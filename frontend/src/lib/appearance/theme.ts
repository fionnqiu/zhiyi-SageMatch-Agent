export type ThemeMode = "dark" | "light";

const STORAGE_KEY = "sagematch-theme";

/**
 * 读上次选择：只认显式存下的 "dark"，其余一切（没有记录、值损坏、存储不可读）都回到浅色。
 * 浅色是站点默认外观，因此不按白名单判断 light，而是把 dark 当作唯一的「非默认」选项——
 * 这样写入 localStorage 之外的任何脏值都不会意外把新访客带回深色。
 */
export function readTheme(): ThemeMode {
  try {
    return localStorage.getItem(STORAGE_KEY) === "dark" ? "dark" : "light";
  } catch {
    return "light";
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
