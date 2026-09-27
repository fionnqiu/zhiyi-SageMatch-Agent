/** 全局气泡通知。页面只调用 notify，不各自画错误条。 */

export type ToastTone = "info" | "ok" | "error";

export type ToastItem = {
  id: number;
  text: string;
  tone: ToastTone;
};

const VISIBLE_MS = 3000;
const EXIT_MS = 180;

let seq = 0;
let items: ToastItem[] = [];
const listeners = new Set<(next: ToastItem[]) => void>();

function emit() {
  const snapshot = items;
  listeners.forEach((fn) => fn(snapshot));
}

export function notify(text: string, tone: ToastTone = "info") {
  const message = text.trim();
  if (!message) return;
  const id = ++seq;
  items = [...items, { id, text: message, tone }];
  emit();
  // 先停留，再给退场留 180ms，最后才从列表拿掉。
  window.setTimeout(() => {
    items = items.filter((item) => item.id !== id);
    emit();
  }, VISIBLE_MS + EXIT_MS);
}

export function subscribeToasts(fn: (next: ToastItem[]) => void) {
  listeners.add(fn);
  fn(items);
  return () => {
    listeners.delete(fn);
  };
}
