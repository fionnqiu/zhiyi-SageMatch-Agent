import { useEffect, useState } from "react";
import { subscribeToasts, type ToastItem } from "../../lib/feedback/notify";

const EXIT_MS = 180;

type Phase = "in" | "out";

/** 顶部居中的气泡条，挂在应用根上，不跟某个页面走。 */
export function ToastHost() {
  const [items, setItems] = useState<ToastItem[]>([]);
  const [phase, setPhase] = useState<Record<number, Phase>>({});

  useEffect(() => {
    return subscribeToasts((next) => {
      setItems((current) => {
        const incoming = next.filter((item) => !current.some((row) => row.id === item.id));
        if (incoming.length) {
          // 先以退场状态挂上，下一帧再切到进入，过渡才有起点。
          setPhase((prev) => {
            const added = { ...prev };
            incoming.forEach((item) => {
              added[item.id] = "out";
            });
            return added;
          });
          window.requestAnimationFrame(() => {
            setPhase((prev) => {
              const entered = { ...prev };
              incoming.forEach((item) => {
                entered[item.id] = "in";
              });
              return entered;
            });
          });
        }
        const leaving = current.filter((item) => !next.some((row) => row.id === item.id));
        leaving.forEach((item) => {
          setPhase((prev) => ({ ...prev, [item.id]: "out" }));
          window.setTimeout(() => {
            setItems((rows) => rows.filter((row) => row.id !== item.id));
            setPhase((prev) => {
              const rest = { ...prev };
              delete rest[item.id];
              return rest;
            });
          }, EXIT_MS);
        });
        const staying = current.filter((item) => next.some((row) => row.id === item.id));
        return [...staying, ...incoming];
      });
    });
  }, []);

  if (!items.length) return null;

  return (
    <div className="toast-host" aria-live="polite">
      {items.map((item) => (
        <div key={item.id} data-phase={phase[item.id] ?? "out"} className={`toast-bubble toast-${item.tone}`} role="status">
          {item.text}
        </div>
      ))}
    </div>
  );
}
