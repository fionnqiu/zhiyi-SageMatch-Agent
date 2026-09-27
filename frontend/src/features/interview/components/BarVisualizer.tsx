import { useEffect, useState } from "react";

export type BarVisualizerState = "connecting" | "initializing" | "listening" | "speaking" | "thinking" | "idle";

type BarVisualizerProps = {
  state: BarVisualizerState;
  barCount?: number;
  minHeight?: number;
  maxHeight?: number;
  className?: string;
};

const BASE_LEVELS = [28, 44, 62, 78, 54, 86, 68, 96, 70, 88, 58, 76, 48, 64, 36];

/**
 * Renders the compact frequency-bar treatment used below the interview voice state.
 * The animation is state-driven because browser speech recognition does not expose a
 * MediaStream here; this keeps the visual feedback honest while matching the voice flow.
 */
export function BarVisualizer({ state, barCount = 15, minHeight = 20, maxHeight = 100, className = "" }: BarVisualizerProps) {
  const [frame, setFrame] = useState(0);
  const animated = state === "listening" || state === "speaking" || state === "thinking";

  useEffect(() => {
    if (!animated) return;
    const timer = window.setInterval(() => setFrame((value) => value + 1), state === "thinking" ? 180 : 110);
    return () => window.clearInterval(timer);
  }, [animated, state]);

  const levels = Array.from({ length: barCount }, (_, index) => {
    const base = BASE_LEVELS[index % BASE_LEVELS.length];
    const wave = animated ? Math.sin((frame * 0.9) + index * 0.85) * (state === "thinking" ? 12 : 30) : 0;
    return Math.max(minHeight, Math.min(maxHeight, base + wave));
  });

  return (
    <div className={`bar-visualizer bar-visualizer--${state} ${className}`} aria-hidden="true">
      {levels.map((level, index) => (
        <span
          key={index}
          className="bar-visualizer__bar"
          style={{ height: `${level}%`, animationDelay: `${index * 28}ms` }}
        />
      ))}
    </div>
  );
}
