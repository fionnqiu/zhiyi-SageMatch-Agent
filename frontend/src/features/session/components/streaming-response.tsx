"use client";
// beui.dev/components/agents/streaming-response
// 只保留回答表面：正文稳定，完成后操作从下方浮出。来源折叠和评价不接，当前回答没有这两项数据。

import { Check, Copy, RotateCcw } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { EASE_OUT, SPRING_PRESS } from "../../../lib/appearance/ease";
import { cn } from "../../../lib/utils";

export type StreamingResponseStatus = "streaming" | "complete" | "error";

export interface StreamingResponseProps {
  /** 已经渲染好的回答。Markdown 在外面转成元素后再传进来。 */
  children: ReactNode;
  status?: StreamingResponseStatus;
  /** 复制按钮写入剪贴板的原文。 */
  copyText?: string;
  /** 覆盖内置复制。没有 copyText 时也能自己处理。 */
  onCopy?: () => void | Promise<void>;
  onRetry?: () => void;
  /** 外层会话列表已经播报文字时关掉，避免读屏重复念。 */
  announce?: boolean;
  /** 只藏操作，不改回答状态。 */
  showActions?: boolean;
  className?: string;
  contentClassName?: string;
  actionsClassName?: string;
}

function ResponseAction({
  label,
  onClick,
  children,
}: {
  label: string;
  onClick: () => void;
  children: ReactNode;
}) {
  const reduce = useReducedMotion() ?? false;

  return (
    <motion.button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      whileTap={reduce ? undefined : { scale: 0.9 }}
      transition={SPRING_PRESS}
      className="grid size-7 place-items-center rounded-md text-dim outline-none transition-colors hover:bg-field hover:text-ink focus-visible:ring-2 focus-visible:ring-forest-2"
    >
      {children}
    </motion.button>
  );
}

export function StreamingResponse({
  children,
  status = "streaming",
  copyText,
  onCopy,
  onRetry,
  announce = true,
  showActions = true,
  className,
  contentClassName,
  actionsClassName,
}: StreamingResponseProps) {
  const reduce = useReducedMotion() ?? false;
  const [copied, setCopied] = useState(false);
  const copyTimer = useRef<number | undefined>(undefined);
  const streaming = status === "streaming";
  const [actionsReady, setActionsReady] = useState(status !== "streaming");
  const canCopy = Boolean(copyText || onCopy);
  // 字播完后留 450ms，和演示里把 status 切到 complete 的停顿一致。减少动态效果时不等。
  useEffect(() => {
    if (streaming) {
      setActionsReady(false);
      return;
    }
    const timer = window.setTimeout(() => setActionsReady(true), reduce ? 0 : 450);
    return () => window.clearTimeout(timer);
  }, [reduce, streaming]);
  const shouldShowActions = showActions && actionsReady && (canCopy || Boolean(onRetry));

  useEffect(
    () => () => {
      if (copyTimer.current) window.clearTimeout(copyTimer.current);
    },
    [],
  );

  const handleCopy = useCallback(async () => {
    if (onCopy) await onCopy();
    else if (copyText) await navigator.clipboard?.writeText(copyText);

    setCopied(true);
    if (copyTimer.current) window.clearTimeout(copyTimer.current);
    copyTimer.current = window.setTimeout(() => setCopied(false), 1600);
  }, [copyText, onCopy]);

  return (
    <div data-state={status} aria-busy={streaming} className={cn("w-full", className)}>
      <div aria-live={announce ? "polite" : "off"} className={cn("text-sm leading-6", contentClassName)}>
        {children}
      </div>

      <AnimatePresence initial={false}>
        {shouldShowActions ? (
          <motion.div
            initial={reduce ? { opacity: 0 } : { opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: reduce ? 0.12 : 0.22, ease: EASE_OUT }}
            className="mt-3"
          >
            <div className={cn("flex items-center gap-0.5", actionsClassName)}>
              {canCopy ? (
                <ResponseAction label={copied ? "已复制" : "复制回答"} onClick={handleCopy}>
                  {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
                </ResponseAction>
              ) : null}
              {onRetry ? (
                <ResponseAction label="重试" onClick={onRetry}>
                  <RotateCcw className="size-3.5" />
                </ResponseAction>
              ) : null}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
