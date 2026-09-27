import { useEffect, useId, useMemo, useRef, useState, type CSSProperties } from "react";
import { ChevronDown } from "lucide-react";

export type SelectOption = {
  value: string;
  label: string;
  disabled?: boolean;
};

type SelectProps = {
  value: string;
  onChange: (value: string) => void;
  options: SelectOption[];
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  /** 贴在容器底边时向上展开，避免菜单被 overflow 裁掉。 */
  placement?: "bottom" | "top";
  "aria-label"?: string;
};

/**
 * 全局单选下拉：自定义面板，避开系统原生 select 在暗色主题下的白底弹出层。
 * 交互对齐管理端已有模型选择器（点击展开、外点关闭、键盘方向键）。
 */
export function Select({
  value,
  onChange,
  options,
  placeholder = "请选择",
  disabled,
  className = "",
  placement = "bottom",
  "aria-label": ariaLabel,
}: SelectProps) {
  const listId = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  // Keep the panel mounted briefly after closing so CSS can animate the exit
  // state instead of abruptly removing the menu from the accessibility tree.
  const [menuMounted, setMenuMounted] = useState(false);
  const selected = useMemo(() => options.find((o) => o.value === value), [options, value]);
  const [active, setActive] = useState(value);
  const activeRef = useRef(active);
  const optionsRef = useRef(options);
  const onChangeRef = useRef(onChange);
  activeRef.current = active;
  optionsRef.current = options;
  onChangeRef.current = onChange;

  useEffect(() => {
    if (open) {
      setMenuMounted(true);
      return;
    }
    const timer = window.setTimeout(() => setMenuMounted(false), 180);
    return () => window.clearTimeout(timer);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    // 只在打开瞬间对齐高亮；选项数组常是调用方内联创建，不能放进依赖。
    setActive(value || optionsRef.current.find((o) => !o.disabled)?.value || "");
    const onDoc = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        return;
      }
      const enabled = optionsRef.current.filter((o) => !o.disabled);
      if (!enabled.length) return;
      const current = activeRef.current;
      const idx = Math.max(
        0,
        enabled.findIndex((o) => o.value === current),
      );
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActive(enabled[(idx + 1) % enabled.length].value);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActive(enabled[(idx - 1 + enabled.length) % enabled.length].value);
      } else if (e.key === "Enter") {
        e.preventDefault();
        const next = enabled.find((o) => o.value === current) || enabled[0];
        onChangeRef.current(next.value);
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, value]);

  return (
    <div ref={rootRef} className={`ui-select ${className}`.trim()}>
      <button
        type="button"
        disabled={disabled}
        className="ui-select-trigger"
        data-state={open ? "open" : "closed"}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listId}
        aria-label={ariaLabel}
        onClick={() => !disabled && setOpen((v) => !v)}
      >
        <span className={selected ? "truncate text-ink-2" : "truncate text-faint"}>
          {selected?.label || placeholder}
        </span>
        <ChevronDown size={14} className={`shrink-0 text-dim ${open ? "rotate-180" : ""}`} />
      </button>
      {menuMounted ? (
        <ul
          id={listId}
          role="listbox"
          aria-hidden={!open}
          data-state={open ? "open" : "closed"}
          className={`ui-select-menu${placement === "top" ? " is-top" : ""}`}
        >
          {options.length === 0 ? (
            <li className="ui-select-empty">暂无选项</li>
          ) : (
            options.map((opt, index) => {
              const on = opt.value === value;
              return (
                <li key={opt.value} role="presentation">
                  <button
                    type="button"
                    role="option"
                    aria-selected={on}
                    disabled={opt.disabled}
                    data-active={opt.value === active}
                    data-selected={on}
                    className="ui-select-option"
                    tabIndex={open ? 0 : -1}
                    style={{ "--select-item-index": index } as CSSProperties}
                    onMouseEnter={() => !opt.disabled && setActive(opt.value)}
                    onClick={() => {
                      if (opt.disabled) return;
                      onChange(opt.value);
                      setOpen(false);
                    }}
                  >
                    <span className="truncate">{opt.label}</span>
                    {on ? <span className="text-mint">✓</span> : null}
                  </button>
                </li>
              );
            })
          )}
        </ul>
      ) : null}
    </div>
  );
}
