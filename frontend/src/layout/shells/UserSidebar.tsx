import { Mic, PanelLeftOpen, Plus, Trash2 } from "lucide-react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import type { ChatSession } from "../../api";
import { ThemeMenu } from "../preferences/ThemeMenu";
import { BrandMark } from "../../components/brand/BrandMark";

export function UserSidebar({
  sessions,
  currentId,
  collapsed,
  onToggle,
  onNew,
  onSelect,
  onDelete,
}: {
  sessions: ChatSession[];
  currentId?: string | null;
  collapsed: boolean;
  onToggle: () => void;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const interviewActive = pathname.startsWith("/interview");

  if (collapsed) {
    return (
      <aside className="apple-sidebar flex h-full w-14 shrink-0 flex-col items-center gap-3 border-r border-line-strong bg-rail py-4">
        <button
          onClick={onToggle}
          className="flex h-8 w-8 items-center justify-center rounded-md text-dim transition-[background-color,color,transform] duration-150 hover:bg-field hover:text-ink active:scale-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forest-2/60"
          aria-label="展开侧栏"
        >
          <PanelLeftOpen size={16} />
        </button>
        {/* 侧栏操作统一提供 hover、按下和键盘焦点反馈，避免折叠状态失去可交互线索。 */}
        <button
          onClick={onNew}
          className="flex h-8 w-8 items-center justify-center rounded-md bg-forest text-mint-2 transition-[background-color,color,transform] duration-150 hover:bg-forest-2 hover:text-mint-3 active:scale-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mint/50"
          aria-label="新建会话"
        >
          <Plus size={16} />
        </button>
        <NavLink
          to="/interview"
          aria-current={interviewActive ? "page" : undefined}
          className={`flex h-8 w-8 items-center justify-center rounded-md border transition-[background-color,border-color,color,transform] duration-150 active:scale-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forest-2/60 ${
            interviewActive
              ? "border-line-strong bg-field text-ink hover:border-line-strong hover:bg-elevated"
              : "border-transparent text-mute hover:border-line-strong hover:bg-field hover:text-ink"
          }`}
          aria-label="模拟面试"
        >
          <Mic size={16} className={interviewActive ? "text-mint" : "text-dim"} />
        </NavLink>
        <div className="mt-auto">
          <ThemeMenu />
        </div>
      </aside>
    );
  }

  return (
    <aside className="apple-sidebar flex h-full w-[260px] shrink-0 flex-col justify-between border-r border-line-strong bg-rail px-3 py-4">
      <div className="flex min-h-0 flex-1 flex-col gap-3">
        <div className="flex items-center justify-between p-1">
          <div className="flex items-center gap-2">
            <BrandMark />
          </div>
        </div>

        <button
          onClick={onNew}
          /* Keep the rail's existing 36px height; the shared class supplies Apple press and color states. */
          className="apple-button apple-button--primary flex h-9 w-full items-center justify-center gap-1.5 rounded-lg border border-forest-2/40 bg-forest px-3 text-[13px] font-semibold text-mint-4 transition-[background-color,border-color,color,transform] duration-150 hover:border-forest-2 hover:bg-forest-2 hover:text-mint-4 active:scale-[0.98] active:bg-forest focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mint/50"
        >
          <Plus size={14} className="text-current" />
          新建会话
        </button>

        <nav className="flex flex-col gap-1">
          <NavLink
            to="/interview"
            aria-current={interviewActive ? "page" : undefined}
            /* Active navigation gets a filled secondary treatment; inactive navigation stays outlined. */
            className={`apple-button ${interviewActive ? "apple-button--secondary" : "apple-button--outline"} flex h-9 items-center gap-2.5 rounded-md border px-3 text-xs transition-[background-color,border-color,color,transform] duration-150 active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forest-2/60 ${
              interviewActive
                ? "border-line-strong bg-field font-semibold text-ink hover:border-line-strong hover:bg-elevated"
                : "border-transparent text-mute hover:border-line-strong hover:bg-field hover:text-ink"
            }`}
          >
            <Mic size={15} className={interviewActive ? "text-mint" : "text-dim"} />
            模拟面试
          </NavLink>
        </nav>

        <div className="px-2 pt-1 text-[11px] font-medium text-dim">历史会话</div>
        <div className="min-h-0 flex-1 space-y-0.5 overflow-y-auto">
          {sessions.map((item) => {
            const active = !interviewActive && currentId === item.id;
            return (
              <div
                key={item.id}
                /* Hover 属于整行，删除按钮单独保留自己的 focus/active 状态。 */
                className={`group flex w-full items-center gap-1 rounded-lg border border-transparent pr-1 text-xs transition-[background-color,border-color,color] duration-150 ${
                  active
                    ? "sidebar-state-active font-medium text-ink"
                    : "text-mute hover:border-line-strong/70 hover:bg-field hover:text-ink"
                }`}
              >
                <button
                  onClick={() => {
                    onSelect(item.id);
                    if (pathname !== "/") navigate("/");
                  }}
                  aria-current={active ? "page" : undefined}
                  className="user-session-link flex min-w-0 flex-1 items-center gap-2 rounded-[inherit] px-2.5 py-2 text-left transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forest-2/60 focus-visible:ring-inset"
                >
                  <span className="truncate">{item.title}</span>
                </button>
                <button
                  onClick={() => onDelete(item.id)}
                  className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-dim opacity-0 transition-[background-color,color,opacity] duration-150 group-hover:opacity-100 hover:bg-line hover:text-ink active:bg-line-strong focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forest-2/60"
                  aria-label={`删除会话 ${item.title}`}
                >
                  <Trash2 size={13} />
                </button>
              </div>
            );
          })}
        </div>
      </div>

      <div className="flex items-center justify-between border-t border-line px-2 py-2.5">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-forest text-[11px] font-semibold text-mint-2">
            七
          </div>
          <span className="text-xs font-medium text-ink-2">小七</span>
        </div>
        <ThemeMenu />
      </div>
    </aside>
  );
}
