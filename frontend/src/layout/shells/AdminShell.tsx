import {
  Cpu,
  Database,
  Gauge,
  Search,
  ShieldCheck,
} from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { ThemeMenu } from "../preferences/ThemeMenu";
import { BrandMark } from "../../components/brand/BrandMark";

const NAV = [
  { to: "/admin/providers", label: "模型供应商接入", icon: Cpu },
  { to: "/admin/materials", label: "知识物料与分块", icon: Database },
  { to: "/admin/recall", label: "召回调试", icon: Search },
  { to: "/admin/eval", label: "出题与评分质检", icon: Gauge },
  { to: "/admin/audit", label: "调用与安全审计", icon: ShieldCheck },
];

export function AdminShell() {
  return (
    <div className="flex h-full min-h-0 overflow-hidden bg-shell text-ink">
      <aside aria-label="管理控制台" className="apple-sidebar flex h-full w-[260px] shrink-0 flex-col justify-between overflow-hidden border-r border-line-strong bg-rail px-3 py-4">
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2 px-2 py-1">
            <BrandMark compact />
            <span className="text-sm font-bold">管理控制台</span>
          </div>
          <nav className="flex flex-col gap-1">
            {NAV.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    `admin-nav-item flex h-9 items-center gap-2.5 rounded-xl px-3 text-xs ${
                      isActive
                        ? "admin-nav-item--active sidebar-state-active text-ink"
                        : "text-mute"
                    }`
                  }
                >
                  {({ isActive }) => (
                    <>
                      <Icon size={15} className={isActive ? "text-ink-2" : "text-dim"} />
                      {item.label}
                    </>
                  )}
                </NavLink>
              );
            })}
          </nav>
        </div>
        <div className="flex items-center justify-between border-t border-line px-2 py-2.5">
          <div className="flex items-center gap-2">
            <span className="h-6 w-6 rounded-full border border-line-strong bg-field" />
            <span className="text-[11px] text-ink-2">小七 · 系统超级管理员</span>
          </div>
          <ThemeMenu />
        </div>
      </aside>
      <main className="min-h-0 min-w-0 flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}
