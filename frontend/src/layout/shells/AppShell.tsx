import { useEffect, useState } from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import { api, type ChatSession } from "../../api";
import { ConfirmDialog } from "../../components/feedback/ConfirmDialog";
import { notify } from "../../lib/feedback/notify";
import { UserSidebar } from "./UserSidebar";

export function AppShell() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  // Start compact on touch-sized viewports so the main workspace keeps a
  // usable reading width; the user can still expand the rail when needed.
  const [collapsed, setCollapsed] = useState(() =>
    typeof window !== "undefined" && window.matchMedia("(max-width: 720px)").matches,
  );
  const [pendingDelete, setPendingDelete] = useState<ChatSession | null>(null);
  const [deleting, setDeleting] = useState(false);
  const liveStage = /^\/interview\/[^/]+$/.test(pathname) && !pathname.endsWith("/report");

  async function refresh(preferredId?: string) {
    const list = await api.sessions();
    setSessions(list);
    // 刷新不自动打开最近一条。只有调用方明确指定，或用户已经点开某条时才选中。
    const next = preferredId || currentId || null;
    setCurrentId(next);
    return next;
  }

  useEffect(() => {
    refresh().catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 720px)");
    const syncRail = () => {
      if (media.matches) setCollapsed(true);
    };
    syncRail();
    media.addEventListener("change", syncRail);
    return () => media.removeEventListener("change", syncRail);
  }, []);

  async function onNew() {
    const created = await api.createSession();
    // Do not put the empty draft into history; just open a blank cockpit.
    setCurrentId(created.id);
    navigate("/");
  }

  function onDeleteSession(id: string) {
    const target = sessions.find((item) => item.id === id);
    setPendingDelete(
      target ?? { id, title: "这条会话", job_title: null, created_at: "", updated_at: "" },
    );
  }

  async function confirmDeleteSession() {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      await api.deleteSession(pendingDelete.id);
      const list = await api.sessions();
      setSessions(list);
      if (currentId === pendingDelete.id) setCurrentId(list[0]?.id ?? null);
      setPendingDelete(null);
      notify("会话已删除", "ok");
    } catch (err) {
      notify(err instanceof Error ? err.message : "删除失败", "error");
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="flex h-full min-h-0 bg-canvas text-ink">
      {liveStage ? null : (
        <UserSidebar
          sessions={sessions}
          currentId={currentId}
          collapsed={collapsed}
          onToggle={() => setCollapsed((v) => !v)}
          onNew={onNew}
          onSelect={(id) => setCurrentId(id)}
          onDelete={onDeleteSession}
        />
      )}
      <main className="min-w-0 flex-1">
        <Outlet context={{ sessions, currentId, setCurrentId, refresh, onNew, collapsed, setCollapsed }} />
      </main>
      <ConfirmDialog
        open={pendingDelete !== null}
        title="删除会话"
        body={`删除「${pendingDelete?.title || "这条会话"}」？这个会话里生成的题目和对应面试也会一起删除，不能恢复。`}
        confirmLabel="确认删除"
        busyLabel="删除中…"
        busy={deleting}
        onCancel={() => setPendingDelete(null)}
        onConfirm={confirmDeleteSession}
      />
    </div>
  );
}
