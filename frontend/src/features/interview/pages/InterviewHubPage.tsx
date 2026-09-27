import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Play, Plus, Sparkles, Trash2 } from "lucide-react";
import { api, type Interview } from "../../../api";
import { ConfirmDialog } from "../../../components/feedback/ConfirmDialog";
import { notify } from "../../../lib/feedback/notify";

function formatElapsed(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatWhen(iso?: string | null) {
  if (!iso) return "";
  const d = new Date(iso);
  return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

export function InterviewHubPage() {
  const navigate = useNavigate();
  const [items, setItems] = useState<Interview[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<Interview | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer = 0;

    async function loadInterviews() {
      try {
        const data = await api.interviews();
        if (cancelled) return;
        setItems(data);
        setLoaded(true);
        // Ending an interview stops the timer before the background report job finishes.
        // Refresh only while an ended card has no report, then stop polling once it is ready.
        if (data.some((item) => item.status === "ended" && !item.report)) {
          timer = window.setTimeout(() => void loadInterviews(), 2500);
        }
      } catch {
        if (!cancelled) setLoaded(true);
      }
    }

    void loadInterviews();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);

  async function confirmDeleteInterview() {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      await api.deleteInterview(pendingDelete.id);
      setItems((prev) => prev.filter((row) => row.id !== pendingDelete.id));
      setPendingDelete(null);
      notify("面试已删除", "ok");
    } catch (err) {
      notify(err instanceof Error ? err.message : "删除失败", "error");
    } finally {
      setDeleting(false);
    }
  }

  async function openCard(item: Interview) {
    if (item.status === "ended") {
      // The report route can render a waiting state, but the hub keeps this action
      // disabled until the report has actually been persisted.
      if (!item.report) return;
      navigate(`/interview/${item.id}/report`);
      return;
    }
    if (item.status === "ready" || item.status === "abandoned") {
      try {
        const iv = await api.openInterview(item.id);
        navigate(`/interview/${iv.id}`);
      } catch (err) {
        notify(err instanceof Error ? err.message : "无法开始面试", "error");
      }
      return;
    }
    navigate(`/interview/${item.id}`);
  }

  const live = items.find((i) => i.status === "live");
  const rest = items.filter((i) => i.id !== live?.id);

  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas">
      <header className="flex h-16 items-center justify-between border-b border-line px-6">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-forest">
            <Sparkles size={16} className="text-mint-2" />
          </div>
          <div className="text-sm font-semibold">模拟面试</div>
        </div>
        <button
          onClick={() => navigate("/interview/new")}
          className="apple-button apple-button--primary apple-button--sm flex items-center gap-1.5 rounded-lg bg-forest px-3.5 py-2 text-xs font-semibold text-mint-4"
        >
          <Plus size={14} className="text-current" />
          创建面试
        </button>
      </header>

      <div className="min-h-0 flex-1 space-y-[22px] overflow-y-auto px-8 py-7">
        <div className="space-y-1.5">
          <h1 className="text-2xl font-bold">模拟面试</h1>
          <p className="text-[13px] text-mute">选择目标场次开始实战，或查阅历史场次的面试报告</p>
        </div>
        {live ? (
          <div className="flex items-center justify-between gap-5 rounded-[14px] border border-forest-2/40 bg-card-live p-5">
            <div className="min-w-0 flex-1 space-y-2">
              <div className="flex items-center gap-2.5">
                <span className="flex items-center gap-1.5 rounded-full bg-forest px-2.5 py-0.5 text-[11px] font-semibold text-mint-4">
                  <span className="h-1.5 w-1.5 rounded-full bg-mint" />
                  进行中
                </span>
                <span className="text-[11px] text-dim">
                  {formatWhen(live.started_at)} 发起 · 已进行 {live.current_question_index + 1} 轮深度追问
                </span>
              </div>
              <div className="text-[17px] font-semibold">{live.title}</div>
              <div className="flex flex-wrap gap-1.5">
                {(live.tags || []).map((tag) => (
                  <span key={tag} className="rounded-full bg-chip px-2.5 py-0.5 text-[10px] text-mute">
                    {tag}
                  </span>
                ))}
              </div>
              <p className="text-xs text-mute">{live.summary || "支持随时回到面试场次。"}</p>
            </div>
            <div className="flex flex-col items-center gap-2.5">
              <div className="font-mono text-[22px] font-semibold text-mint-2">{formatElapsed(live.elapsed_seconds)}</div>
              <button
                onClick={() => navigate(`/interview/${live.id}`)}
                /* Resuming a live interview is the strongest action in this card. */
                className="apple-button apple-button--primary flex items-center gap-1.5 rounded-lg border border-forest-2/40 bg-forest px-[18px] py-2.5 text-xs font-semibold text-mint-4"
              >
                <Play size={14} className="text-mint-2" />
                继续面试 ➔
              </button>
              <button onClick={() => setPendingDelete(live)} className="text-[11px] text-dim">
                删除
              </button>
            </div>
          </div>
        ) : null}

        {loaded && !live && rest.length === 0 ? (
          <div className="rounded-[14px] border border-line bg-card p-8 text-sm text-dim">
            还没有面试场次。点右上角「创建面试」，写下岗位后再开始。
          </div>
        ) : null}

        <div className="grid min-h-0 grid-cols-3 gap-4">
          {rest.map((item) => {
            const ready = item.status === "ready" || item.status === "abandoned";
            const reviewing = item.status === "ended" && !item.report;
            const ended = item.status === "ended" && Boolean(item.report);
            return (
              // Keep each status color consistent across the card surface, badge, and action.
              <div
                key={item.id}
                className={`flex h-[248px] flex-col justify-between overflow-hidden rounded-[14px] border p-4 ${
                  reviewing
                    ? "border-status-review-border bg-status-review-card"
                    : ended
                      ? "border-forest-2/40 bg-card-live"
                      : ready
                        ? "border-status-ready-border bg-status-ready-card"
                        : "border-line bg-card"
                }`}
              >
                <div className="min-h-0 flex-1 space-y-2 overflow-hidden">
                  <div className="flex items-center justify-between">
                    <span
                      className={`rounded-full px-2.5 py-0.5 text-[10px] font-semibold ${
                        reviewing
                          ? "bg-status-review-badge text-status-review-text"
                          : ended
                            ? "bg-forest text-mint-4"
                            : ready
                              ? "bg-status-ready-badge text-status-ready-text"
                              : "bg-chip text-mute"
                      }`}
                    >
                      {reviewing ? "正在复盘" : ended ? "已完成" : "待开始"}
                    </span>
                    <span className="flex items-center gap-2">
                      {item.score != null ? <span className="text-base font-bold">{item.score.toFixed(1)}</span> : null}
                      <button
                        onClick={() => setPendingDelete(item)}
                        className="flex h-6 w-6 items-center justify-center rounded-md text-dim"
                        aria-label={`删除面试 ${item.title}`}
                      >
                        <Trash2 size={13} />
                      </button>
                    </span>
                  </div>
                  <div className="line-clamp-2 text-[15px] font-semibold">{item.title}</div>
                  <div className="text-[11px] text-dim">
                    {reviewing
                      ? `${formatWhen(item.ended_at)} 结束 · 复盘生成中`
                      : ended
                        ? `${formatWhen(item.ended_at)} 完成`
                        : "基于岗位要求生成 · 预计时长 30 分钟"}
                  </div>
                  <div className="flex h-5 gap-1.5 overflow-hidden">
                    {(item.tags || []).slice(0, 3).map((tag) => (
                      <span key={tag} className="shrink-0 rounded-full bg-chip px-2 py-0.5 text-[10px] text-mute">
                        {tag}
                      </span>
                    ))}
                  </div>
                  <p className="line-clamp-2 text-xs leading-5 text-mute">{item.summary}</p>
                </div>
                <button
                  onClick={() => openCard(item)}
                  disabled={reviewing}
                  /* Ready cards use their blue status treatment; report-wait stays secondary. */
                  className={`apple-button ${reviewing ? "apple-button--secondary" : ended ? "apple-button--primary" : ready ? "apple-button--ready" : "apple-button--primary"} apple-button--sm mt-3 rounded-lg border px-3 py-2 text-xs transition-colors ${
                    reviewing
                      ? "cursor-wait border-status-review-border bg-status-review-badge font-semibold text-status-review-text"
                      : ended
                        ? "border-forest-2/40 bg-forest font-semibold text-mint-4"
                        : ready
                          ? "border-status-ready-border bg-status-ready-badge font-semibold text-status-ready-text"
                          : "border-line-strong bg-row text-ink-2"
                  }`}
                >
                  {reviewing ? "正在生成复盘" : ended ? "查看报告 ↗" : "开始面试 ➔"}
                </button>
              </div>
            );
          })}
        </div>
      </div>
      <ConfirmDialog
        open={pendingDelete !== null}
        title="删除面试"
        body={`删除「${pendingDelete?.title || "这场面试"}」？这场面试的对话和报告会一起删掉，不能恢复。`}
        confirmLabel="确认删除"
        busyLabel="删除中…"
        busy={deleting}
        onCancel={() => setPendingDelete(null)}
        onConfirm={confirmDeleteInterview}
      />
    </div>
  );
}
