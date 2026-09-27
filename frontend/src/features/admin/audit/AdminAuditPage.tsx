import { ChevronLeft, ChevronRight, Download } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, type AuditLog, type CallLog } from "../../../api";
import { Select } from "../../../components/forms/Select";
import { notify } from "../../../lib/feedback/notify";

const PAGE_SIZES = [10, 20, 50, 100] as const;

function formatWhen(iso: string) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString();
}

function detailText(detail: AuditLog["detail"]) {
  if (detail == null) return "—";
  if (typeof detail === "string") return detail || "—";
  if (typeof detail !== "object") return String(detail);
  const entries = Object.entries(detail as Record<string, unknown>);
  if (!entries.length) return "—";
  return entries
    .map(([key, value]) => `${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`)
    .join("，");
}

/** 调用日志和操作留痕共用同一套前端分页，避免两张表各写一套页码状态。 */
function usePagedRows<T>(items: T[]) {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState<(typeof PAGE_SIZES)[number]>(10);
  const [jumpDraft, setJumpDraft] = useState("1");
  const pageCount = Math.max(1, Math.ceil(items.length / pageSize));
  const safePage = Math.min(page, pageCount);

  useEffect(() => {
    if (page !== safePage) {
      setPage(safePage);
      setJumpDraft(String(safePage));
    }
  }, [page, safePage]);

  const rows = useMemo(() => {
    const start = (safePage - 1) * pageSize;
    return items.slice(start, start + pageSize);
  }, [items, pageSize, safePage]);

  function goPage(next: number) {
    const clamped = Math.min(pageCount, Math.max(1, next));
    setPage(clamped);
    setJumpDraft(String(clamped));
  }

  function applyJump() {
    const n = Number.parseInt(jumpDraft, 10);
    goPage(Number.isFinite(n) ? n : safePage);
  }

  return {
    rows,
    pageSize,
    setPageSize: (next: (typeof PAGE_SIZES)[number]) => {
      setPageSize(next);
      setPage(1);
      setJumpDraft("1");
    },
    jumpDraft,
    setJumpDraft,
    safePage,
    pageCount,
    total: items.length,
    goPage,
    applyJump,
  };
}

function Pager({
  page,
  label,
}: {
  page: ReturnType<typeof usePagedRows<unknown>>;
  label: string;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-3 py-2 text-[11px] text-dim">
      <span>
        共 {page.total} 条 · 第 {page.safePage}/{page.pageCount} 页
      </span>
      <div className="flex h-8 flex-wrap items-center gap-2">
        <span className="leading-8">每页</span>
        <div className="w-[88px] shrink-0">
          <Select
            aria-label={`${label}每页条数`}
            placement="top"
            value={String(page.pageSize)}
            options={PAGE_SIZES.map((n) => ({ value: String(n), label: `${n} 条` }))}
            onChange={(value) => page.setPageSize(Number(value) as (typeof PAGE_SIZES)[number])}
          />
        </div>
        <button
          type="button"
          disabled={page.safePage <= 1}
          onClick={() => page.goPage(page.safePage - 1)}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-line-strong disabled:opacity-40"
          aria-label={`${label}上一页`}
        >
          <ChevronLeft size={14} />
        </button>
        <button
          type="button"
          disabled={page.safePage >= page.pageCount}
          onClick={() => page.goPage(page.safePage + 1)}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-line-strong disabled:opacity-40"
          aria-label={`${label}下一页`}
        >
          <ChevronRight size={14} />
        </button>
        <span className="leading-8">跳至</span>
        <input
          className="h-8 w-12 shrink-0 rounded-md border border-field-line bg-field px-1 text-center text-[11px] text-ink-2 outline-none focus:border-forest-2"
          value={page.jumpDraft}
          onChange={(event) => page.setJumpDraft(event.target.value.replace(/[^0-9]/g, ""))}
          onKeyDown={(event) => {
            if (event.key === "Enter") page.applyJump();
          }}
          aria-label={`${label}跳转页码`}
        />
        <span className="leading-8">页</span>
        <button
          type="button"
          onClick={page.applyJump}
          className="admin-action flex h-8 shrink-0 items-center rounded-md border border-line-strong px-2.5 text-ink-2"
        >
          跳转
        </button>
      </div>
    </div>
  );
}

export function AdminAuditPage() {
  const [overview, setOverview] = useState<any>(null);
  const [calls, setCalls] = useState<CallLog[]>([]);
  const [audits, setAudits] = useState<AuditLog[]>([]);
  const callPage = usePagedRows(calls);
  const auditPage = usePagedRows(audits);

  async function exportLogs() {
    try {
      const blob = await api.exportLogs();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "sagematch-error-logs.log";
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      notify(err instanceof Error ? err.message : "导出失败", "error");
    }
  }

  useEffect(() => {
    Promise.all([api.adminOverview(), api.callLogs(), api.auditLogs()])
      .then(([ov, c, a]) => {
        setOverview(ov);
        setCalls(c);
        setAudits(a);
      })
      .catch((err) => notify(err instanceof Error ? err.message : "加载失败", "error"));
  }, []);

  const cards = [
    { label: "24h 调用", val: overview?.calls_24h ?? 0 },
    { label: "失败率", val: overview?.fail_rate ?? "0%" },
    { label: "P95 延迟", val: `${overview?.p95_ms ?? 0} ms` },
    { label: "SSE 完成率", val: typeof overview?.sse_completion_rate_24h === "number"
      ? `${(overview.sse_completion_rate_24h * 100).toFixed(1)}%` : "—" },
    { label: "恢复成功率", val: typeof overview?.checkpoint_recovery_rate_24h === "number"
      ? `${(overview.checkpoint_recovery_rate_24h * 100).toFixed(1)}%` : "—" },
  ];

  return (
    <div className="flex h-full flex-col bg-shell">
      <header className="flex h-12 items-center justify-between border-b border-line px-6 text-xs">
        <span className="font-medium text-ink-2">调用与安全审计</span>
        <button type="button" onClick={exportLogs} className="admin-action flex items-center gap-1.5 rounded-md border border-line-strong px-2.5 py-1.5 text-ink-2">
          <Download size={13} /> 导出错误日志
        </button>
      </header>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-8 py-5">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
          {cards.map((c) => (
            <div key={c.label} className="admin-card rounded-[10px] border border-line bg-card px-4 py-3">
              <div className="text-[11px] text-dim">{c.label}</div>
              <div className="text-lg font-semibold text-mint">{c.val}</div>
            </div>
          ))}
        </div>
        <section className="overflow-hidden admin-card rounded-xl border border-line bg-card">
          <h2 className="px-4 py-3 text-[13px] font-semibold">llm_call_log</h2>
          <div className="overflow-x-auto border-t border-line">
            <table className="w-full min-w-[720px] border-collapse text-left text-[11px]">
              <thead className="bg-row text-dim">
                <tr>
                  <th className="px-3 py-2 font-medium">角色</th>
                  <th className="px-3 py-2 font-medium">供应商</th>
                  <th className="px-3 py-2 font-medium">模型</th>
                  <th className="px-3 py-2 font-medium">状态</th>
                  <th className="px-3 py-2 font-medium">延迟</th>
                  <th className="px-3 py-2 font-medium">时间</th>
                </tr>
              </thead>
              <tbody>
                {callPage.rows.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-3 py-8 text-center text-dim">
                      还没有调用记录。
                    </td>
                  </tr>
                ) : (
                  callPage.rows.map((row) => (
                    <tr key={row.id} className="border-t border-line">
                      <td className="whitespace-nowrap px-3 py-2 text-ink-2">{row.role}</td>
                      <td className="max-w-[180px] truncate px-3 py-2 text-ink-2" title={row.provider_name}>
                        {row.provider_name}
                      </td>
                      <td className="max-w-[180px] truncate px-3 py-2 text-ink-2" title={row.model}>
                        {row.model}
                      </td>
                      <td className={`whitespace-nowrap px-3 py-2 ${row.status === "ok" ? "text-mint" : "text-danger"}`}>
                        {row.status}
                        {row.error ? <div className="max-w-[220px] truncate text-[10px] text-danger" title={row.error}>{row.error}</div> : null}
                      </td>
                      <td className="whitespace-nowrap px-3 py-2 text-dim">{row.latency_ms}ms</td>
                      <td className="whitespace-nowrap px-3 py-2 text-dim">{formatWhen(row.created_at)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          <Pager page={callPage} label="调用日志" />
        </section>
        <section className="overflow-hidden admin-card rounded-xl border border-line bg-card">
          <h2 className="px-4 py-3 text-[13px] font-semibold">audit_logs</h2>
          <div className="overflow-x-auto border-t border-line">
            <table className="w-full min-w-[720px] border-collapse text-left text-[11px]">
              <thead className="bg-row text-dim">
                <tr>
                  <th className="px-3 py-2 font-medium">操作者</th>
                  <th className="px-3 py-2 font-medium">动作</th>
                  <th className="px-3 py-2 font-medium">对象</th>
                  <th className="px-3 py-2 font-medium">详情</th>
                  <th className="px-3 py-2 font-medium">时间</th>
                </tr>
              </thead>
              <tbody>
                {auditPage.rows.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-3 py-8 text-center text-dim">
                      还没有操作留痕。
                    </td>
                  </tr>
                ) : (
                  auditPage.rows.map((row) => (
                    <tr key={row.id} className="border-t border-line">
                      <td className="whitespace-nowrap px-3 py-2 text-ink-2">{row.actor}</td>
                      <td className="whitespace-nowrap px-3 py-2 text-ink-2">{row.action}</td>
                      <td className="max-w-[220px] truncate px-3 py-2 text-ink-2" title={row.target}>
                        {row.target}
                      </td>
                      <td className="max-w-[280px] truncate px-3 py-2 text-dim" title={detailText(row.detail)}>
                        {detailText(row.detail)}
                      </td>
                      <td className="whitespace-nowrap px-3 py-2 text-dim">{formatWhen(row.created_at)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          <Pager page={auditPage} label="操作留痕" />
        </section>
      </div>
    </div>
  );
}
