import { ChevronLeft, ChevronRight, FileText, FolderUp, Trash2, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Material } from "../../../api";
import { Select } from "../../../components/forms/Select";
import { enqueueMaterialUpload, subscribeMaterialUpload, type MaterialUploadJob } from "../../knowledge/model/materialUpload";
import { notify } from "../../../lib/feedback/notify";

const PAGE_SIZES = [10, 20, 50, 100] as const;

const ALLOWED_EXT = [".pdf", ".md", ".txt"] as const;

type RagCheckboxProps = {
  checked: boolean;
  indeterminate?: boolean;
  disabled?: boolean;
  label: string;
  onChange: () => void;
};

/**
 * Shared RAG material checkbox keeps the hit area aligned with table rows while
 * exposing native checked/indeterminate semantics to assistive technology.
 */
function RagCheckbox({ checked, indeterminate = false, disabled = false, label, onChange }: RagCheckboxProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (inputRef.current) inputRef.current.indeterminate = indeterminate;
  }, [indeterminate]);

  return (
    <label className={`rag-checkbox-control${disabled ? " is-disabled" : ""}`} title={label}>
      <input
        ref={inputRef}
        type="checkbox"
        className="rag-checkbox-input"
        checked={checked}
        disabled={disabled}
        aria-label={label}
        onChange={onChange}
      />
      <span className="rag-checkbox-box" aria-hidden="true" />
    </label>
  );
}

/** RAG 物料只吃这三类；文件夹里其余文件直接跳过。 */
function isAllowedMaterial(name: string) {
  const lower = name.toLowerCase();
  return ALLOWED_EXT.some((ext) => lower.endsWith(ext));
}

function pickAllowed(list: FileList | File[]) {
  return Array.from(list).filter((file) => isAllowedMaterial(file.name));
}

function formatBytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatWhen(iso: string) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString();
}

export function AdminMaterialsPage() {
  const [items, setItems] = useState<Material[]>([]);
  const [job, setJob] = useState<MaterialUploadJob>({
    running: false,
    done: 0,
    total: 0,
    current: "",
    message: "",
  });
  const [page, setPage] = useState(1);
  const [pageDirection, setPageDirection] = useState<"next" | "prev">("next");
  const [pageSize, setPageSize] = useState<(typeof PAGE_SIZES)[number]>(10);
  const [jumpDraft, setJumpDraft] = useState("1");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);

  async function load() {
    setItems(await api.materials());
  }

  useEffect(() => {
    load().catch((err) => notify(err instanceof Error ? err.message : "加载失败", "error"));
  }, []);

  const wasRunning = useRef(false);
  useEffect(() => {
    return subscribeMaterialUpload((next) => {
      setJob(next);
      // 任务在别的页面跑完时，回到本页要看到新物料。
      if (wasRunning.current && !next.running) {
        load().catch((err) => notify(err instanceof Error ? err.message : "加载失败", "error"));
      }
      wasRunning.current = next.running;
    });
  }, []);

  useEffect(() => {
    if (!job.running) return;
    const timer = window.setInterval(() => {
      load().catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(timer);
  }, [job.running]);

  useEffect(() => {
    const el = folderRef.current;
    if (!el) return;
    // Chromium 用 webkitdirectory 打开系统文件夹选择器；directory 是旧前缀。
    el.setAttribute("webkitdirectory", "");
    el.setAttribute("directory", "");
    el.multiple = true;
  }, []);

  function uploadMany(files: File[]) {
    const accepted = pickAllowed(files);
    const skipped = files.length - accepted.length;
    if (!accepted.length) {
      notify(skipped ? "文件夹里没有 PDF / MD / TXT" : "未选择文件", "error");
      return;
    }
    enqueueMaterialUpload(accepted, skipped);
  }

  const pageCount = Math.max(1, Math.ceil(items.length / pageSize));
  const safePage = Math.min(page, pageCount);

  useEffect(() => {
    // 删到最后一页变空时，把页码和跳转框一起收回合法范围。
    if (page !== safePage) {
      setPage(safePage);
      setJumpDraft(String(safePage));
    }
  }, [page, safePage]);

  const rows = useMemo(() => {
    const start = (safePage - 1) * pageSize;
    return items.slice(start, start + pageSize);
  }, [items, pageSize, safePage]);
  const pageIds = rows.map((item) => item.id);
  const pageSelected = pageIds.filter((id) => selected.has(id)).length;
  const allPage = pageIds.length > 0 && pageSelected === pageIds.length;
  const somePage = pageSelected > 0 && !allPage;

  useEffect(() => {
    // 入库或删除后，去掉已经不在清单里的勾选。
    const alive = new Set(items.map((item) => item.id));
    setSelected((cur) => {
      const next = new Set([...cur].filter((id) => alive.has(id)));
      return next.size === cur.size ? cur : next;
    });
  }, [items]);

  function togglePage() {
    setSelected((cur) => {
      const next = new Set(cur);
      if (allPage) pageIds.forEach((id) => next.delete(id));
      else pageIds.forEach((id) => next.add(id));
      return next;
    });
  }

  function toggleRow(id: string) {
    setSelected((cur) => {
      const next = new Set(cur);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function deleteSelected() {
    const ids = [...selected];
    if (!ids.length || deleting) return;
    setDeleting(true);
    const fails: string[] = [];
    try {
      for (const id of ids) {
        try {
          await api.deleteMaterial(id);
        } catch (err) {
          fails.push(err instanceof Error ? err.message : id);
        }
      }
      setSelected(new Set());
      await load();
      if (fails.length) notify(`有 ${fails.length} 条删除失败`, "error");
    } finally {
      setDeleting(false);
    }
  }

  function goPage(next: number) {
    const clamped = Math.min(pageCount, Math.max(1, next));
    if (clamped !== safePage) setPageDirection(clamped > safePage ? "next" : "prev");
    setPage(clamped);
    setJumpDraft(String(clamped));
  }

  function applyJump() {
    const n = Number.parseInt(jumpDraft, 10);
    goPage(Number.isFinite(n) ? n : safePage);
  }

  const ready = items.filter((m) => m.status === "ready");
  const chunks = items.reduce((n, m) => n + m.chunk_count, 0);
  const stats = [
    { label: "已入库核心文档", val: `${ready.length} 份`, sub: "删除后立即退出检索", color: "text-mint" },
    { label: "切分向量 Chunks", val: `${chunks} 块`, sub: "分块大小不进管理端", color: "text-blue-500" },
    { label: "最新构建索引状态", val: chunks ? "就绪" : "空索引", sub: "本期词法召回，向量后置", color: "text-mint" },
    { label: "失败物料", val: `${items.filter((m) => m.status === "failed").length}`, sub: "失败原因写在清单里", color: "text-amber-400" },
  ];

  return (
    <div className="flex h-full flex-col bg-shell">
      <header className="flex h-12 items-center justify-between border-b border-line px-6 text-xs">
        <span className="font-medium text-ink-2">知识物料与分块</span>
      </header>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-8 py-5">
        <div className="grid grid-cols-4 gap-4">
          {stats.map((s) => (
            <div key={s.label} className="space-y-1 admin-card rounded-[10px] border border-line bg-card px-4 py-3">
              <div className="text-[11px] text-dim">{s.label}</div>
              <div className={`text-lg font-semibold ${s.color}`}>{s.val}</div>
              <div className="text-[10px] text-faint">{s.sub}</div>
            </div>
          ))}
        </div>
        <section className="space-y-3 admin-card rounded-xl border border-line bg-card p-4">
          <div className="flex items-center justify-between">
            <h2 className="text-[13px] font-semibold">上传知识物料 (文档 / 规范)</h2>
            <span className="text-[10px] text-dim">支持 PDF / MD / TXT · 可选文件夹批量入库</span>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept=".txt,.md,.pdf"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.length) uploadMany(Array.from(e.target.files));
              e.target.value = "";
            }}
          />
          <input
            ref={folderRef}
            type="file"
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.length) uploadMany(Array.from(e.target.files));
              e.target.value = "";
            }}
          />
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              if (e.dataTransfer.files?.length) uploadMany(Array.from(e.dataTransfer.files));
            }}
            className="flex h-[90px] w-full flex-col items-center justify-center gap-2 rounded-lg border border-field-line bg-field/50"
          >
            <Upload size={22} className="text-mint" />
            <div className="text-[11px] text-ink-3">
              {job.running
                ? `正在入库 ${job.done + 1}/${job.total}${job.current ? ` · ${job.current}` : ""}`
                : "拖拽文件到此处，或选择文件 / 文件夹"}
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                className="rounded-md border border-line-strong px-2.5 py-1 text-[11px] text-ink-2 disabled:opacity-50"
              >
                选择文件
              </button>
              <button
                type="button"
                onClick={() => folderRef.current?.click()}
                className="flex items-center gap-1 rounded-md border border-forest-2/40 bg-forest/30 px-2.5 py-1 text-[11px] text-mint-3 disabled:opacity-50"
              >
                <FolderUp size={12} />
                选择文件夹
              </button>
            </div>
          </div>
          {job.message && !job.running ? <div className="text-[10px] text-dim">{job.message}</div> : null}
          <div className="overflow-hidden rounded-lg border border-line">
            <table className="w-full table-fixed border-collapse text-left text-[11px]">
              <thead className="bg-row text-dim">
                <tr>
                  <th className="w-16 px-3 py-2 font-medium">
                    <RagCheckbox
                      checked={allPage}
                      indeterminate={somePage}
                      disabled={!rows.length}
                      label={allPage ? "取消选择本页物料" : "选择本页物料"}
                      onChange={togglePage}
                    />
                  </th>
                  <th className="px-3 py-2 font-medium">文件</th>
                  <th className="w-20 px-3 py-2 font-medium">状态</th>
                  <th className="w-16 px-3 py-2 font-medium">切块</th>
                  <th className="w-20 px-3 py-2 font-medium">大小</th>
                  <th className="w-36 px-3 py-2 font-medium">入库时间</th>
                  <th className="w-20 px-3 py-2 font-medium">操作</th>
                </tr>
              </thead>
              <tbody className={`paged-list paged-list--${pageDirection}`} key={`${safePage}-${pageSize}`}>
                {rows.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-3 py-8 text-center text-dim">
                      还没有物料。上传 PDF / MD / TXT 后会出现在这里。
                    </td>
                  </tr>
                ) : (
                  rows.map((item) => (
                    <tr key={item.id} className="border-t border-line bg-card">
                      <td className="px-3 py-2">
                        <RagCheckbox
                          checked={selected.has(item.id)}
                          label={`选择 ${item.filename}`}
                          onChange={() => toggleRow(item.id)}
                        />
                      </td>
                      <td className="overflow-hidden px-3 py-2">
                        <div className="flex min-w-0 items-center gap-2">
                          <FileText size={14} className="shrink-0 text-mint" />
                          <div className="min-w-0">
                            <div className="truncate text-ink-2">{item.filename}</div>
                            {/* 失败原因可能是一整段无空格异常，悬停看全文，避免把右侧列挤出表格。 */}
                            {item.error ? (
                              <div className="truncate text-[10px] text-danger" title={item.error}>
                                {item.error}
                              </div>
                            ) : null}
                          </div>
                        </div>
                      </td>
                      <td
                        className={`whitespace-nowrap px-3 py-2 ${
                          item.status === "failed" ? "text-danger" : item.status === "pending" ? "text-dim" : "text-mint"
                        }`}
                      >
                        {item.status === "ready" ? "已入库" : item.status === "failed" ? "失败" : item.status === "pending" ? "入库中" : item.status}
                      </td>
                      <td className="whitespace-nowrap px-3 py-2 text-ink-2">{item.chunk_count}</td>
                      <td className="whitespace-nowrap px-3 py-2 text-ink-2">{formatBytes(item.size_bytes)}</td>
                      <td className="whitespace-nowrap px-3 py-2 text-dim">{formatWhen(item.created_at)}</td>
                      <td className="whitespace-nowrap px-3 py-2">
                        <button
                          type="button"
                          onClick={async () => {
                            await api.deleteMaterial(item.id);
                            await load();
                          }}
                          className="flex items-center gap-1 text-dim"
                        >
                          <Trash2 size={12} />
                          删除
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3 text-[11px] text-dim">
            <div className="flex items-center gap-3">
              <span>
                共 {items.length} 条 · 第 {safePage}/{pageCount} 页
                {selected.size ? ` · 已选 ${selected.size}` : ""}
              </span>
              <button
                type="button"
                disabled={!selected.size || deleting}
                onClick={deleteSelected}
                className="admin-action flex items-center gap-1 rounded-md border border-line-strong px-2 py-1 text-danger disabled:opacity-40"
              >
                <Trash2 size={12} />
                {deleting ? "删除中…" : "删除所选"}
              </button>
            </div>
            <div className="flex h-8 flex-wrap items-center gap-2">
              <span className="leading-8">每页</span>
              <div className="w-[88px] shrink-0">
                <Select
                  aria-label="每页条数"
                  placement="top"
                  value={String(pageSize)}
                  options={PAGE_SIZES.map((n) => ({ value: String(n), label: `${n} 条` }))}
                  onChange={(v) => {
                    const next = Number(v) as (typeof PAGE_SIZES)[number];
                    setPageSize(next);
                    setPage(1);
                    setJumpDraft("1");
                  }}
                />
              </div>
              <button
                type="button"
                disabled={safePage <= 1}
                onClick={() => goPage(safePage - 1)}
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-line-strong disabled:opacity-40"
                aria-label="上一页"
              >
                <ChevronLeft size={14} />
              </button>
              <button
                type="button"
                disabled={safePage >= pageCount}
                onClick={() => goPage(safePage + 1)}
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-line-strong disabled:opacity-40"
                aria-label="下一页"
              >
                <ChevronRight size={14} />
              </button>
              <span className="leading-8">跳至</span>
              <input
                className="h-8 w-12 shrink-0 rounded-md border border-field-line bg-field px-1 text-center text-[11px] text-ink-2 outline-none focus:border-forest-2"
                value={jumpDraft}
                onChange={(e) => setJumpDraft(e.target.value.replace(/[^0-9]/g, ""))}
                onKeyDown={(e) => {
                  if (e.key === "Enter") applyJump();
                }}
                aria-label="跳转页码"
              />
              <span className="leading-8">页</span>
              <button
                type="button"
                onClick={applyJump}
                className="admin-action flex h-8 shrink-0 items-center rounded-md border border-line-strong px-2.5 text-ink-2"
              >
                跳转
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
