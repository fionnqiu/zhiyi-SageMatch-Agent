import { useEffect, useState } from "react";
import { api, type EvalRun } from "../../../api";
import { notify } from "../../../lib/feedback/notify";

export function AdminEvalPage() {
  const [job, setJob] = useState("");
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [busy, setBusy] = useState("");

  async function load() {
    setRuns(await api.evalRuns());
  }

  useEffect(() => {
    load().catch((err) => notify(err instanceof Error ? err.message : "加载失败", "error"));
  }, []);

  const latestQ = runs.find((r) => r.kind === "question");
  const latestS = runs.find((r) => r.kind === "score");

  return (
    <div className="flex h-full flex-col bg-shell">
      <header className="flex h-12 items-center justify-between border-b border-line px-6 text-xs">
        <span className="font-medium text-ink-2">出题与评分质检</span>
      </header>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-8 py-5">
        <section className="space-y-2 admin-card rounded-xl border border-line bg-card p-4">
          <h2 className="text-[13px] font-semibold">出题质量评测</h2>
          {/* Keep the prompt and its action in one row so the primary control stays adjacent to its input. */}
          <div className="flex items-stretch gap-2">
            <textarea
              value={job}
              onChange={(e) => setJob(e.target.value)}
              rows={1}
              className="min-w-0 h-9 flex-1 resize-y rounded-lg border border-field-line bg-field p-3 text-xs outline-none"
            />
            <button
              onClick={async () => {
                setBusy("q");
                try {
                  await api.evalQuestions(job);
                  await load();
                } catch (err) {
                  notify(err instanceof Error ? err.message : "评测失败", "error");
                } finally {
                  setBusy("");
                }
              }}
              className="admin-action shrink-0 self-center rounded-lg bg-forest px-4 py-2 text-xs text-mint-4 disabled:opacity-60"
              disabled={busy === "q"}
            >
              {busy === "q" ? "评测中…" : "开始评测"}
            </button>
          </div>
          {latestQ?.metrics ? (
            <div className="grid grid-cols-5 gap-2 text-[11px]">
              {Object.entries(latestQ.metrics).map(([k, v]) => (
                <div key={k} className="rounded-md bg-row px-3 py-2">
                  <div className="text-dim">{k}</div>
                  <div className="text-mint">{String(v)}</div>
                </div>
              ))}
            </div>
          ) : null}
        </section>
        <section className="relative space-y-2 admin-card rounded-xl border border-line bg-card p-4">
          <div className="flex items-start justify-between gap-3 pr-1">
            <h2 className="text-[13px] font-semibold">评分一致性评测</h2>
            <button
              onClick={async () => {
                setBusy("s");
                try {
                  await api.evalScores();
                  await load();
                } catch (err) {
                  notify(err instanceof Error ? err.message : "评测失败", "error");
                } finally {
                  setBusy("");
                }
              }}
              className="admin-action shrink-0 rounded-lg border border-forest-2/40 px-3 py-1.5 text-xs text-mint disabled:opacity-60"
              disabled={busy === "s"}
            >
              {busy === "s" ? "评测中…" : "开始评测"}
            </button>
          </div>
          <p className="text-[11px] text-dim">对最近一场已结束面试重复评分 N=5，查看总分与四项维度的波动 σ。</p>
          {latestS?.metrics ? (
            <div className="grid grid-cols-5 gap-2 text-[11px]">
              {Object.entries(latestS.metrics).map(([k, v]) => (
                <div key={k} className="rounded-md bg-row px-3 py-2">
                  <div className="text-dim">{k}</div>
                  <div className="text-mint">{Array.isArray(v) ? v.join(", ") : String(v)}</div>
                </div>
              ))}
            </div>
          ) : null}
        </section>
        <section className="space-y-2 admin-card rounded-xl border border-line bg-card p-4">
          <h2 className="text-[13px] font-semibold">eval_runs 历史</h2>
          {runs.map((run) => (
            <div key={run.id} className="flex items-center justify-between rounded-md bg-row px-3 py-2 text-[11px]">
              <span>
                {run.kind} · {run.status}
              </span>
              <span className="text-dim">{new Date(run.created_at).toLocaleString()}</span>
            </div>
          ))}
        </section>
      </div>
    </div>
  );
}
