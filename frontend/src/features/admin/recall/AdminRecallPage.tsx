import { Play, Search } from "lucide-react";
import { useState } from "react";
import { api, type RecallHit } from "../../../api";
import { notify } from "../../../lib/feedback/notify";

export function AdminRecallPage() {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<RecallHit[]>([]);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    try {
      const data = await api.recall(query);
      setHits(data.hits);
      setStatus(data.index_status);
    } catch (err) {
      notify(err instanceof Error ? err.message : "召回失败", "error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-full flex-col bg-shell">
      <header className="flex h-12 items-center justify-between border-b border-line px-6 text-xs">
        <span className="font-medium text-ink-2">召回调试</span>
      </header>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-8 py-5">
        <section className="space-y-2.5 rounded-xl border border-line bg-card p-4">
          <div className="flex items-center justify-between">
            <h2 className="text-[13px] font-semibold">输入岗位要求或模拟提问 Query</h2>
            <span className="text-[10px] text-dim">{status || "模拟 RAG 链路生成考题前置召回"}</span>
          </div>
          <div className="flex items-center gap-2.5">
            <div className="apple-input-shell flex h-[42px] min-w-0 flex-1 items-center gap-2.5 rounded-lg border border-field-line bg-field px-3.5">
              <Search size={16} className="text-dim" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") run();
                }}
                className="h-full w-full bg-transparent text-xs text-ink outline-none"
              />
            </div>
            <button onClick={run} className="flex h-[42px] items-center gap-1.5 rounded-lg bg-primary px-4 text-xs text-white">
              <Play size={14} />
              {busy ? "召回中…" : "执行召回测试"}
            </button>
          </div>
        </section>
        <section className="space-y-2 rounded-xl border border-line bg-card p-4">
          {hits.length === 0 ? (
            <div className="text-sm text-mute">输入测试内容以查看召回结果</div>
          ) : (
            hits.map((hit, i) => (
              <div key={hit.chunk_id} className="rounded-lg border border-line-strong bg-row p-3">
                <div className="mb-1 flex items-center justify-between text-[11px] text-dim">
                  <span>
                    #{i + 1} · {hit.filename} · chunk {hit.ordinal}
                  </span>
                  <span className="text-mint" title="词法召回相关度：命中词数 / 查询词数，整句命中再加 0.35，上限 1.00">
                    相关度 {hit.score.toFixed(2)}
                  </span>
                </div>
                <p className="text-xs leading-5 text-ink-3">{hit.text.slice(0, 420)}</p>
              </div>
            ))
          )}
        </section>
      </div>
    </div>
  );
}
