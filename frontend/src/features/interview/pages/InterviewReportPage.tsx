import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, RefreshCw, Sparkles, Upload } from "lucide-react";
import { api, type Interview } from "../../../api";

const dimensionLabels: Record<string, string> = {
  technical_ability: "技术能力",
  problem_analysis: "问题分析",
  solution_tradeoffs: "方案权衡",
  communication: "表达沟通",
};

function scoreBand(score: number) {
  if (score >= 80) return "达到建议线";
  if (score >= 65) return "接近建议线";
  return "尚未达到建议线";
}

export function InterviewReportPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [interview, setInterview] = useState<Interview | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const [regenerating, setRegenerating] = useState(false);
  const [regenerateError, setRegenerateError] = useState("");

  useEffect(() => {
    if (!id) return;
    const interviewId = id;
    let cancelled = false;
    let timer = 0;

    async function load() {
      const data = await api.interview(interviewId);
      if (cancelled) return;
      setInterview(data);
      // 复盘在后台写。报告还没落库时隔几秒再问一次，不占着结束请求。
      if (!data.report) timer = window.setTimeout(load, 2500);
    }

    load().catch(() => undefined);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [id, reloadToken]);

  async function regenerate() {
    if (!id || regenerating) return;
    setRegenerating(true);
    setRegenerateError("");
    try {
      const next = await api.regenerateReport(id);
      setInterview(next);
      // The response intentionally has no report. Restart the existing polling
      // effect so the page observes the newly queued result.
      setReloadToken((value) => value + 1);
    } catch (error) {
      setRegenerateError(error instanceof Error ? error.message : "重新生成复盘失败");
    } finally {
      setRegenerating(false);
    }
  }

  const report = interview?.report;
  const score = report?.score ?? 0;
  const when = interview?.ended_at || interview?.created_at;
  const whenLabel = when ? new Date(when).toISOString().slice(0, 16).replace("T", " ") : "";

  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas">
      <header className="flex h-16 items-center justify-between border-b border-line px-6">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-forest">
            <Sparkles size={16} className="text-mint-2" />
          </div>
          <div className="text-sm font-semibold">
            {interview?.title ? interview.title.replace("全真模拟面试", "面试报告") : "面试报告"}
            {whenLabel ? ` (${whenLabel})` : ""}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={regenerate}
            disabled={regenerating || !interview}
            aria-busy={regenerating}
            /* Regeneration is recoverable maintenance work, so it remains secondary to exporting. */
            className="apple-button apple-button--secondary apple-button--sm flex items-center gap-1.5 rounded-lg border border-mint-3/50 bg-mint-3/10 px-4 py-2 text-xs text-mint-2 disabled:cursor-wait disabled:opacity-60"
          >
            <RefreshCw size={14} className={regenerating ? "animate-spin" : ""} />
            {regenerating ? "重新生成中…" : "重新生成复盘（测试）"}
          </button>
          <button onClick={() => id && api.downloadReport(id)} className="apple-button apple-button--outline apple-button--sm flex items-center gap-1.5 rounded-lg border border-line-strong bg-row px-4 py-2 text-xs">
            <Upload size={14} className="text-mute" />
            导出复盘报告 ↗
          </button>
        </div>
      </header>

      {regenerateError ? (
        <div role="alert" className="border-b border-red-400/30 bg-red-500/10 px-6 py-2 text-xs text-red-200">
          {regenerateError}
        </div>
      ) : null}

      <div className="min-h-0 flex-1 space-y-[18px] overflow-y-auto px-8 py-[22px]">
        {/* Return navigation is intentionally quiet so the score and report remain the visual focus. */}
        <button onClick={() => navigate("/interview")} className="apple-button apple-button--ghost apple-button--sm flex items-center gap-1.5 text-xs text-mute">
          <ArrowLeft size={13} />
          返回模拟面试
        </button>

        <div className="flex flex-wrap items-center justify-between gap-5 rounded-[14px] border border-forest-2/40 bg-card-live px-[26px] py-[22px]">
          <div className="flex flex-wrap items-center gap-5">
            <div>
              <div className="text-[11px] text-mint-3">综合总评分</div>
              <div className="font-mono text-[44px] leading-none text-mint-2">{score.toFixed(1)}</div>
            </div>
            <div className="space-y-1 text-[11px] text-dim">
              <div>{!report ? "等待评分" : report.scoring_status === "unavailable" ? "评分服务不可用" : report.scoring_status === "invalid" ? "评分结果无效" : report.scoring_status === "legacy" ? "历史综合评分" : scoreBand(score)}</div>
              <div>建议线 ≥80 · 四项各 25 分</div>
            </div>
          </div>
        </div>

        <section className="space-y-3 rounded-[14px] border border-line bg-card p-[22px]">
          <div className="text-sm font-semibold">分项评分与回答依据</div>
          {report?.scoring_status === "valid" && report.dimensions ? (
            <div className="divide-y divide-line">
              {Object.entries(report.dimensions).map(([key, dimension]) => (
                <div key={key} className="grid gap-2 py-3 first:pt-0 last:pb-0 sm:grid-cols-[130px_62px_minmax(0,1fr)] sm:items-start">
                  <div className="text-[13px] font-medium text-ink">{dimensionLabels[key] || key}</div>
                  <div className="font-mono text-[13px] text-mint">{dimension.score.toFixed(1)} / 25</div>
                  <div className="space-y-1.5 text-xs leading-5">
                    <div className="text-dim">依据：{dimension.evidence}</div>
                    <div className="text-mint-3">建议：{dimension.advice}</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs leading-5 text-dim">
              {!report
                ? "分项评分将在复盘完成后显示。"
                : report.scoring_status === "unavailable"
                  ? "评分模型当前不可用，本次报告没有有效分数。修复模型配置后重新评估。"
                  : report.scoring_status === "invalid"
                    ? "本次评分未通过完整性校验，报告中的 0 分仅表示评分失败。"
                    : "此报告生成于分项评分启用之前，仅包含历史综合评分。"}
            </p>
          )}
        </section>

        <div className="rounded-[14px] border border-line bg-card p-[22px]">
          <div className="mb-3 text-sm font-semibold">面试综合表现点评</div>
          <p className="text-[13px] leading-6 text-ink-3">
            {report?.review || (interview && !report ? "复盘还在后台生成，页面可以先离开。" : "")}
          </p>
        </div>

        <div className="space-y-4 rounded-[14px] border border-line bg-card p-[22px]">
          <div className="text-sm font-semibold">关键失分点复盘 & 下次改进建议</div>
          {(report?.issues || []).map((item) => (
            <div key={item.issue} className="space-y-2 rounded-[10px] border border-line bg-well p-3.5">
              <div className="text-[13px] text-ink">• {item.issue}</div>
              <div className="text-xs text-dim">{item.quote}</div>
              <div className="text-xs text-mint-3">{item.advice}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
