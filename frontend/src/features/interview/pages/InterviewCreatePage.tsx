import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, FileText, Sparkles, X } from "lucide-react";
import { api, type ChatAttachment } from "../../../api";
import { pendingInterviewGeneration } from "../../../api/domains/interview";
import { notify } from "../../../lib/feedback/notify";

const ROLES = ["后端工程师", "前端工程师", "算法工程师", "测试工程师", "产品经理"];

export function InterviewCreatePage() {
  const navigate = useNavigate();
  const fileRef = useRef<HTMLInputElement>(null);
  const [role, setRole] = useState("");
  const [customRole, setCustomRole] = useState(false);
  // Restore the exact submitted payload so a reload reuses its durable request key.
  const [jd, setJd] = useState(() => pendingInterviewGeneration()?.content || "");
  const [file, setFile] = useState<ChatAttachment | null>(null);
  const [reading, setReading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [createdId, setCreatedId] = useState("");

  const position = customRole ? role.trim() : role;
  const description = [position ? `目标岗位：${position}` : "", jd.trim(), file?.text ? `【附件 ${file.name}】\n${file.text}` : ""]
    .filter(Boolean)
    .join("\n\n");
  const ready = description.trim().length >= 8 && !reading && !generating;

  async function onPickFile(picked: File) {
    setReading(true);
    try {
      const prepared = await api.prepareChatFile(picked);
      setFile(prepared);
    } catch (err) {
      notify(err instanceof Error ? err.message : "附件读取失败", "error");
    } finally {
      setReading(false);
    }
  }

  async function create() {
    if (!ready) return;
    setGenerating(true);
    setCreatedId("");
    try {
      let nextId = "";
      let failure = "";
      await api.generateInterview(description, (event) => {
        // 生成中不接收思考原文。结构化内容和题干都不能出现在这一页。
        if (event.type === "done") nextId = event.interview.id;
        if (event.type === "error") failure = event.message;
      });
      if (failure) throw new Error(failure);
      if (!nextId) throw new Error("题目没有生成");
      setCreatedId(nextId);
      notify("题目已生成", "ok");
    } catch (err) {
      notify(err instanceof Error ? err.message : "出题失败", "error");
    } finally {
      setGenerating(false);
    }
  }

  async function openCreated() {
    if (!createdId) return;
    try {
      // 新生成的场次是待开始。和列表点卡片一样，先开启再进入。
      const opened = await api.openInterview(createdId);
      navigate(`/interview/${opened.id}`);
    } catch (err) {
      notify(err instanceof Error ? err.message : "无法开始面试", "error");
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas">
      <header className="flex h-16 items-center gap-3 border-b border-line px-6">
        <button
          onClick={() => navigate("/interview")}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-mute hover:text-ink"
          aria-label="返回模拟面试"
        >
          <ArrowLeft size={16} />
        </button>
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-forest">
          <Sparkles size={16} className="text-mint-2" />
        </div>
        <div className="text-sm font-semibold">创建面试</div>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-8 py-8">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-6">
          <div className="space-y-1.5">
            <h1 className="text-2xl font-bold">准备一场模拟面试</h1>
            <p className="text-[13px] text-mute">选一个岗位，或直接贴上招聘描述。题目生成后可以进入这场面试。</p>
          </div>

          <section className="space-y-3">
            <div className="text-sm font-medium">面试岗位</div>
            <div className="flex flex-wrap gap-2">
              {ROLES.map((item) => {
                const selected = !customRole && role === item;
                return (
                  <button
                    key={item}
                    type="button"
                    onClick={() => {
                      setCustomRole(false);
                      setRole(item);
                    }}
                    disabled={generating}
                    className={`rounded-full border px-3.5 py-1.5 text-xs ${
                      selected
                        ? "border-forest-2/50 bg-forest font-semibold text-mint-4"
                        : "border-field-line bg-well text-ink-3"
                    }`}
                  >
                    {item}
                  </button>
                );
              })}
              <button
                type="button"
                onClick={() => {
                  setCustomRole(true);
                  setRole("");
                }}
                disabled={generating}
                className={`rounded-full border px-3.5 py-1.5 text-xs ${
                  customRole
                    ? "border-forest-2/50 bg-forest font-semibold text-mint-4"
                    : "border-field-line bg-well text-ink-3"
                }`}
              >
                自定义
              </button>
            </div>
            {customRole ? (
              <input
                value={role}
                onChange={(event) => setRole(event.target.value)}
                disabled={generating}
                placeholder="输入岗位名称"
                className="apple-input-no-dim w-full rounded-xl border border-field-line bg-well px-3.5 py-2.5 text-sm text-ink outline-none placeholder:text-dim"
              />
            ) : null}
          </section>

          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">岗位描述</div>
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                disabled={generating || reading}
                className="text-xs text-mint-2"
              >
                {reading ? "正在读取" : "上传 JD"}
              </button>
            </div>
            <textarea
              value={jd}
              onChange={(event) => setJd(event.target.value)}
              disabled={generating}
              rows={9}
              placeholder="粘贴招聘描述，或写下职责、要求和业务场景。只选岗位也可以开始。"
              className="apple-input-no-dim w-full resize-y rounded-2xl border border-field-line bg-well px-4 py-3 text-sm leading-6 text-ink outline-none placeholder:text-dim"
            />
            {file ? (
              <div className="flex max-w-sm items-center gap-2 rounded-xl border border-field-line bg-well px-2.5 py-2">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-forest text-mint-2">
                  <FileText size={15} />
                </span>
                <span className="min-w-0 flex-1 truncate text-xs text-ink-2">{file.name}</span>
                <button
                  type="button"
                  onClick={() => setFile(null)}
                  disabled={generating}
                  aria-label={`移除 ${file.name}`}
                  className="text-dim hover:text-ink"
                >
                  <X size={13} />
                </button>
              </div>
            ) : null}
            <input
              ref={fileRef}
              type="file"
              accept=".txt,.md,.pdf,.doc,.docx"
              className="hidden"
              onChange={(event) => {
                const picked = event.target.files?.[0];
                if (picked) void onPickFile(picked);
                event.target.value = "";
              }}
            />
          </section>

          {generating || createdId ? (
            <section className="rounded-2xl border border-field-line bg-well px-4 py-3" aria-live="polite">
              <div className="flex items-center gap-2 text-[13px] text-ink-2">
                {generating ? <GeneratingOrb /> : null}
                <span>{createdId ? "题目已生成" : "正在生成题目"}</span>
              </div>
            </section>
          ) : null}

          <div className="flex justify-end">
            {createdId ? (
              <button
                type="button"
                onClick={() => void openCreated()}
                /* Both outcomes lead directly into the interview, so they share the primary CTA treatment. */
                className="apple-button apple-button--primary apple-button--md rounded-lg bg-forest px-5 py-2.5 text-sm font-semibold text-mint-4"
              >
                查看面试
              </button>
            ) : (
              <button
                type="button"
                onClick={() => void create()}
                disabled={!ready}
                className="apple-button apple-button--primary apple-button--md rounded-lg bg-forest px-5 py-2.5 text-sm font-semibold text-mint-4 disabled:opacity-40"
              >
                {generating ? "正在生成" : "开始生成"}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// M2：每个点沿自己的半径在中心小圆和圆周之间来回。单位是 28px 舞台上的像素。
const GENERATING_DOTS = [
  ["0.0px, -1.5px", "0.0px, -7.0px", "0.0px, -1.5px", "0.0px, -7.0px"],
  ["1.1px, -1.1px", "4.9px, -4.9px", "1.1px, -1.1px", "4.9px, -4.9px"],
  ["1.5px, 0.0px", "7.0px, 0.0px", "1.5px, 0.0px", "7.0px, 0.0px"],
  ["1.1px, 1.1px", "4.9px, 4.9px", "1.1px, 1.1px", "4.9px, 4.9px"],
  ["0.0px, 1.5px", "0.0px, 7.0px", "0.0px, 1.5px", "0.0px, 7.0px"],
  ["-1.1px, 1.1px", "-4.9px, 4.9px", "-1.1px, 1.1px", "-4.9px, 4.9px"],
  ["-1.5px, 0.0px", "-7.0px, 0.0px", "-1.5px, 0.0px", "-7.0px, 0.0px"],
  ["-1.1px, -1.1px", "-4.9px, -4.9px", "-1.1px, -1.1px", "-4.9px, -4.9px"],
] as const;

function GeneratingOrb() {
  return (
    <span className="sage-morph" aria-hidden="true">
      <span className="sage-morph-stage">
        {GENERATING_DOTS.map((dot) => (
          <span
            key={dot[1]}
            className="sage-morph-dot"
            style={{
              ["--m-1" as string]: dot[0],
              ["--m-2" as string]: dot[1],
              ["--m-3" as string]: dot[2],
              ["--m-4" as string]: dot[3],
            }}
          />
        ))}
      </span>
    </span>
  );
}
