import { useEffect, useRef, useState, type ReactNode } from "react";
import { LayoutGroup, motion, useReducedMotion } from "motion/react";
import { SPRING_LAYOUT } from "../../../lib/appearance/ease";
import { useNavigate, useOutletContext } from "react-router-dom";
import {
  Bot,
  FileText,
  Mic,
  Paperclip,
  User,
  X,
} from "lucide-react";
import { api, type ChatAttachment, type ChatMessage, type ChatSession, type ClarificationAnswer } from "../../../api";
import { PromptInput } from "../components/prompt-input";
import { StreamingResponse } from "../components/streaming-response";
import { MarkdownView } from "../../../components/MarkdownView";
import { notify } from "../../../lib/feedback/notify";
import type { AppOutlet } from "../model/outlet";

const USER_AVATAR_STORAGE_KEY = "sagematch-user-avatar";

function readUserAvatar() {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(USER_AVATAR_STORAGE_KEY) || "";
}

const abortControllers = new Set<AbortController>();

type AskItem = {
  id: string;
  prompt: string;
  options: { id: string; label: string }[];
};

export function SessionPage() {
  const navigate = useNavigate();
  const { currentId, setCurrentId, refresh } = useOutletContext<AppOutlet>();
  const [current, setCurrent] = useState<ChatSession | null>(null);
  const [draft, setDraft] = useState("");
  // 选中的文件先挂在输入框上。点发送才和文字一起出去。
  const [pendingFiles, setPendingFiles] = useState<ChatAttachment[]>([]);
  const [preparing, setPreparing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [revealId, setRevealId] = useState("");
  // 正文已经播完的那条。revealId 还留着，避免回答组件被重置后从头再播。
  const [settledId, setSettledId] = useState("");
  // 真流式正文播完的那条。思考留到这一刻再藏，不能在第一个字到达时就收。
  const [streamSettledId, setStreamSettledId] = useState("");
  // 本轮等待开始的时间。思考标签只显示耗时，不再记录思考正文。
  const [thinkingStartedAt, setThinkingStartedAt] = useState(0);
  // 本轮发出、后端还没回写的用户消息。避免等待时列表仍停在上一轮。
  const [pendingUser, setPendingUser] = useState<ChatMessage | null>(null);
  const [pendingBaseCount, setPendingBaseCount] = useState(0);
  // 模型已经吐出、但还没落库的回答。done 到达后由正式消息替换。
  const [liveReply, setLiveReply] = useState<ChatMessage | null>(null);
  const [userAvatar] = useState(readUserAvatar);
  const streamRef = useRef<HTMLDivElement>(null);
  const scrollFrameRef = useRef<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const reduceMotion = useReducedMotion() ?? false;

  useEffect(() => {
    if (!currentId) {
      setCurrent(null);
      return;
    }
    api.session(currentId).then(setCurrent).catch(() => setCurrent(null));
  }, [currentId]);

  useEffect(() => {
    // 真流式每个增量都会触发渲染；合并到下一帧并用即时滚动，避免连续
    // smooth scroll 动画互相排队，在思考切正文时造成一次明显卡顿。
    if (scrollFrameRef.current !== null) window.cancelAnimationFrame(scrollFrameRef.current);
    scrollFrameRef.current = window.requestAnimationFrame(() => {
      const node = streamRef.current;
      if (node) {
        const streaming = Boolean(liveReply?.content) || busy;
        node.scrollTo({ top: node.scrollHeight, behavior: streaming ? "auto" : "smooth" });
      }
      scrollFrameRef.current = null;
    });
    return () => {
      if (scrollFrameRef.current !== null) {
        window.cancelAnimationFrame(scrollFrameRef.current);
        scrollFrameRef.current = null;
      }
    };
  }, [current?.messages?.length, busy, liveReply?.content, liveReply?.extra?.thinking, liveReply?.extra?.reasoning]);

  async function onSend(text?: string, answers?: ClarificationAnswer[]) {
    const content = (text ?? draft).trim();
    const files = text === undefined && !answers?.length ? pendingFiles : [];
    if ((!content && !answers?.length && !files.length) || busy || preparing) return;
    const controller = new AbortController();
    abortControllers.add(controller);
    setDraft("");
    setPendingFiles([]);
    // 上一轮的展开标记必须先清掉，否则等待期间会把旧回答重新播一遍。
    setRevealId("");
    setSettledId("");
    setStreamSettledId("");
    setThinkingStartedAt(Date.now());
    setLiveReply(null);
    setPendingBaseCount(current?.messages?.length ?? 0);
    setPendingUser(
      content || files.length
        ? {
            id: `pending-${Date.now()}`,
            role: "user",
            content,
            extra: files.length ? { kind: "attachment", files: files.map(({ name, size }) => ({ name, size })) } : null,
            created_at: new Date().toISOString(),
          }
        : null,
    );
    setBusy(true);
    let finishedId = current?.id || "";
    try {
      let blocked = false;
      let redirected = false;
      let streamed = "";
      let thought = "";
      let rationale = "";
      const resumeId = { current: finishedId };
      await api.streamChat(content, current?.id, answers, files, (event) => {
        if (controller.signal.aborted) return;
        if (event.type === "blocked") {
          blocked = true;
          return;
        }
        if (event.type === "redirect") {
          // 出题不在会话里做。这条流没有落消息，直接去面试页。
          redirected = true;
          return;
        }
        if (event.type === "meta") {
          // 思考可能先于正文到达。气泡先占位，标签展开后再填思考文本。
          setLiveReply({
            id: "live-reply",
            role: "assistant",
            content: "",
            extra: event.extra,
            created_at: new Date().toISOString(),
          });
          setRevealId("live-reply");
          if (event.session_id) {
            finishedId = event.session_id;
            resumeId.current = event.session_id;
            setCurrentId(event.session_id);
            // The user turn is already persisted before streaming starts. Refresh
            // the rail at meta time so a newly created conversation appears as
            // soon as it has its first record, instead of waiting for done.
            void refresh(event.session_id).catch(() => undefined);
          }
          return;
        }
        if (event.type === "thinking" || event.type === "reasoning") {
          if (event.type === "thinking") thought += event.text;
          else rationale += event.text;
          const thinkingText = thought;
          const reasoningText = rationale;
          // 思考块可能比 meta 更早到。没有占位消息时也要先建出来，否则这段思考会被丢掉。
          setLiveReply((item) => ({
            id: item?.id || "live-reply",
            role: "assistant",
            content: item?.content || "",
            extra: {
              ...(item?.extra || {}),
              ...(thinkingText ? { thinking: thinkingText } : {}),
              ...(reasoningText ? { reasoning: reasoningText } : {}),
            },
            created_at: item?.created_at || new Date().toISOString(),
          }));
          setRevealId("live-reply");
          return;
        }
        if (event.type === "delta") {
          streamed += event.text;
          const text = streamed;
          setLiveReply((item) => (item ? { ...item, content: text } : item));
          return;
        }
        if (event.type === "reset") {
          // A restarted worker begins a new candidate after replaying old frames.
          streamed = "";
          thought = "";
          rationale = "";
          setLiveReply((item) => item ? { ...item, content: "", extra: {} } : item);
          return;
        }
        if (event.type === "done") {
          const messages = [...(event.session.messages || [])];
          const latest = [...messages].reverse().find((item) => item.role === "assistant");
          if (latest && (thought || rationale)) {
            // 流里已经看到的思考和推理不能在换成正式消息时丢掉。库里已有的字段优先。
            latest.extra = {
              ...(latest.extra || {}),
              ...(!latest.extra?.thinking && thought ? { thinking: thought } : {}),
              ...(!latest.extra?.reasoning && rationale ? { reasoning: rationale } : {}),
            };
          }
          setLiveReply(null);
          setPendingUser(null);
          setCurrent({ ...event.session, messages });
          finishedId = event.session.id;
          setCurrentId(event.session.id);
          // 正文已经按增量显示过。换成落库消息后直接标完成，避免再播一遍。
          if (latest) {
            setRevealId(latest.id);
            setSettledId(latest.id);
            setStreamSettledId(latest.id);
          }
        }
      }, controller.signal);
      if (controller.signal.aborted) {
        // 停止只打断前端。用户消息已经落库时，把这一轮已保存的内容拉回来。
        if (resumeId.current) {
          const saved = await api.session(resumeId.current);
          setCurrent(saved);
          setCurrentId(saved.id);
        }
        setPendingUser(null);
        setLiveReply(null);
        return;
      }
      if (redirected) {
        navigate("/interview/new");
        return;
      }
      if (blocked) {
        const next = await api.chat(content, current?.id, answers, controller.signal, files);
        if ([...(next.messages || [])].reverse().find((item) => item.role === "assistant")?.extra?.kind === "redirect") {
          navigate("/interview/new");
          return;
        }
        const latest = [...(next.messages || [])].reverse().find((item) => item.role === "assistant");
        if (latest) setRevealId(latest.id);
        setPendingUser(null);
        setCurrent(next);
        finishedId = next.id;
        setCurrentId(next.id);
      }
      if (finishedId) await refresh(finishedId);
    } catch (err) {
      if (controller.signal.aborted) return;
      setPendingUser(null);
      setLiveReply(null);
      // The server may already have committed the user turn. Refresh that run,
      // while keeping the unsent text available if no session ID was received.
      if (finishedId) {
        try {
          const saved = await api.session(finishedId);
          setCurrent(saved);
          setCurrentId(saved.id);
          await refresh(saved.id);
        } catch { setDraft(content); }
      } else {
        setDraft(content);
        if (files.length) setPendingFiles(files);
      }
      notify(err instanceof Error ? err.message : "发送失败", "error");
    } finally {
      abortControllers.delete(controller);
      setBusy(false);
    }
  }

  function stopRun() {
    // 只停前端读取。abort 后的收尾会把已经落库的用户消息拉回来。
    for (const controller of abortControllers) controller.abort();
    abortControllers.clear();
  }

  async function onPickFile(file: File) {
    // 先解析，解析完只放进输入框。失败时不占住发送中的状态。
    setPreparing(true);
    try {
      const prepared = await api.prepareChatFile(file);
      setPendingFiles((items) => {
        const next = items.filter((item) => item.name !== prepared.name);
        return [...next, prepared];
      });
    } catch (err) {
      notify(err instanceof Error ? err.message : "附件读取失败", "error");
    } finally {
      setPreparing(false);
    }
  }

  // Session refresh may load the durable user turn before the answer finishes.
  // Once its row increases the count, the optimistic copy must disappear.
  const showPendingUser = pendingUser && (current?.messages?.length ?? 0) <= pendingBaseCount;
  const messages = [...(current?.messages || []), ...(showPendingUser ? [pendingUser] : []), ...(liveReply ? [liveReply] : [])];
  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
  const actions = lastAssistant?.extra?.kind === "clarification" ? [] : lastAssistant?.extra?.actions || [];
  const empty = !busy && messages.length === 0;

  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas">
      <input
        ref={fileRef}
        type="file"
        accept=".txt,.md,.pdf,.doc,.docx"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void onPickFile(file);
          e.target.value = "";
        }}
      />
      <LayoutGroup>
      {empty ? (
        <HomeState
          draft={draft}
          setDraft={setDraft}
          busy={busy}
          files={pendingFiles}
          preparing={preparing}
          onSend={onSend}
          onAttach={() => fileRef.current?.click()}
          onRemoveFile={(name) => setPendingFiles((items) => items.filter((item) => item.name !== name))}
          onStop={() => undefined}
          onChip={(kind) => {
            if (kind === "ask") onSend("请用一段话说清楚缓存击穿、缓存穿透和缓存雪崩的区别。");
            if (kind === "interview") navigate("/interview/new");
          }}
        />
      ) : (
        <>
          <header className="flex h-16 items-center justify-between border-b border-line px-6">
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold">{current?.title || "会话"}</div>
              {current?.job_title ? (
                <span className="rounded-full bg-chip px-2.5 py-0.5 text-[11px] text-mute">{current.job_title}</span>
              ) : null}
            </div>
          </header>
          <div ref={streamRef} className="min-h-0 flex-1 overflow-y-auto px-8 py-8 md:px-16">
            <div className="mx-auto flex w-full max-w-3xl flex-col gap-5">
              {messages.map((msg, index) => (
                <Turn
                  key={msg.id}
                  message={msg}
                  userAvatar={userAvatar}
                  enter={msg.id === revealId || msg.id === pendingUser?.id}
                  streaming={msg.id === revealId && msg.id !== settledId}
                  play={msg.id === revealId}
                  live={msg.id === "live-reply"}
                  thinking={
                    msg.id === revealId &&
                    thinkingStartedAt > 0 &&
                    streamSettledId !== msg.id &&
                    (msg.id === "live-reply" || !msg.content)
                  }
                  onStreamSettled={() => setStreamSettledId("live-reply")}
                  answered={Boolean(messages[index + 1])}
                  onClarify={(answers) => onSend("", answers)}
                  onSettled={() => {
                    if (msg.id === revealId) setSettledId(msg.id);
                  }}
                  onRetry={() => {
                    const previous = [...messages.slice(0, index)].reverse().find((item) => item.role === "user");
                    if (previous?.extra?.answers?.length) onSend("", previous.extra.answers);
                    else if (previous) onSend(previous.content);
                  }}
                />
              ))}
              {busy && !liveReply ? (
                <motion.div
                  key="thinking"
                  initial={reduceMotion ? false : { opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 6 }}
                  transition={reduceMotion ? { duration: 0 } : { duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
                >
                  <LiveActivity />
                </motion.div>
              ) : null}
              {actions.length && !busy ? (
                <div className="flex flex-wrap gap-2 pl-11">
                  {actions.map((action) => (
                    <button
                      key={action}
                      onClick={() => {
                        if (action.includes("模拟面试")) navigate("/interview/new");
                        else onSend(action);
                      }}
                      className="rounded-full border border-field-line bg-well px-3.5 py-2 text-left text-xs text-ink-3 hover:text-ink"
                    >
                      {action}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
          <div className="px-8 pb-8 md:px-16">
            <div className="mx-auto w-full max-w-3xl">
              <Composer
                draft={draft}
                setDraft={setDraft}
                busy={busy}
                files={pendingFiles}
                preparing={preparing}
                onSend={() => onSend()}
                onStop={stopRun}
                onAttach={() => fileRef.current?.click()}
                onRemoveFile={(name) => setPendingFiles((items) => items.filter((item) => item.name !== name))}
              />
            </div>
          </div>
        </>
      )}
      </LayoutGroup>
    </div>
  );
}

function Turn({
  message,
  userAvatar,
  streaming,
  play,
  live,
  thinking,
  answered,
  enter,
  onClarify,
  onRetry,
  onSettled,
  onStreamSettled,
}: {
  message: ChatMessage;
  userAvatar: string;
  streaming: boolean;
  play: boolean;
  /** 这条正文来自模型增量。不再用本地打字机重播。 */
  live: boolean;
  /** 回答还没输出完。思考块只活在这一段。 */
  thinking: boolean;
  answered: boolean;
  /** 只有这一轮新出现的消息入场。历史记录直接就位，避免每次打开都重播。 */
  enter: boolean;
  onClarify: (answers: ClarificationAnswer[]) => void;
  onRetry: () => void;
  onSettled: () => void;
  /** 真流式正文播完。历史消息没有这条回调。 */
  onStreamSettled?: () => void;
}) {
  if (message.role === "user") {
    return (
      <MessageEnter play={enter}>
        <MessageRow from="user" name="你" time={formatTime(message.created_at)} avatar={userAvatar}>
          <div className="flex flex-col items-end gap-2">
            {message.extra?.files?.length ? (
              <div className="flex flex-wrap justify-end gap-2">
                {message.extra.files.map((file) => (
                  <FileChip key={file.name} name={file.name} size={file.size} />
                ))}
              </div>
            ) : null}
            {message.content ? <Bubble variant="solid">{message.content}</Bubble> : null}
          </div>
        </MessageRow>
      </MessageEnter>
    );
  }

  const extra = message.extra;
  const asks = extra?.kind === "clarification" ? asAsks(extra) : [];
  return (
    <MessageEnter play={enter}>
    <MessageRow from="assistant" name="知弈" time={formatTime(message.created_at)}>
      <div className="flex w-full flex-col gap-2.5">
        {/* 思考跟着这一轮输出走。正文播完就卸掉，历史消息不再露出。 */}
        {thinking && (extra?.thinking || extra?.reasoning) ? (
          <ThinkingBlock thinking={extra.thinking} reasoning={extra.reasoning} live />
        ) : thinking ? (
          <ThinkingLabel />
        ) : null}
        {message.content ? (
        <Bubble variant="soft">
          <StreamingAnswer
            text={message.content}
            status={play ? "streaming" : "complete"}
            live={live}
            onRetry={onRetry}
            onSettled={onSettled}
            onStreamSettled={onStreamSettled}
          />
        </Bubble>
        ) : null}
        {asks.length ? <ClarificationCard questions={asks} disabled={answered || streaming} onSubmit={onClarify} /> : null}
      </div>
    </MessageRow>
    </MessageEnter>
  );
}

function MessageEnter({ play, children }: { play: boolean; children: ReactNode }) {
  const reduceMotion = useReducedMotion() ?? false;
  return (
    <motion.div
      initial={play && !reduceMotion ? { opacity: 0, y: 6 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={play && !reduceMotion ? { duration: 0.18, ease: [0.16, 1, 0.3, 1] } : { duration: 0 }}
    >
      {children}
    </motion.div>
  );
}

function MessageRow({
  from,
  name,
  time,
  children,
  avatar,
}: {
  from: "user" | "assistant";
  name: string;
  time: string;
  children: ReactNode;
  avatar?: string;
}) {
  const mine = from === "user";
  return (
    <div className={`flex gap-3 ${mine ? "flex-row-reverse" : ""}`}>
      <div
        className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
          mine ? "bg-ink text-canvas" : "bg-forest text-mint-2"
        }`}
      >
        {mine && avatar ? (
          <img src={avatar} alt="" className="h-full w-full rounded-full object-cover" />
        ) : mine ? (
          <User size={15} />
        ) : (
          <Bot size={15} />
        )}
      </div>
      <div className={`flex min-w-0 flex-1 flex-col gap-1.5 ${mine ? "items-end" : "items-start"}`}>
        <div className="flex items-center gap-2 text-[11px] text-dim">
          <span className="font-medium text-mute">{name}</span>
          <span>{time}</span>
        </div>
        {children}
      </div>
    </div>
  );
}

function Bubble({ variant, children }: { variant: "solid" | "soft"; children: ReactNode }) {
  const solid = variant === "solid";
  return (
    <div
      className={`max-w-[680px] rounded-2xl px-4 py-3 text-sm leading-6 ${
        solid ? "rounded-tr-md border border-forest-2/40 bg-card-live text-ink-2" : "rounded-tl-md bg-elevated text-ink-2"
      }`}
    >
      {children}
    </div>
  );
}

function ThinkingLabel() {
  // 思考文本还没到时只留扫光标签。正文一到，调用方会把这块卸掉。
  return (
    <div className="flex items-center gap-1.5 text-[13px]" aria-live="polite">
      <ThinkingOrb />
      <span className="sage-think-shimmer">正在思考</span>
    </div>
  );
}

function ThinkingBlock({
  thinking,
  reasoning,
  live,
}: {
  thinking?: string;
  reasoning?: string;
  live: boolean;
}) {
  const viewportRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    // 思考变长时停在最新一行。回答开始后不再强制滚动，避免把用户正在看的段落拽走。
    const node = viewportRef.current;
    if (live && node) node.scrollTop = node.scrollHeight;
  }, [thinking, reasoning, live]);
  return (
    <div className="w-full max-w-[420px]">
      <div className="flex items-center gap-1.5 text-[13px] leading-[18px]" aria-live="polite">
        {live ? <ThinkingOrb /> : null}
        {live ? (
          <span className="sage-think-shimmer font-medium">正在思考</span>
        ) : (
          <span className="font-medium text-dim">已思考</span>
        )}
      </div>
      <div
        ref={viewportRef}
        className="mt-1.5 max-h-[180px] overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
      >
        <div className="flex flex-col gap-2">
          {thinking ? <ThoughtSection label="正在想" text={thinking} /> : null}
          {reasoning ? <ThoughtSection label="推理" text={reasoning} /> : null}
        </div>
      </div>
    </div>
  );
}

function ThoughtSection({ label, text }: { label: string; text: string }) {
  return (
    <section>
      <p className="m-0 text-[11px] font-medium text-mute">{label}</p>
      <p className="m-0 mt-0.5 whitespace-pre-wrap text-[13px] leading-5 text-dim">{text}</p>
    </section>
  );
}

function ThinkingOrb() {
  return (
    <span className="sage-orb" aria-hidden="true">
      <span className="sage-orb-stage">
        <span className="sage-orb-shape sage-orb-a" />
        <span className="sage-orb-shape sage-orb-b" />
        <span className="sage-orb-shape sage-orb-c" />
      </span>
    </span>
  );
}

function LiveActivity() {
  return (
    <MessageRow from="assistant" name="知弈" time="现在">
      <ThinkingLabel />
    </MessageRow>
  );
}

function StreamingAnswer({
  text,
  status,
  live,
  onRetry,
  onSettled,
  onStreamSettled,
}: {
  text: string;
  status: "streaming" | "complete";
  /** 真流式已经按到达顺序显示。只有整段返回的澄清和出题才走本地打字机。 */
  live: boolean;
  onRetry: () => void;
  onSettled: () => void;
  /** 真流式没有本地打字机。done 把状态改成 complete 时，思考才收。 */
  onStreamSettled?: () => void;
}) {
  const reduceMotion = useReducedMotion() ?? false;
  const { shown, settled } = useStreamedText(text, status === "streaming" && !live, reduceMotion);
  useEffect(() => {
    // 整段回答在本地播完后收起。真流式由 done 把状态改成 complete。
    if (!live && status === "streaming" && settled) onSettled();
    if (live && status === "complete") onStreamSettled?.();
  }, [live, onSettled, onStreamSettled, settled, status]);
  const visible = live ? text : shown;
  const typing = live ? status === "streaming" : !settled;

  return (
    <StreamingResponse
      status={typing ? "streaming" : "complete"}
      copyText={text}
      onRetry={onRetry}
      announce={false}
    >
      <MarkdownView text={visible || " "} />
      {typing ? <span className="sage-caret" aria-hidden="true" /> : null}
    </StreamingResponse>
  );
}

function ClarificationCard({
  questions,
  disabled,
  onSubmit,
}: {
  questions: AskItem[];
  disabled: boolean;
  onSubmit: (answers: ClarificationAnswer[]) => void;
}) {
  const [picked, setPicked] = useState<Record<string, string>>({});
  const ready = questions.every((question) => picked[question.id]);
  return (
    <div className="w-full max-w-[680px] rounded-xl border border-line bg-card p-3">
      <div className="mb-2 text-xs text-mute">继续之前，先确认方向</div>
      <div className="space-y-3">
        {questions.map((question) => (
          <div key={question.id}>
            <div className="mb-1.5 text-[13px] text-ink-2">{question.prompt}</div>
            <div className="flex flex-wrap gap-1.5">
              {question.options.map((option) => {
                const on = picked[question.id] === option.id;
                return (
                  <button
                    key={option.id}
                    type="button"
                    disabled={disabled}
                    onClick={() => setPicked((current) => ({ ...current, [question.id]: option.id }))}
                    className={`rounded-full border px-3 py-1.5 text-xs ${
                      on ? "border-forest-2 bg-forest text-mint-2" : "border-field-line text-ink-3"
                    } disabled:opacity-60`}
                  >
                    {option.label}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <button
        type="button"
        disabled={disabled || !ready}
        onClick={() =>
          onSubmit(
            questions.map((question) => {
              const option = question.options.find((item) => item.id === picked[question.id]);
              return {
                id: question.id,
                option_id: option?.id || "",
                label: option?.label || "",
                prompt: question.prompt,
              };
            }),
          )
        }
        className="mt-3 rounded-lg bg-forest px-3 py-1.5 text-xs text-mint-2 disabled:opacity-40"
      >
        {disabled ? "已回复" : "按这个方向继续"}
      </button>
    </div>
  );
}

function HomeState({
  draft,
  setDraft,
  busy,
  files,
  preparing,
  onSend,
  onChip,
  onAttach,
  onRemoveFile,
  onStop,
}: {
  draft: string;
  setDraft: (v: string) => void;
  busy: boolean;
  files: ChatAttachment[];
  preparing: boolean;
  onSend: () => void;
  onChip: (kind: "ask" | "interview") => void;
  onAttach: () => void;
  onRemoveFile: (name: string) => void;
  onStop: () => void;
}) {
  return (
    <div className="flex h-full items-center justify-center bg-canvas">
      <div className="flex w-[720px] max-w-full flex-col items-center gap-7 px-6">
        <motion.div
          className="flex flex-col items-center gap-2.5 text-center"
          exit={{ opacity: 0 }}
          transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
        >
          <h1 className="text-[32px] font-semibold tracking-tight">你好，我是知弈</h1>
          <p className="text-[15px] text-mute">请向我提问</p>
        </motion.div>
        <motion.div layoutId="composer" transition={SPRING_LAYOUT} className="w-full">
          <Composer
            draft={draft}
            setDraft={setDraft}
            busy={busy}
            files={files}
            preparing={preparing}
            onSend={onSend}
            onAttach={onAttach}
            onRemoveFile={onRemoveFile}
            onStop={onStop}
            home
          />
        </motion.div>
        <motion.div
          className="flex flex-wrap items-center justify-center gap-2"
          exit={{ opacity: 0 }}
          transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
        >
          <Chip icon={<FileText size={13} />} label="解释一下" onClick={() => onChip("ask")} />
          <Chip icon={<Mic size={13} />} label="去生成面试" onClick={() => onChip("interview")} />
        </motion.div>
        <motion.p
          className="text-[11px] text-faint"
          exit={{ opacity: 0 }}
          transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
        >
          内容由 AI 分析生成，仅供技术面试训练参考
        </motion.p>
      </div>
    </div>
  );
}

function formatFileSize(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${Math.max(1, Math.round(size / 1024))} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function FileChip({
  name,
  size,
  pending = false,
  onRemove,
}: {
  name: string;
  size: number;
  pending?: boolean;
  onRemove?: () => void;
}) {
  return (
    <div className="flex max-w-[240px] items-center gap-2 rounded-xl border border-field-line bg-well px-2.5 py-2">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-forest text-mint-2">
        <FileText size={15} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-xs text-ink-2">{name}</span>
        <span className="block text-[11px] text-dim">{pending ? "读取中" : formatFileSize(size)}</span>
      </span>
      {onRemove ? (
        <button type="button" onClick={onRemove} aria-label={`移除 ${name}`} className="text-dim hover:text-ink">
          <X size={13} />
        </button>
      ) : null}
    </div>
  );
}

function Chip({ icon, label, onClick }: { icon: ReactNode; label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-1.5 rounded-full border border-field-line bg-well px-3.5 py-2 text-xs text-ink-3"
    >
      <span className="text-mute">{icon}</span>
      {label}
    </button>
  );
}

function Composer({
  draft,
  setDraft,
  onSend,
  onAttach,
  onRemoveFile,
  onStop,
  busy,
  files,
  preparing,
  home = false,
}: {
  draft: string;
  setDraft: (v: string) => void;
  onSend: () => void;
  onAttach: () => void;
  onRemoveFile: (name: string) => void;
  onStop: () => void;
  busy: boolean;
  files: ChatAttachment[];
  preparing: boolean;
  home?: boolean;
}) {
  const reduceMotion = useReducedMotion() ?? false;
  const box = (
    <div className="flex flex-col gap-2">
      {files.length || preparing ? (
        <div className="flex flex-wrap gap-2">
          {files.map((file) => (
            <FileChip key={file.name} name={file.name} size={file.size} onRemove={() => onRemoveFile(file.name)} />
          ))}
          {preparing ? <FileChip name="正在读取" size={0} pending /> : null}
        </div>
      ) : null}
      <PromptInput
        value={draft}
        onValueChange={setDraft}
        onSubmit={() => onSend()}
        loading={busy}
        allowEmpty={files.length > 0 && !preparing}
        onStop={onStop}
        minRows={home ? 3 : 2}
        maxRows={8}
        placeholder={home ? "问一个面试知识点…" : "继续提问…"}
        aria-label={home ? "新会话" : "继续对话"}
        actions={[
          {
            value: "file",
            label: "上传资料",
            description: "TXT、MD、PDF 或 Word",
            icon: <Paperclip />,
          },
        ]}
        onAction={(action) => {
          if (action === "file") onAttach();
        }}
      />
    </div>
  );
  // 首页那个框带着 layoutId，落到对话底部时接着同一条弹簧。减少动态效果时不滑。
  if (home || reduceMotion) return box;
  return (
    <motion.div layoutId="composer" transition={SPRING_LAYOUT}>
      {box}
    </motion.div>
  );
}

function useStreamedText(text: string, live: boolean, reduceMotion: boolean) {
  // 历史消息直接给全文。新回答用 rAF 按字符速率展开，和 Streaming Response 的演示节奏一致。
  const [count, setCount] = useState(() => (live && !reduceMotion ? 0 : text.length));
  const played = useRef(false);
  useEffect(() => {
    if (!live || reduceMotion || played.current) {
      // 已经播过的回答保持全文。完成态切换和减少动态效果都不从头再来。
      setCount(text.length);
      return;
    }
    played.current = true;
    setCount(0);
    const startedAt = performance.now();
    let frame = 0;
    const stream = (now: number) => {
      const next = Math.min(text.length, Math.floor(((now - startedAt) / 1000) * 110));
      setCount(next);
      if (next < text.length) frame = window.requestAnimationFrame(stream);
    };
    frame = window.requestAnimationFrame(stream);
    return () => window.cancelAnimationFrame(frame);
  }, [live, reduceMotion, text]);
  return { shown: text.slice(0, count), settled: count >= text.length };
}

function asAsks(value: ChatMessage["extra"] | undefined): AskItem[] {
  const rows = value && "questions" in value ? value.questions : undefined;
  if (!Array.isArray(rows)) return [];
  return rows.flatMap((item) => {
    if (!item?.prompt || !item.options?.length) return [];
    return [{ id: String(item.id || item.prompt), prompt: item.prompt, options: item.options }];
  });
}

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}
