import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Hand, Keyboard, LogOut, Mic, Sparkles, Square } from "lucide-react";
import { api, type Interview, type InterviewTurn } from "../../../api";
import { prepareInterviewAnswer, reconcileInterviewAnswer } from "../../../api/domains/interview";
import Strands from "../components/Strands";
import { BarVisualizer } from "../components/BarVisualizer";
import { ConfirmDialog } from "../../../components/feedback/ConfirmDialog";
import { notify } from "../../../lib/feedback/notify";
import "./interview-live.css";

function formatElapsed(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

type Recog = {
  start: () => void;
  stop: () => void;
  abort?: () => void;
  onresult: ((ev: { results: ArrayLike<{ 0: { transcript: string }; isFinal: boolean }> }) => void) | null;
  onend: (() => void) | null;
  onerror: ((ev: { error?: string }) => void) | null;
  lang: string;
  continuous: boolean;
  interimResults: boolean;
};

function getSpeechRecognition(): Recog | null {
  const Ctor = (window as unknown as { webkitSpeechRecognition?: new () => Recog; SpeechRecognition?: new () => Recog })
    .webkitSpeechRecognition || (window as unknown as { SpeechRecognition?: new () => Recog }).SpeechRecognition;
  if (!Ctor) return null;
  return new Ctor();
}

function selectChineseVoice(): SpeechSynthesisVoice | null {
  const voices = window.speechSynthesis?.getVoices() || [];
  const chineseVoices = voices.filter((voice) => /^(zh|cmn)([-_]|$)/i.test(voice.lang));
  if (!chineseVoices.length) return null;

  // Browser voices expose no gender/style metadata, so use common Mandarin feminine voice names as a best-effort signal.
  const warmFeminineName = /xiaoxiao|xiaoyi|huihui|yaoyao|ting[- ]?ting|female|woman|女/i;
  const mainland = chineseVoices.filter((voice) => /^zh[-_]?(cn|hans)$/i.test(voice.lang));
  return mainland.find((voice) => warmFeminineName.test(voice.name) && voice.localService)
    || mainland.find((voice) => warmFeminineName.test(voice.name))
    || mainland.find((voice) => voice.localService)
    || mainland[0]
    || chineseVoices.find((voice) => warmFeminineName.test(voice.name))
    || chineseVoices.find((voice) => voice.localService)
    || chineseVoices[0];
}

export function InterviewLivePage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [interview, setInterview] = useState<Interview | null>(null);
  const [pendingTurn, setPendingTurn] = useState<InterviewTurn | null>(null);
  const [streamingTurn, setStreamingTurn] = useState<InterviewTurn | null>(null);
  const [draft, setDraft] = useState("");
  const [textMode, setTextMode] = useState(false);
  const [listening, setListening] = useState(false);
  const [busy, setBusy] = useState(false);
  const [ending, setEnding] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const [confirmLeave, setConfirmLeave] = useState(false);
  const [reportReadyNotice, setReportReadyNotice] = useState(false);
  // 选择结束并复盘后，计时锁在点击那一刻。后端停表返回前不再跑秒。
  const [frozenElapsed, setFrozenElapsed] = useState<number | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [interviewError, setInterviewError] = useState("");
  const started = useMemo(
    () => (interview?.started_at ? Date.parse(interview.started_at) : Date.now()),
    [interview?.started_at],
  );
  const [now, setNow] = useState(Date.now());
  const recogRef = useRef<Recog | null>(null);
  const conversationRef = useRef<HTMLDivElement | null>(null);
  const lastSpoken = useRef("");
  const requestRef = useRef<AbortController | null>(null);
  const finalTranscript = useRef("");
  const [speechVoices, setSpeechVoices] = useState<SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    const synthesis = window.speechSynthesis;
    if (!synthesis) return;
    const refreshVoices = () => setSpeechVoices(synthesis.getVoices());
    refreshVoices();
    synthesis.addEventListener("voiceschanged", refreshVoices);
    return () => synthesis.removeEventListener("voiceschanged", refreshVoices);
  }, []);

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!id) return;
    let active = true;
    api.interview(id).then(async (data) => {
      if (!active) return;
      // A direct link can reach a ready card without the hub's open command.
      // Start it through the same idempotent API before enabling answer input.
      if (data.status === "ready" || data.status === "abandoned") {
        data = await api.openInterview(id);
      }
      if (!active) return;
      setInterview(data);
      if (data.status === "live") {
        // A reload can interrupt the response after the server commits the turn.
        // Restore the exact submitted payload so retry uses its original key.
        const pending = reconcileInterviewAnswer(id, data.turns);
        if (pending) {
          setDraft(pending.content);
          setTextMode(pending.mode === "text");
          setInterviewError("上次作答结果尚未确认，请重试");
        }
      }
      if (data.status === "ended") navigate(`/interview/${data.id}/report`, { replace: true });
    }).catch((err) => {
      if (active) notify(err instanceof Error ? err.message : "无法加载面试", "error");
    });
    return () => { active = false; };
  }, [id, navigate]);

  const turns = interview?.turns || [];
  const visibleTurns = [
    ...turns.filter((turn) => !streamingTurn || turn.id !== streamingTurn.id),
    ...(pendingTurn ? [pendingTurn] : []),
    ...(streamingTurn ? [streamingTurn] : []),
  ];
  const lastInterviewer = [...visibleTurns].reverse().find((t) => t.role === "interviewer");
  const lastUser = [...visibleTurns].reverse().find((t) => t.role === "user");
  const question = interview?.current_question;
  // The interviewer turn changes on follow-up while current_question stays on the same stem.
  const prompt = lastInterviewer?.content || question?.stem || "";
  const promptKey = lastInterviewer?.id || question?.id || prompt;
  const interviewerReplied = Boolean(lastUser && lastInterviewer && visibleTurns.lastIndexOf(lastInterviewer) > visibleTurns.lastIndexOf(lastUser));

  useEffect(() => {
    // Keep the newest turn visible as the API appends interviewer and candidate messages.
    const element = conversationRef.current;
    if (!element) return;
    const frame = window.requestAnimationFrame(() => {
      element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [turns.length]);

  useEffect(() => {
    // 流式文本尚未完成时不要把首个字符当成完整问题播报；等 streamingTurn 清除、
    // 服务端完整 turn 写回后再触发一次 TTS，避免后续轮次被同一个 turn id 去重。
    if (streamingTurn || !prompt || !interview || interview.status !== "live") return;
    if (lastSpoken.current === promptKey) return;
    lastSpoken.current = promptKey;
    const synthesis = window.speechSynthesis;
    if (!synthesis || typeof SpeechSynthesisUtterance === "undefined") {
      setSpeaking(false);
      return;
    }
    synthesis.cancel();
    const utter = new SpeechSynthesisUtterance(prompt);
    utter.lang = "zh-CN";
    utter.voice = selectChineseVoice();
    utter.rate = 1.1;
    utter.pitch = 1.06;
    utter.volume = 1;
    utter.onstart = () => setSpeaking(true);
    utter.onend = () => setSpeaking(false);
    utter.onerror = () => setSpeaking(false);
    setSpeaking(true);
    // Chromium can leave the speech queue paused after several utterances; resume
    // before each completed interviewer turn so ASR -> TTS remains repeatable.
    synthesis.resume();
    synthesis.speak(utter);
  }, [prompt, promptKey, interview?.status, speechVoices, streamingTurn]);

  useEffect(() => () => {
    requestRef.current?.abort();
    window.speechSynthesis?.cancel();
    recogRef.current?.abort?.();
  }, []);

  useEffect(() => () => {
    // Stop a simulated response stream when leaving the page or changing interviews.
    setStreamingTurn(null);
  }, [id]);

  function interruptSpeech() {
    window.speechSynthesis?.cancel();
    setSpeaking(false);
    recogRef.current?.abort?.();
    setListening(false);
    finalTranscript.current = "";
  }

  async function send(content?: string, mode: "text" | "voice" = textMode ? "text" : "voice") {
    const text = (content ?? draft).trim();
    if (!id || !text || busy || ending || leaving) return;
    interruptSpeech();
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setBusy(true);
    setInterviewError("");
    // Save before clearing the composer; a refresh during the request can then
    // restore the answer and reuse the same server idempotency key.
    const previousAnswerId = [...(interview?.turns || [])].reverse().find((turn) => turn.role === "user")?.id;
    prepareInterviewAnswer(id, text, mode, previousAnswerId);
    // The sent answer now lives in the conversation, so clear the composer immediately.
    setDraft("");
    // Show the submitted answer immediately while the interviewer response is generated.
    setPendingTurn({
      id: `pending-${Date.now()}`,
      role: "user",
      content: text,
      answer_mode: mode,
      created_at: new Date().toISOString(),
    });
    try {
      // Drop TTS before waiting for the model so the answer request owns the audio channel.
      const pause = new Promise<void>((resolve) => window.setTimeout(resolve, 180));
      await pause;
      if (controller.signal.aborted) return;
      const next = await api.answerInterview(id, text, mode, controller.signal);
      if (requestRef.current !== controller) return;
      const reply = [...(next.turns || [])].reverse().find((turn) => turn.role === "interviewer");
      if (reply) {
        // The answer endpoint currently returns JSON; reveal its persisted interviewer turn incrementally.
        for (let index = 1; index <= reply.content.length; index += 1) {
          if (controller.signal.aborted || requestRef.current !== controller) return;
          setStreamingTurn({ ...reply, content: reply.content.slice(0, index) });
          await new Promise<void>((resolve) => window.setTimeout(resolve, 18));
        }
      }
      setStreamingTurn(null);
      // Keep the optimistic candidate turn visible until the persisted interview replaces it.
      // This preserves the question -> answer -> interviewer response order during streaming.
      setPendingTurn(null);
      setInterview(next);
    } catch (err) {
      if (controller.signal.aborted) return;
      const message = err instanceof Error ? err.message : "作答失败";
      setInterviewError(message);
      setPendingTurn(null);
      setDraft(text);
      notify(message, "error");
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setBusy(false);
      }
    }
  }

  function toggleMic() {
    if (textMode) {
      void send(undefined, "text");
      return;
    }
    if (listening) {
      recogRef.current?.stop();
      setListening(false);
      return;
    }
    const rec = getSpeechRecognition();
    if (!rec) {
      notify("当前浏览器不支持语音识别，请改用文字作答", "error");
      setTextMode(true);
      return;
    }
    interruptSpeech();
    setDraft("");
    setInterviewError("");
    rec.lang = "zh-CN";
    rec.continuous = false;
    rec.interimResults = true;
    finalTranscript.current = "";
    rec.onresult = (ev) => {
      let interim = "";
      for (let index = 0; index < ev.results.length; index += 1) {
        const result = ev.results[index];
        const phrase = result?.[0]?.transcript || "";
        if (result?.isFinal) finalTranscript.current += phrase;
        else interim += phrase;
      }
      setDraft(`${finalTranscript.current}${interim}`.trim());
    };
    rec.onerror = (ev) => {
      setListening(false);
      if (ev.error && ev.error !== "aborted") {
        setInterviewError("语音识别中断，已识别的文字会保留，可继续或改用文字。");
        notify("语音识别中断，可保留已识别内容或改用文字作答", "error");
      }
    };
    rec.onend = () => {
      setListening(false);
      const text = finalTranscript.current.trim();
      finalTranscript.current = "";
      if (text && !busy && !ending && !leaving) void send(text, "voice");
    };
    recogRef.current = rec;
    try {
      window.speechSynthesis?.cancel();
      setSpeaking(false);
      rec.start();
      setListening(true);
      setInterviewError("");
    } catch {
      setListening(false);
      setTextMode(true);
      notify("无法启动语音识别，请改用文字作答", "error");
    }
  }

  async function end() {
    if (!id || ending || leaving) return;
    interruptSpeech();
    requestRef.current?.abort();
    setFrozenElapsed(Math.max(0, Math.floor((Date.now() - started) / 1000)));
    setEnding(true);
    try {
      const next = await api.endInterview(id);
      // The report is queued in the background; keep the empty report page out of the flow.
      void next;
      setReportReadyNotice(true);
    } catch (err) {
      notify(err instanceof Error ? err.message : "无法结束面试", "error");
      setFrozenElapsed(null);
      setEnding(false);
    }
  }

  async function leave() {
    if (!id || ending || leaving) return;
    interruptSpeech();
    requestRef.current?.abort();
    setFrozenElapsed(Math.max(0, Math.floor((Date.now() - started) / 1000)));
    setLeaving(true);
    try {
      await api.abandonInterview(id);
      navigate("/interview", { replace: true });
    } catch (err) {
      notify(err instanceof Error ? err.message : "无法退出面试", "error");
      setFrozenElapsed(null);
      setLeaving(false);
    }
  }

  const elapsed =
    frozenElapsed ??
    (interview?.status === "live" ? Math.floor((now - started) / 1000) : interview?.elapsed_seconds || 0);
  const stageLabel = ending
    ? "正在生成复盘"
    : leaving
      ? "正在退出面试"
      : busy
        ? "正在生成追问"
        : listening
          ? "正在听你作答"
          : speaking
            ? "面试官正在提问"
            : interviewError
              ? "上一轮未提交，可重试"
              : interviewerReplied
                ? "面试官已追问，等待作答"
                : "等待作答";

  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas">
      <header className="relative z-20 flex h-16 shrink-0 items-center gap-4 border-b border-line px-6">
        <div className="flex min-w-0 flex-1 items-center gap-2.5 overflow-hidden">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-forest">
            <Sparkles size={16} className="text-mint-2" />
          </div>
          <div className="truncate text-sm font-semibold">{interview?.title || "模拟面试"}</div>
          <span className="flex shrink-0 items-center gap-1.5 rounded-full bg-forest/15 px-2 py-0.5 text-[11px] text-mint-3">
            <span className="h-1.5 w-1.5 rounded-full bg-mint" />
            进行中
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <span className="text-[11px] text-dim">已用时</span>
          <span className="font-mono text-xl font-semibold">{formatElapsed(elapsed)}</span>
        </div>
        <button
          type="button"
          onClick={() => setConfirmLeave(true)}
          disabled={ending || leaving}
          className="relative z-10 flex shrink-0 cursor-pointer items-center gap-1.5 rounded-lg border border-line-strong bg-row px-4 py-2 text-xs disabled:cursor-wait disabled:opacity-50"
        >
          <LogOut size={14} className="text-mute" />
          {leaving ? "正在退出…" : "退出"}
        </button>
        <button
          type="button"
          onClick={end}
          disabled={ending || leaving}
          className="relative z-10 flex shrink-0 cursor-pointer items-center gap-1.5 rounded-lg border border-line-strong bg-row px-4 py-2 text-xs disabled:cursor-wait disabled:opacity-50"
        >
          <Square size={14} className="text-danger" />
          {ending ? "正在生成复盘…" : "结束面试并生成复盘"}
        </button>
      </header>

      <div className="relative min-h-0 flex-1 overflow-hidden">
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-[22px] pb-20 sm:pb-0">
          <div className="interview-strands-stage relative h-[240px] w-[min(680px,60vw)]" aria-hidden="true">
            {/* Existing speech state controls shader energy without changing recording or playback behavior. */}
            <Strands
              colors={["#F97316", "#10B981", "#06B6D4"]}
              count={3}
              speed={busy ? 0.32 : listening ? 0.62 : speaking ? 0.5 : 0.38}
              amplitude={1}
              waviness={1}
              thickness={0.7}
              glow={busy ? 2.1 : 2.6}
              taper={3}
              spread={1}
              intensity={busy ? 0.68 : 0.82}
              saturation={1.5}
              opacity={1}
              scale={1.5}
              className={`interview-strands interview-strands--${listening ? "listening" : speaking ? "speaking" : busy ? "thinking" : "idle"}`}
            />
          </div>
          <div className="text-[13px] font-medium text-mint-3">{stageLabel}</div>
          <BarVisualizer state={busy ? "thinking" : listening ? "listening" : speaking ? "speaking" : "idle"} />
        </div>

        <div
          ref={conversationRef}
          aria-label="面试对话记录"
          aria-live="polite"
          className="interview-conversation absolute right-4 top-4 flex w-[min(420px,calc(100vw-32px))] flex-col gap-4 overflow-y-auto sm:right-12 sm:top-6 sm:max-h-[calc(100%-220px)] sm:w-[min(420px,42vw)]"
        >
          {visibleTurns.length ? visibleTurns.map((turn) => (
            <article key={turn.id} className={`interview-turn interview-turn--${turn.role}`}>
              <div className="mb-1.5 text-[11px] font-medium">
                {turn.role === "interviewer" ? `面试官${speaking && turn.id === lastInterviewer?.id ? " · 正在提问" : ""}` : "你"}
              </div>
              <p className="whitespace-pre-wrap text-sm leading-6">{turn.content}</p>
            </article>
          )) : (
            <article className="interview-turn interview-turn--interviewer">
              <div className="mb-1.5 text-[11px] font-medium text-mint">面试官 · {speaking ? "正在提问" : "字幕"}</div>
              <p className="interview-subtitle-text">{prompt || "题目将随提问出现"}</p>
            </article>
          )}
          {listening || Boolean(interviewError) ? (
            <div className="interview-turn interview-turn--user space-y-2">
              <div className="flex items-center gap-1.5 text-[11px] font-medium text-mint-2">
                <span className="h-1.5 w-1.5 rounded-full bg-mint" />
                实时转写
              </div>
              <p className="text-sm leading-6 text-ink-2">{listening ? draft || "正在聆听…" : draft || interviewError}</p>
            </div>
          ) : null}
          {busy && !streamingTurn ? (
            <div className="interview-turn interview-turn--interviewer interview-turn--processing space-y-2" aria-live="polite">
              <div className="flex items-center gap-1.5 text-[11px] font-medium">
                <span className="h-1.5 w-1.5 rounded-full bg-mint" />
                面试官
              </div>
              <p className="text-sm leading-6 text-ink-2">正在处理你的回答…</p>
            </div>
          ) : null}
        </div>

        <div className="absolute bottom-3 left-1/2 flex w-[min(420px,calc(100vw-24px))] -translate-x-1/2 flex-col items-center gap-2 sm:bottom-8 sm:gap-2.5">
          <div className={`interview-answer-shell w-full ${textMode ? "interview-answer-shell--open" : ""}`} aria-hidden={!textMode}>
            <div className="apple-input-shell interview-answer-box flex w-full items-end gap-2 rounded-xl border border-field-line bg-elevated px-3 py-2">
              {/* Multiline text input keeps longer interview answers readable before submission. */}
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(undefined, "text"); }
                }}
                disabled={busy || ending || leaving}
                rows={4}
                placeholder="输入文字作答，Enter 发送；Shift + Enter 换行"
                className="interview-answer-input scrollbar-hide min-h-[104px] w-full resize-none bg-transparent text-sm leading-6 outline-none placeholder:text-faint"
              />
            </div>
          </div>
          <div className="interview-control-track">
            <button onClick={() => {
              if (listening) recogRef.current?.stop();
              else interruptSpeech();
            }} className="interview-control-button flex h-10 items-center justify-center gap-1.5 text-xs text-mute" disabled={ending || leaving}>
              <Hand size={14} />
              打断
            </button>
            <div className={`interview-mic-wrap ${listening ? "interview-mic-wrap--listening" : ""}`}>
              <div className="interview-mic-meter" aria-hidden="true">
                {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map((bar) => (
                  <span key={bar} style={{ "--meter-index": bar } as React.CSSProperties} />
                ))}
              </div>
              <button
              onClick={toggleMic}
              disabled={busy || ending || leaving}
              className={`interview-mic flex h-16 w-16 items-center justify-center rounded-full border border-glow/30 bg-forest disabled:cursor-wait disabled:opacity-60 ${listening ? "interview-mic--listening" : busy ? "interview-mic--busy" : ""}`}
              aria-label={busy ? "正在生成追问" : listening ? "停止录音" : "语音作答"}
              aria-pressed={listening}
            >
                {listening ? <Square size={19} className="text-mint-4" /> : <Mic size={24} className="text-mint-4" />}
              </button>
            </div>
            <button
              onClick={() => {
                interruptSpeech();
                setTextMode((v) => !v);
              }}
              className="interview-control-button flex h-10 items-center justify-center gap-1.5 text-xs text-mute"
            >
              <Keyboard size={14} />
              {textMode ? "语音作答" : "文字作答"}
            </button>
          </div>
          {interviewError && draft ? (
            <button type="button" onClick={() => void send(draft, textMode ? "text" : "voice")} disabled={busy} className="text-xs text-mint-3 disabled:opacity-50">
              重试提交已识别的回答
            </button>
          ) : null}
          <p className="text-[11px] text-faint">浏览器端语音识别 / TTS  ·  可随时打断</p>
        </div>
      </div>
      <ConfirmDialog
        open={confirmLeave}
        title="退出面试"
        body="退出后这场面试回到未开始状态，不生成复盘。下次可以重新开始。"
        confirmLabel="确认退出"
        busyLabel="正在退出…"
        busy={leaving}
        onCancel={() => {
          if (!leaving) setConfirmLeave(false);
        }}
        onConfirm={() => void leave()}
      />
      {reportReadyNotice ? (
        <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/50 px-4" role="dialog" aria-modal="true">
          <div className="apple-material apple-dialog w-[420px] max-w-full border border-line bg-card p-6">
            <div className="text-base font-semibold text-ink">面试已结束</div>
            <p className="mt-2 text-sm leading-6 text-mute">正在准备生成复盘报告，请稍后回来查看。</p>
            <div className="mt-6 flex justify-end">
              <button
                type="button"
                onClick={() => navigate("/interview", { replace: true })}
                className="apple-button apple-button--primary apple-button--sm"
              >
                返回模拟面试
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
