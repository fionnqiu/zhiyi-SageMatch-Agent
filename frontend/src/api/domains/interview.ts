import { readEventStream, request } from "../core/http";

type PendingAnswer = { key: string; content: string; mode: "text" | "voice"; previousAnswerId?: string };
const pendingAnswers = new Map<string, PendingAnswer>();
const pendingGenerationKeys = new Map<string, string>();
const generationStorageKey = "sagematch:pending-generation";

export function pendingInterviewGeneration(): { content: string; key: string } | null {
  try {
    const raw = sessionStorage.getItem(generationStorageKey);
    const value = raw ? JSON.parse(raw) as { content?: unknown; key?: unknown } : null;
    if (typeof value?.content === "string" && typeof value.key === "string") return value as { content: string; key: string };
  } catch { /* Storage may be disabled. */ }
  return null;
}

function answerStorageKey(id: string): string {
  return `sagematch:pending-answer:${id}`;
}

export function pendingInterviewAnswer(id: string): PendingAnswer | null {
  const cached = pendingAnswers.get(id);
  if (cached) return cached;
  try {
    const raw = sessionStorage.getItem(answerStorageKey(id));
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    const answer = parsed as Partial<PendingAnswer>;
    if (typeof answer.key !== "string" || typeof answer.content !== "string" ||
      (answer.mode !== "text" && answer.mode !== "voice")) return null;
    if (answer.previousAnswerId !== undefined && typeof answer.previousAnswerId !== "string") return null;
    const pending = answer as PendingAnswer;
    pendingAnswers.set(id, pending);
    return pending;
  } catch {
    // Storage may be disabled; the in-memory copy still handles retries in this tab.
    return null;
  }
}

export function prepareInterviewAnswer(id: string, content: string, mode: "text" | "voice", previousAnswerId?: string): PendingAnswer {
  const previous = pendingInterviewAnswer(id);
  if (previous?.content === content && previous.mode === mode) return previous;
  // The key identifies exactly one payload. Editing a retry must create a new key.
  const pending = { key: crypto.randomUUID(), content, mode, previousAnswerId };
  pendingAnswers.set(id, pending);
  try { sessionStorage.setItem(answerStorageKey(id), JSON.stringify(pending)); } catch { /* Storage is optional. */ }
  return pending;
}

function confirmInterviewAnswer(id: string, key: string): void {
  if (pendingInterviewAnswer(id)?.key !== key) return;
  pendingAnswers.delete(id);
  try { sessionStorage.removeItem(answerStorageKey(id)); } catch { /* Storage is optional. */ }
}

export function reconcileInterviewAnswer(id: string, turns: InterviewTurn[] | undefined): PendingAnswer | null {
  const pending = pendingInterviewAnswer(id);
  if (!pending) return null;
  const latest = [...(turns || [])].reverse().find((turn) => turn.role === "user");
  // A committed server turn is authoritative after a lost HTTP response. The
  // baseline excludes an identical answer from the previous question.
  if (latest && latest.id !== pending.previousAnswerId && latest.cite !== "answer_pending" &&
    latest.content === pending.content && latest.answer_mode === pending.mode) {
    confirmInterviewAnswer(id, pending.key);
    return null;
  }
  return pending;
}

export type Question = {
  id: string;
  ordinal: number;
  stem: string;
  options: { key: string; text: string }[];
  explanation?: string | null;
};

export type InterviewTurn = {
  id: string;
  role: "interviewer" | "user";
  content: string;
  answer_mode?: string | null;
  cite?: string | null;
  question_id?: string | null;
  created_at: string;
};

export type Report = {
  id: string;
  score: number;
  review: string;
  issues: { issue: string; quote: string; advice: string }[];
  dimensions?: Record<string, { score: number; evidence: string; advice: string }> | null;
  scoring_status?: "valid" | "unavailable" | "invalid" | "legacy";
  created_at: string;
};

export type Interview = {
  id: string;
  title: string;
  status: string;
  current_question_index: number;
  started_at?: string | null;
  ended_at?: string | null;
  elapsed_seconds: number;
  tags: string[];
  summary?: string | null;
  score?: number | null;
  created_at: string;
  current_question?: Question | null;
  report?: Report | null;
  turns?: InterviewTurn[];
};

export type InterviewGenerateEvent =
  | { type: "thought"; text: string }
  | { type: "done"; interview: Pick<Interview, "id" | "title" | "status"> & Partial<Interview> }
  | { type: "error"; message: string };

export const interviewApi = {
  interviews: () => request<Interview[]>("/api/interviews"),
  interview: (id: string) => request<Interview>(`/api/interviews/${id}`),
  generateInterview: (content: string, onEvent: (event: InterviewGenerateEvent) => void, signal?: AbortSignal) => {
    const fingerprint = content.trim();
    const saved = pendingInterviewGeneration();
    let key = pendingGenerationKeys.get(fingerprint) || (saved?.content === fingerprint ? saved.key : "");
    if (!key) {
      // Keep the key until a durable done event, including across a dropped stream.
      key = crypto.randomUUID();
      pendingGenerationKeys.set(fingerprint, key);
    }
    try { sessionStorage.setItem(generationStorageKey, JSON.stringify({ content: fingerprint, key })); } catch { /* Optional storage. */ }
    return readEventStream("/api/interviews/generate/stream", { content }, (event) => {
      if (event.type === "done") {
        pendingGenerationKeys.delete(fingerprint);
        if (pendingInterviewGeneration()?.key === key) {
          try { sessionStorage.removeItem(generationStorageKey); } catch { /* Optional storage. */ }
        }
      }
      onEvent(event as InterviewGenerateEvent);
    }, signal, { "Idempotency-Key": key }, true);
  },
  startInterview: (sessionId?: string) =>
    request<Interview>("/api/interviews", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    }),
  openInterview: (id: string) => request<Interview>(`/api/interviews/${id}/start`, { method: "POST" }),
  answerInterview: async (id: string, content: string, answerMode: "text" | "voice" = "text", signal?: AbortSignal) => {
    const { key } = prepareInterviewAnswer(id, content, answerMode);
    const interview = await request<Interview>(`/api/interviews/${id}/answer`, {
      method: "POST",
      body: JSON.stringify({ content, answer_mode: answerMode }),
      headers: { "Idempotency-Key": key },
      signal,
    });
    confirmInterviewAnswer(id, key);
    return interview;
  },
  endInterview: (id: string) => request<Interview>(`/api/interviews/${id}/end`, { method: "POST" }),
  // Temporary QA hook: replace the persisted report and run the same async pipeline again.
  regenerateReport: (id: string) =>
    request<Interview>(`/api/interviews/${id}/report/regenerate`, { method: "POST" }),
  // 直接退出回到待开始，不生成复盘；后端会清理半场进度。
  abandonInterview: (id: string) => request<Interview>(`/api/interviews/${id}/abandon`, { method: "POST" }),
  deleteInterview: (id: string) => request<{ ok: string }>(`/api/interviews/${id}`, { method: "DELETE" }),
  downloadReport: (id: string) => {
    window.open(`/api/interviews/${id}/report.txt`, "_blank");
  },
};
