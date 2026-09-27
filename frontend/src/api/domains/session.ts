import { readEventStream, request, sendFile } from "../core/http";

// 输入框里暂存的附件。text 只在发送时交给模型，气泡不展示。

export type ActivityTodo = {
  id: string;
  label: string;
  status?: "pending" | "active" | "complete" | "cancelled";
};

export type ClarificationQuestion = {
  id: string;
  prompt: string;
  options: { id: string; label: string }[];
};

export type ClarificationAnswer = {
  id: string;
  option_id: string;
  label: string;
  prompt: string;
};

export type ChatAttachment = {
  name: string;
  size: number;
  text?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  extra?: {
    kind?: string;
    question_set_id?: string;
    questions?: { stem?: string; ordinal?: number; id?: string; prompt?: string; options?: { id: string; label: string }[] }[];
    actions?: string[];
    todos?: ActivityTodo[];
    answers?: ClarificationAnswer[];
    files?: { name: string; size: number }[];
    thinking?: string;
    reasoning?: string;
  } | null;
  created_at: string;
};

export type ChatStreamEvent =
  | { type: "blocked" }
  | { type: "redirect" }
  | { type: "meta"; session_id: string; extra?: ChatMessage["extra"] }
  | { type: "thinking"; text: string }
  | { type: "reasoning"; text: string }
  | { type: "delta"; text: string }
  | { type: "reset" }
  | { type: "error"; message: string; retryable?: boolean }
  | { type: "done"; session: ChatSession };

export type ChatSession = {
  id: string;
  title: string;
  job_title?: string | null;
  created_at: string;
  updated_at: string;
  messages?: ChatMessage[];
};

export const sessionApi = {
  sessions: () => request<ChatSession[]>("/api/sessions"),
  session: (id: string) => request<ChatSession>(`/api/sessions/${id}`),
  createSession: () => request<ChatSession>("/api/sessions", { method: "POST" }),
  clearSession: (id: string) => request<ChatSession>(`/api/sessions/${id}/clear`, { method: "POST" }),
  deleteSession: (id: string) => request<{ ok: string }>(`/api/sessions/${id}`, { method: "DELETE" }),
  chat: (
    content: string,
    sessionId?: string,
    answers?: ClarificationAnswer[],
    signal?: AbortSignal,
    attachments?: ChatAttachment[],
  ) =>
    request<ChatSession>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ content, session_id: sessionId, answers: answers ?? [], attachments: attachments ?? [] }),
      signal,
    }),
  prepareChatFile: (file: File) => sendFile<ChatAttachment>("/api/chat/prepare", file),
  streamChat: (
    content: string,
    sessionId: string | undefined,
    answers: ClarificationAnswer[] | undefined,
    attachments: ChatAttachment[] | undefined,
    onEvent: (event: ChatStreamEvent) => void,
    signal?: AbortSignal,
  ) =>
    readEventStream(
      "/api/chat/stream",
      { content, session_id: sessionId, answers: answers ?? [], attachments: attachments ?? [] },
      (event) => onEvent(event as ChatStreamEvent),
      signal,
    ),
};
