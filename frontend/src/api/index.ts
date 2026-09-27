/**
 * Flat client kept for existing pages. Methods are grouped in sibling modules
 * so a new screen imports only the business it talks to.
 */
import { auditApi } from "./domains/audit";
import { evalApi } from "./domains/eval";
import { interviewApi } from "./domains/interview";
import { knowledgeApi } from "./domains/knowledge";
import { providerApi } from "./domains/providers";
import { sessionApi } from "./domains/session";

export type { AuditLog, CallLog } from "./domains/audit";
export type { EvalRun } from "./domains/eval";
export type { Interview, InterviewGenerateEvent, InterviewTurn, Question, Report } from "./domains/interview";
export type { Material, RecallHit } from "./domains/knowledge";
export type { Provider, RoleBinding } from "./domains/providers";
export type { ActivityTodo, ChatAttachment, ChatMessage, ChatSession, ChatStreamEvent, ClarificationAnswer, ClarificationQuestion } from "./domains/session";

export const api = {
  ...sessionApi,
  ...interviewApi,
  ...knowledgeApi,
  ...providerApi,
  ...evalApi,
  ...auditApi,
};
