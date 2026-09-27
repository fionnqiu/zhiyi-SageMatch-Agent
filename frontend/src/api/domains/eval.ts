import { request } from "../core/http";

export type EvalRun = {
  id: string;
  kind: string;
  status: string;
  input_text: string;
  metrics?: Record<string, unknown> | null;
  detail?: Record<string, unknown> | null;
  created_at: string;
};

export const evalApi = {
  evalRuns: () => request<EvalRun[]>("/api/admin/eval"),
  evalQuestions: (jobText: string) =>
    request<EvalRun>("/api/admin/eval/questions", { method: "POST", body: JSON.stringify({ job_text: jobText }) }),
  evalScores: (interviewId?: string) =>
    request<EvalRun>("/api/admin/eval/scores", {
      method: "POST",
      body: JSON.stringify({ interview_id: interviewId, repeats: 5 }),
    }),
};
