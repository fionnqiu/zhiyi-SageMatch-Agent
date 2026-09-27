import { request } from "../core/http";

export type CallLog = {
  id: string;
  role: string;
  provider_name: string;
  model: string;
  status: string;
  latency_ms: number;
  error?: string | null;
  created_at: string;
};

export type AuditLog = {
  id: string;
  actor: string;
  action: string;
  target: string;
  detail?: Record<string, unknown> | null;
  created_at: string;
};

export const auditApi = {
  callLogs: () => request<CallLog[]>("/api/admin/logs/calls"),
  auditLogs: () => request<AuditLog[]>("/api/admin/logs/audit"),
  exportLogs: () => fetch("/api/admin/logs/export-errors").then(async (response) => {
    if (!response.ok) throw new Error("导出错误日志失败");
    return response.blob();
  }),
};
