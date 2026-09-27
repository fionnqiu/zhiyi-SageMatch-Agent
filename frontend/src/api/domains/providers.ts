import { request } from "../core/http";

export type Provider = {
  id: string;
  name: string;
  protocol: string;
  base_url: string;
  capability: string;
  status: string;
  latency_ms?: number | null;
  models: string[];
  notes: string;
  key_masked: string;
  has_key: boolean;
};

export type RoleBinding = {
  id: string;
  role: string;
  label: string;
  provider_id?: string | null;
  model: string;
  /** 语音角色的 TTS 槽；ASR 仍用 provider_id / model。 */
  tts_provider_id?: string | null;
  tts_model?: string;
  temperature: number;
};

export const providerApi = {
  adminOverview: () => request<Record<string, unknown>>("/api/admin/overview"),
  providers: () => request<Provider[]>("/api/admin/providers"),
  createProvider: (body: Partial<Provider> & { api_key?: string }) =>
    request<Provider>("/api/admin/providers", { method: "POST", body: JSON.stringify(body) }),
  updateProvider: (id: string, body: Partial<Provider> & { api_key?: string }) =>
    request<Provider>(`/api/admin/providers/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  pingProvider: (id: string) => request<Provider>(`/api/admin/providers/${id}/ping`, { method: "POST" }),
  deleteProvider: (id: string) => request<{ ok: string }>(`/api/admin/providers/${id}/delete`, { method: "POST" }),
  probeModels: (body: { protocol: string; base_url: string; api_key?: string; provider_id?: string }) =>
    request<{ models: string[] }>("/api/admin/provider-models", { method: "POST", body: JSON.stringify(body) }),
  roles: () => request<RoleBinding[]>("/api/admin/roles"),
  updateRole: (role: string, body: Partial<RoleBinding>) =>
    request<RoleBinding>(`/api/admin/roles/${role}`, { method: "PATCH", body: JSON.stringify(body) }),
};
