import { request, sendFile } from "../core/http";

export type Material = {
  id: string;
  filename: string;
  source: string;
  mime: string;
  status: string;
  error?: string | null;
  size_bytes: number;
  chunk_count: number;
  created_at: string;
};

export type RecallHit = {
  chunk_id: string;
  material_id: string;
  filename: string;
  ordinal: number;
  score: number;
  text: string;
};

export const knowledgeApi = {
  materials: () => request<Material[]>("/api/admin/materials"),
  uploadMaterial: (file: File) => sendFile<Material>("/api/admin/materials", file),
  deleteMaterial: (id: string) => request<{ ok: string }>(`/api/admin/materials/${id}`, { method: "DELETE" }),
  recall: (query: string) =>
    request<{ query: string; hits: RecallHit[]; index_status: string }>("/api/admin/recall", {
      method: "POST",
      body: JSON.stringify({ query }),
    }),
};
