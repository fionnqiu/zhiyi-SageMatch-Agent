import { useEffect, useState, type ReactNode } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Check, ChevronDown, LoaderCircle, Trash2, X } from "lucide-react";
import { api, type Provider, type RoleBinding } from "../../../api";
import { ConfirmDialog } from "../../../components/feedback/ConfirmDialog";
import { Select } from "../../../components/forms/Select";
import { notify } from "../../../lib/feedback/notify";

type ProviderForm = {
  name: string;
  protocol: string;
  capability: string;
  base_url: string;
  api_key: string;
  notes: string;
};

// 新增时不预选协议或能力，避免没看清就存成 Anthropic / LLM。
const EMPTY_FORM: ProviderForm = {
  name: "",
  protocol: "",
  capability: "",
  base_url: "",
  api_key: "",
  notes: "",
};

export function AdminProvidersPage() {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [roles, setRoles] = useState<RoleBinding[]>([]);
  const [modal, setModal] = useState<"create" | Provider | null>(null);
  const [pendingDelete, setPendingDelete] = useState<Provider | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [pinging, setPinging] = useState<string | null>(null);
  const [pinged, setPinged] = useState<string | null>(null);

  async function load() {
    const [p, r] = await Promise.all([api.providers(), api.roles()]);
    setProviders(p);
    setRoles(r);
  }

  useEffect(() => {
    load().catch((err) => notify(err instanceof Error ? err.message : "加载失败", "error"));
  }, []);

  return (
    <div className="flex h-full flex-col bg-canvas">
      <header className="flex h-12 items-center justify-between border-b border-line px-6 text-xs">
        <span className="font-medium text-ink-2">模型供应商接入</span>
      </header>
      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-10 py-6">
        <section className="admin-card space-y-3 rounded-xl border border-line bg-card p-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-semibold">模型与语音供应商接入</h2>
            <button onClick={() => setModal("create")} className="admin-action rounded-lg bg-forest px-3 py-1.5 text-xs text-mint-4">
              新增供应商
            </button>
          </div>
          {providers.map((p) => (
            <div key={p.id} className="flex items-center justify-between gap-3 rounded-lg border border-line bg-elevated px-3 py-2">
              <div className="min-w-0 flex-1">
                <div className="text-xs text-ink-2">{p.name}</div>
                <div className="truncate text-[10px] text-dim">
                  {p.base_url || "未填 Base URL"} · {protocolLabel(p.protocol)} · {p.key_masked || "无密钥"}
                </div>
              </div>
              <button
                disabled={pinging === p.id}
                onClick={async () => {
                  setPinging(p.id);
                  setPinged(null);
                  try {
                    await api.pingProvider(p.id);
                    await load();
                    setPinged(p.id);
                    window.setTimeout(() => setPinged((cur) => (cur === p.id ? null : cur)), 1600);
                  } catch (err) {
                    notify(err instanceof Error ? err.message : "连通失败", "error");
                    await load();
                  } finally {
                    setPinging(null);
                  }
                }}
                className="admin-action inline-flex h-7 w-[84px] items-center justify-center gap-1 rounded-md border border-forest-2/40 px-2 text-[11px] text-mint disabled:opacity-70"
              >
                {pinging === p.id ? (
                  <LoaderCircle size={13} className="sage-spin" />
                ) : pinged === p.id ? (
                  <Check size={13} />
                ) : (
                  "连通测试"
                )}
              </button>
              <button
                onClick={() => setModal(p)}
                className="admin-action rounded-md border border-line-strong px-2 py-1 text-[11px] text-ink-2"
              >
                编辑
              </button>
              <button
                onClick={() => setPendingDelete(p)}
                className="admin-action rounded-md border border-line-strong px-2 py-1 text-[11px] text-danger"
              >
                删除
              </button>
              <div className={`w-28 text-right text-[11px] ${p.status === "error" ? "text-danger" : "text-mint"}`}>
                {p.status === "configured" ? "✓ 已配置" : p.status === "deferred" ? "○ 后置" : p.status === "error" ? "连通失败" : p.status}
                {p.latency_ms != null ? ` · ${p.latency_ms}ms` : ""}
              </div>
            </div>
          ))}
        </section>
        <section className="admin-card space-y-3 rounded-xl border border-line bg-card p-4">
          <h2 className="text-xs font-semibold">Agent 角色路由</h2>
          {roles.map((role) => {
            if (role.role === "speech") {
              // 语音是唯一需要两套模型的角色：ASR 听用户，TTS 念面试官，供应商可以不是同一家。
              return (
                <SpeechRoleRow key={role.role} role={role} providers={providers} onSaved={load} />
              );
            }
            const wantCap = "llm";
            const fallback =
              providers.find((p) => p.capability === wantCap && (p.models || []).length) ||
              providers.find((p) => p.capability === wantCap) ||
              providers[0];
            const bound = providers.find((p) => p.id === role.provider_id) || fallback;
            const catalog = bound?.models || [];
            const modelOptions = Array.from(
              new Set([...(catalog || []), ...(role.model ? [role.model] : [])].filter(Boolean)),
            );
            const providerValue = bound?.id || role.provider_id || "";
            return (
              <div key={role.role} className="grid grid-cols-[1.2fr_1fr_1fr] items-center gap-2 rounded-lg border border-line bg-elevated px-3 py-2 text-[11px]">
                <div className="text-ink-2">{role.label}</div>
                <Select
                  aria-label={`${role.label} 供应商`}
                  value={providerValue}
                  options={providers
                    .filter((p) => p.capability === wantCap || p.id === providerValue)
                    .map((p) => ({ value: p.id, label: p.name }))}
                  onChange={(providerId) => {
                    const next = providers.find((p) => p.id === providerId);
                    const nextModel = next?.models?.[0] || "";
                    api.updateRole(role.role, { provider_id: providerId, model: nextModel }).then(load);
                  }}
                />
                <Select
                  aria-label={`${role.label} 模型`}
                  value={modelOptions.includes(role.model) ? role.model : modelOptions[0] || ""}
                  placeholder="先在供应商里勾选模型"
                  options={modelOptions.map((m) => ({ value: m, label: m }))}
                  onChange={(model) => api.updateRole(role.role, { model }).then(load)}
                />
              </div>
            );
          })}
        </section>
      </div>
      <AnimatePresence>
      {modal ? (
        <ProviderModal
          mode={modal === "create" ? "create" : "edit"}
          provider={modal === "create" ? null : modal}
          onClose={() => setModal(null)}
          onError={(msg) => notify(msg, "error")}
          onRequestDelete={() => {
            if (modal && modal !== "create") setPendingDelete(modal);
          }}
          onSaved={async () => {
            setModal(null);
            await load();
          }}
        />
      ) : null}
      </AnimatePresence>
      <ConfirmDialog
        open={pendingDelete !== null}
        title="删除供应商"
        body={
          <>
            <div>确认删除「{pendingDelete?.name || "该供应商"}」？此操作不可撤销。</div>
            <div className="mt-3 rounded-lg border border-line bg-elevated px-3 py-2 text-[11px] text-dim">
              {(() => {
                const bound = roles.filter(
                  (r) => r.provider_id === pendingDelete?.id || r.tts_provider_id === pendingDelete?.id,
                );
                return bound.length
                  ? `将解除 ${bound.length} 个角色绑定：${bound.map((r) => r.label).join("、")}`
                  : "当前没有角色绑定到该供应商";
              })()}
            </div>
          </>
        }
        confirmLabel="确认删除"
        busyLabel="删除中…"
        busy={deleting}
        onCancel={() => setPendingDelete(null)}
        onConfirm={async () => {
          if (!pendingDelete) return;
          const target = pendingDelete;
          setDeleting(true);
          try {
            await api.deleteProvider(target.id);
            setPendingDelete(null);
            if (modal && modal !== "create" && modal.id === target.id) setModal(null);
            await load();
          } catch (err) {
            notify(err instanceof Error ? err.message : "删除失败", "error");
          } finally {
            setDeleting(false);
          }
        }}
      />
    </div>
  );
}

/** 遮罩淡入，面板从 0.97 放大到 1。关闭走同一条路，保持居中。 */
function DialogFrame({
  children,
  onClose,
  panelClass,
  z,
}: {
  children: ReactNode;
  onClose: () => void;
  panelClass: string;
  z: string;
}) {
  const reduce = useReducedMotion() ?? false;
  const fade = reduce ? { duration: 0.08 } : { duration: 0.16, ease: [0.16, 1, 0.3, 1] as const };
  const panel = reduce ? { duration: 0 } : { duration: 0.2, ease: [0.16, 1, 0.3, 1] as const };
  return (
    <motion.div
      className={`fixed inset-0 ${z} flex items-center justify-center bg-black/70 px-4`}
      onClick={onClose}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={fade}
    >
      <motion.div
        className={panelClass}
        onClick={(e) => e.stopPropagation()}
        initial={reduce ? false : { opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={reduce ? { opacity: 0 } : { opacity: 0, scale: 0.97 }}
        transition={panel}
      >
        {children}
      </motion.div>
    </motion.div>
  );
}

/** 语音角色的一行：左 ASR、右 TTS，各选供应商和模型。 */
function SpeechRoleRow({
  role,
  providers,
  onSaved,
}: {
  role: RoleBinding;
  providers: Provider[];
  onSaved: () => Promise<void>;
}) {
  return (
    <div className="space-y-2 rounded-lg border border-line bg-elevated px-3 py-2 text-[11px]">
      <div className="flex items-center justify-between">
        <div className="text-ink-2">{role.label}</div>
        <div className="text-dim">ASR 与 TTS 各绑一个模型</div>
      </div>
      <VoiceSlot
        kind="asr"
        label="ASR"
        providers={providers}
        providerId={role.provider_id}
        model={role.model}
        onChange={(providerId, model) =>
          api.updateRole(role.role, { provider_id: providerId, model }).then(onSaved)
        }
      />
      <VoiceSlot
        kind="tts"
        label="TTS"
        providers={providers}
        providerId={role.tts_provider_id}
        model={role.tts_model || ""}
        onChange={(providerId, model) =>
          api.updateRole(role.role, { tts_provider_id: providerId, tts_model: model }).then(onSaved)
        }
      />
    </div>
  );
}

function VoiceSlot({
  kind,
  label,
  providers,
  providerId,
  model,
  onChange,
}: {
  kind: "asr" | "tts";
  label: string;
  providers: Provider[];
  providerId?: string | null;
  model: string;
  onChange: (providerId: string, model: string) => void;
}) {
  const pool = providers.filter((p) => p.capability === kind || p.id === providerId);
  const fallback = pool.find((p) => (p.models || []).length) || pool[0];
  const bound = providers.find((p) => p.id === providerId) || fallback;
  const providerValue = bound?.id || providerId || "";
  const modelOptions = Array.from(new Set([...(bound?.models || []), ...(model ? [model] : [])]));
  return (
    <div className="grid grid-cols-[3rem_1fr_1fr] items-center gap-2">
      <div className="text-dim">{label}</div>
      <Select
        aria-label={`${label} 供应商`}
        value={providerValue}
        placeholder={`选择 ${label} 供应商`}
        options={pool.map((p) => ({ value: p.id, label: p.name }))}
        onChange={(nextId) => {
          const next = providers.find((p) => p.id === nextId);
          onChange(nextId, next?.models?.[0] || "");
        }}
      />
      <Select
        aria-label={`${label} 模型`}
        value={modelOptions.includes(model) ? model : modelOptions[0] || ""}
        placeholder="先在供应商里勾选模型"
        options={modelOptions.map((m) => ({ value: m, label: m }))}
        onChange={(nextModel) => {
          if (providerValue) onChange(providerValue, nextModel);
        }}
      />
    </div>
  );
}

function ProviderModal({
  mode,
  provider,
  onClose,
  onSaved,
  onError,
  onRequestDelete,
}: {
  mode: "create" | "edit";
  provider: Provider | null;
  onClose: () => void;
  onSaved: () => Promise<void>;
  onError: (msg: string) => void;
  onRequestDelete: () => void;
}) {
  const [form, setForm] = useState<ProviderForm>(() =>
    provider
      ? {
          name: provider.name,
          protocol: provider.protocol,
          capability: provider.capability,
          base_url: provider.base_url,
          api_key: "",
          notes: provider.notes || "",
        }
      : EMPTY_FORM,
  );
  const [selected, setSelected] = useState<string[]>(() => provider?.models || []);
  const [catalog, setCatalog] = useState<string[]>(() => provider?.models || []);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [hint, setHint] = useState("");
  const [modelDraft, setModelDraft] = useState("");

  async function loadModels(openAfter = false) {
    if (!form.protocol) {
      setHint("请先选择协议");
      return;
    }
    if (!form.base_url.trim()) {
      setHint("请先填写 Base URL");
      return;
    }
    if (!form.api_key.trim() && !provider?.has_key) {
      setHint("请先填写 API Key");
      return;
    }
    setLoadingModels(true);
    setHint("");
    try {
      const data = await api.probeModels({
        protocol: form.protocol,
        base_url: form.base_url.trim(),
        api_key: form.api_key.trim() || undefined,
        provider_id: provider?.id,
      });
      const names = data.models || [];
      setCatalog(Array.from(new Set([...names, ...selected])));
      // 填完地址和密钥会后台拉目录，但不主动撑开下拉，避免挡住表单。
      if (openAfter) setOpen(true);
      setHint(names.length ? `已获取 ${names.length} 个模型` : "未返回模型");
    } catch (err) {
      setHint(err instanceof Error ? err.message : "获取模型失败");
    } finally {
      setLoadingModels(false);
    }
  }

  useEffect(() => {
    // WebSocket 音频协议没有 /models，其余能力（含 ASR / TTS 的 HTTP 网关）照常拉目录。
    if (!form.protocol || form.protocol.startsWith("websocket")) return;
    if (!form.base_url.trim()) return;
    if (!form.api_key.trim() && !provider?.has_key) return;
    const t = setTimeout(() => {
      loadModels().catch(() => undefined);
    }, 500);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [form.base_url, form.api_key, form.protocol]);

  function toggle(name: string) {
    setSelected((cur) => (cur.includes(name) ? cur.filter((x) => x !== name) : [...cur, name]));
  }

  // WebSocket 语音网关往往没有 /models，手填的模型名同样进入已选列表。
  function addManualModel() {
    const name = modelDraft.trim();
    if (!name) return;
    setCatalog((cur) => (cur.includes(name) ? cur : [...cur, name]));
    setSelected((cur) => (cur.includes(name) ? cur : [...cur, name]));
    setModelDraft("");
    setHint(`已添加 ${name}`);
  }

  // 手填项不在远端目录里，只取消勾选还会留在下拉中，所以标签上的删除要两处一起清掉。
  function removeModel(name: string) {
    setSelected((cur) => cur.filter((item) => item !== name));
    setCatalog((cur) => cur.filter((item) => item !== name));
  }

  async function submit() {
    setBusy(true);
    onError("");
    if (!form.name.trim() || !form.capability) {
      onError("请填写名称，并选择能力");
      return;
    }
    const protocol = form.capability === "llm" ? "openai_chat" : form.protocol;
    if (!protocol) {
      onError("请选择协议");
      return;
    }
    const payload = {
      name: form.name.trim(),
      protocol,
      capability: form.capability,
      base_url: form.base_url.trim(),
      models: selected,
      notes: form.notes,
      ...(form.api_key.trim() ? { api_key: form.api_key.trim() } : {}),
    };
    try {
      if (mode === "create") await api.createProvider(payload);
      else if (provider) await api.updateProvider(provider.id, payload);
      await onSaved();
    } catch (err) {
      onError(err instanceof Error ? err.message : "保存失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <DialogFrame onClose={onClose} z="z-50" panelClass="w-[560px] rounded-2xl border border-line bg-card p-5">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold text-ink">{mode === "create" ? "新增供应商" : "编辑供应商"}</div>
            <div className="text-[11px] text-dim">密钥只在保存时提交，列表里仅脱敏展示</div>
          </div>
          <button onClick={onClose} className="flex h-7 w-7 items-center justify-center rounded-md text-dim" aria-label="关闭">
            <X size={16} />
          </button>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Field label="名称">
            <input className="admin-input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          </Field>
          <Field label="协议">
            {form.capability === "llm" || !form.capability ? (
              <div className="admin-input flex items-center text-mute">OpenAI Chat Completions</div>
            ) : (
              <Select
                aria-label="协议"
                value={form.protocol}
                placeholder="选择协议"
                onChange={(protocol) => setForm({ ...form, protocol })}
                options={[
                  { value: "openai_chat", label: "OpenAI Chat Completions" },
                  { value: "websocket_audio", label: "WebSocket Audio" },
                ]}
              />
            )}
          </Field>
          <Field label="能力">
            <Select
              aria-label="能力"
              value={form.capability}
              placeholder="选择能力"
              onChange={(capability) =>
                setForm({
                  ...form,
                  capability,
                  // 文本对话没有第二种协议。切到 LLM 时直接写死，避免下拉里还能改。
                  protocol: capability === "llm" ? "openai_chat" : form.protocol,
                })
              }
              options={[
                { value: "llm", label: "LLM" },
                { value: "asr", label: "ASR" },
                { value: "tts", label: "TTS" },
              ]}
            />
          </Field>
          <Field label="Base URL">
            <input className="admin-input" value={form.base_url} onChange={(e) => setForm({ ...form, base_url: e.target.value })} />
          </Field>
          <Field label="API Key" wide>
            <input
              className="admin-input"
              type="password"
              placeholder={provider?.has_key ? "留空则保留原密钥" : "填写密钥"}
              value={form.api_key}
              onChange={(e) => setForm({ ...form, api_key: e.target.value })}
            />
          </Field>
          <Field label="可用模型" wide>
            <input
              className="admin-input"
              placeholder="输入模型名，回车添加"
              value={modelDraft}
              onChange={(e) => setModelDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key !== "Enter") return;
                e.preventDefault();
                addManualModel();
              }}
            />
            <div className="relative">
              <div
                role="button"
                tabIndex={0}
                onClick={() => {
                  if (!catalog.length) loadModels(true);
                  else setOpen((v) => !v);
                }}
                onKeyDown={(e) => {
                  if (e.key !== "Enter" && e.key !== " ") return;
                  e.preventDefault();
                  if (!catalog.length) loadModels(true);
                  else setOpen((v) => !v);
                }}
                className="flex min-h-8 w-full flex-wrap items-center gap-1 rounded-lg border border-field-line bg-field px-2 py-1 text-left text-[11px]"
              >
                {selected.length ? (
                  selected.map((name) => (
                    <span
                      key={name}
                      className="inline-flex max-w-full items-center gap-1 rounded-md border border-line bg-elevated px-1.5 py-0.5 text-[10px] text-ink-2"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <span className="max-w-[160px] truncate">{name}</span>
                      <button
                        type="button"
                        aria-label={`移除 ${name}`}
                        onClick={() => removeModel(name)}
                        className="text-dim"
                      >
                        <X size={10} />
                      </button>
                    </span>
                  ))
                ) : (
                  <span className="text-faint">
                    {loadingModels ? "正在获取模型…" : "填写 Base URL 与 Key 后自动获取"}
                  </span>
                )}
                <ChevronDown size={14} className="ml-auto shrink-0 text-dim" />
              </div>
              {open ? (
                <div className="ui-dropdown-scroll absolute z-10 mt-1 max-h-48 w-full overflow-y-auto rounded-lg border border-line bg-elevated p-1">
                  {catalog.length === 0 ? (
                    <div className="px-2 py-2 text-[11px] text-dim">暂无模型，请先获取</div>
                  ) : (
                    catalog.map((name) => {
                      const on = selected.includes(name);
                      return (
                        <button
                          type="button"
                          key={name}
                          onClick={() => toggle(name)}
                          className={`flex w-full items-center justify-between rounded-md px-2 py-1.5 text-left text-[11px] ${
                            on ? "bg-forest/40 text-mint-3" : "text-ink-2 hover:bg-line"
                          }`}
                        >
                          <span className="truncate">{name}</span>
                          {on ? <span className="text-mint">✓</span> : null}
                        </button>
                      );
                    })
                  )}
                </div>
              ) : null}
            </div>
            <div className="flex items-center justify-between pt-1 text-[10px] text-dim">
              <span>{hint || "可下拉多选，也可直接输入模型名"}</span>
              <button type="button" onClick={() => loadModels()} className="text-mint">
                {loadingModels ? "获取中…" : "重新获取"}
              </button>
            </div>
          </Field>
          <Field label="备注" wide>
            <input className="admin-input" value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} />
          </Field>
        </div>
        <div className="mt-5 flex items-center justify-between">
          {provider ? (
            <button
              onClick={onRequestDelete}
              className="flex items-center gap-1 rounded-lg border border-line-strong px-3 py-2 text-xs text-danger"
            >
              <Trash2 size={12} />
              删除
            </button>
          ) : (
            <span />
          )}
          <div className="flex gap-2">
            <button onClick={onClose} className="rounded-lg border border-line-strong px-4 py-2 text-xs text-ink-2">
              取消
            </button>
            <button onClick={submit} disabled={busy} className="rounded-lg bg-forest px-4 py-2 text-xs font-semibold text-mint-4">
              {busy ? "保存中…" : "保存"}
            </button>
          </div>
        </div>
    </DialogFrame>
  );
}

function protocolLabel(protocol: string) {
  if (protocol === "openai_chat" || protocol.startsWith("openai") || protocol.startsWith("anthropic")) {
    return "OpenAI Chat Completions";
  }
  if (protocol.startsWith("websocket")) return "WebSocket Audio";
  return protocol || "未选协议";
}

function Field({ label, children, wide }: { label: string; children: ReactNode; wide?: boolean }) {
  return (
    <div className={`space-y-1 ${wide ? "col-span-2" : ""}`}>
      <div className="text-[11px] text-dim">{label}</div>
      {children}
    </div>
  );
}
