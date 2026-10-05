/** Shared fetch helpers. Callers pass a path; JSON vs multipart is chosen here. */

function responseError(text: string, statusText: string): Error {
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    if (typeof parsed.detail === "string" && parsed.detail) return new Error(parsed.detail);
  } catch {
    // Proxies and unhandled server errors can return plain text instead of the API JSON shape.
  }
  if (/^internal server error$/i.test(text.trim())) return new Error("服务器内部错误，请稍后重试");
  return new Error(text || statusText);
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const res = await fetch(path, {
    ...init,
    // Caller headers such as Idempotency-Key must not replace the JSON media
    // type required by FastAPI's request-body parser.
    headers,
  });
  if (!res.ok) {
    const text = await res.text();
    throw responseError(text, res.statusText);
  }
  if (res.status === 204) return undefined as T;
  const ctype = res.headers.get("content-type") || "";
  if (!ctype.includes("json")) return (await res.text()) as T;
  return res.json() as Promise<T>;
}

export async function sendFile<T>(path: string, file: File, extra?: Record<string, string>): Promise<T> {
  const body = new FormData();
  body.append("file", file);
  if (extra) {
    for (const [k, v] of Object.entries(extra)) body.append(k, v);
  }
  const res = await fetch(path, { method: "POST", body });
  if (!res.ok) {
    const text = await res.text();
    throw responseError(text, res.statusText);
  }
  return res.json() as Promise<T>;
}

export async function readEventStream(
  path: string,
  body: unknown,
  onEvent: (event: Record<string, unknown>) => void,
  signal?: AbortSignal,
  headers?: Record<string, string>,
  retryInitial = false,
  idleTimeoutMs = 30_000,
): Promise<void> {
  let runId = "";
  let lastId = "";
  let terminal = false;
  const post = () => fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream", ...headers },
    body: JSON.stringify(body), signal,
  });
  let response: Response;
  try { response = await post(); }
  catch (error) {
    if (!retryInitial || signal?.aborted) throw error;
    response = await post();
  }
  for (let attempt = 0; attempt <= 5 && !terminal; attempt++) {
    if (!response.ok || !response.body) {
      throw responseError(await response.text(), response.statusText);
    }
    runId ||= response.headers.get("X-Run-ID") || "";
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let serverError = false;
    let callbackError = false;
    const consume = (frame: string) => {
      const lines = frame.split("\n");
      const id = lines.find((line) => line.startsWith("id:"))?.slice(3).trim() || "";
      const data = lines.filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart()).join("\n");
      if (!data) return;
      // A replay may overlap the previous response; sequence IDs are monotonic per run.
      const idSeparator = id.lastIndexOf(":");
      const lastSeparator = lastId.lastIndexOf(":");
      if (idSeparator > 0 && lastSeparator > 0 &&
          id.slice(0, idSeparator) === lastId.slice(0, lastSeparator) &&
          Number(id.slice(idSeparator + 1)) <= Number(lastId.slice(lastSeparator + 1))) return;
      const event = JSON.parse(data) as Record<string, unknown>;
      if (!runId && typeof event.run_id === "string") runId = event.run_id;
      if (id) {
        lastId = id;
        runId ||= id.slice(0, id.lastIndexOf(":"));
      }
      try { onEvent(event); }
      catch (error) { callbackError = true; throw error; }
      if (["done", "blocked", "redirect"].includes(String(event.type))) terminal = true;
      if (event.type === "error") {
        serverError = true;
        throw new Error(String(event.message || "流式请求失败"));
      }
    };
    try {
      while (!terminal) {
        // A half-open proxy/worker connection can leave read() pending forever.
        // Bound each wait so replay recovery gets a chance to take over and the
        // UI cannot remain stuck in its busy state indefinitely.
        let idleTimer: ReturnType<typeof setTimeout> | undefined;
        const idleTimeout = new Promise<never>((_, reject) => {
          idleTimer = setTimeout(() => reject(new Error("SSE idle timeout")), idleTimeoutMs);
        });
        let chunk: ReadableStreamReadResult<Uint8Array>;
        try {
          chunk = await Promise.race([reader.read(), idleTimeout]);
        } finally {
          if (idleTimer !== undefined) clearTimeout(idleTimer);
        }
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        buffer = buffer.replace(/\r\n/g, "\n");
        let boundary = buffer.indexOf("\n\n");
        while (boundary >= 0) {
          consume(buffer.slice(0, boundary));
          buffer = buffer.slice(boundary + 2);
          if (terminal) break;
          boundary = buffer.indexOf("\n\n");
        }
      }
      // Flush TextDecoder at EOF so a split final UTF-8 sequence is emitted
      // before parsing the trailing SSE frame; otherwise the last character
      // can disappear when the provider closes immediately after its answer.
      buffer += decoder.decode();
      buffer = buffer.replace(/\r\n/g, "\n");
      if (!terminal && buffer.trim()) {
        try { consume(buffer.replace(/\r/g, "\n").trim()); }
        catch (error) {
          // EOF can bisect an unfinished frame; replay starts after the last
          // fully parsed ID. Framed malformed JSON still fails above.
          if (!(error instanceof SyntaxError)) throw error;
        }
      }
    } catch (error) {
      if (serverError || callbackError || signal?.aborted || error instanceof SyntaxError) throw error;
    } finally {
      await reader.cancel().catch(() => undefined);
    }
    if (terminal) break;
    if (!runId) throw new Error("连接已中断，请重试");
    if (attempt === 5) throw new Error("连接恢复超时，请稍后重试");
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => { signal?.removeEventListener("abort", abort); resolve(); }, Math.min(500 * 2 ** attempt, 4000));
      const abort = () => { clearTimeout(timer); reject(new DOMException("已取消", "AbortError")); };
      signal?.addEventListener("abort", abort, { once: true });
      if (signal?.aborted) abort();
    });
    for (let connectionAttempt = 0; ; connectionAttempt++) {
      try {
        response = await fetch(`/api/streams/${encodeURIComponent(runId)}/events`, {
          headers: { Accept: "text/event-stream", ...(lastId ? { "Last-Event-ID": lastId } : {}) }, signal,
        });
        break;
      } catch (error) {
        if (signal?.aborted || connectionAttempt === 2) throw error;
        await new Promise((resolve) => setTimeout(resolve, 500 * (connectionAttempt + 1)));
      }
    }
  }
}
