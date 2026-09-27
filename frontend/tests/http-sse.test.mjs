import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(new URL("../src/api/core/http.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText;
const { readEventStream } = await import(`data:text/javascript,${encodeURIComponent(compiled)}`);

function stream(chunks, headers = {}) {
  return new Response(new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(new TextEncoder().encode(chunk));
      controller.close();
    },
  }), { headers });
}

test("replays from the last ID after a dropped CRLF stream without duplicate deltas", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url, init });
    return calls.length === 1
      ? stream(["id: run-1:1\r", "\ndata: {\"type\":\"delta\",\"text\":\"a\"}\r\n\r\n"])
      : stream(["id: run-1:1\ndata: {\"type\":\"delta\",\"text\":\"a\"}\n\n",
        "id: run-1:2\ndata: {\"type\":\"done\"}\n\n"]);
  };
  try {
    const events = [];
    await readEventStream("/api/chat/stream", {}, (event) => events.push(event.type));
    assert.deepEqual(events, ["delta", "done"]);
    assert.equal(calls[1].url, "/api/streams/run-1/events");
    assert.equal(calls[1].init.headers["Last-Event-ID"], "run-1:1");
    assert.equal(calls[1].init.method, undefined);
  } finally { globalThis.fetch = originalFetch; }
});

test("a durable error frame stops recovery and preserves its message", async () => {
  const originalFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    return stream(["id: run-2:1\ndata: {\"type\":\"error\",\"message\":\"provider failed\"}\n\n"]);
  };
  try {
    await assert.rejects(readEventStream("/api/chat/stream", {}, () => {}), /provider failed/);
    assert.equal(calls, 1);
  } finally { globalThis.fetch = originalFetch; }
});

test("an incomplete JSON tail reconnects from the last complete event", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url, init });
    return calls.length === 1
      ? stream(["id: run-3:1\ndata: {\"type\":\"delta\",\"text\":\"a\"}\n\n",
        "id: run-3:2\ndata: {\"type\":\"delta\",\"text\":\""])
      : stream(["id: run-3:2\ndata: {\"type\":\"done\"}\n\n"]);
  };
  try {
    const events = [];
    await readEventStream("/api/chat/stream", {}, (event) => events.push(event.type));
    assert.deepEqual(events, ["delta", "done"]);
    assert.equal(calls.length, 2);
    assert.equal(calls[1].init.headers["Last-Event-ID"], "run-3:1");
  } finally { globalThis.fetch = originalFetch; }
});

test("a response header recovers a chat stream lost before its first frame", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url, init });
    return calls.length === 1
      ? stream([], { "X-Run-ID": "header-run" })
      : stream(["id: header-run:1\ndata: {\"type\":\"done\"}\n\n"]);
  };
  try {
    const events = [];
    await readEventStream("/api/chat/stream", {}, (event) => events.push(event.type));
    assert.deepEqual(events, ["done"]);
    assert.equal(calls.length, 2);
    assert.equal(calls[1].url, "/api/streams/header-run/events");
    assert.equal(calls[1].init.method, undefined);
  } finally { globalThis.fetch = originalFetch; }
});

test("a consumer exception propagates without replaying the event", async () => {
  const originalFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    return stream(["id: tenant:run-3:1\ndata: {\"type\":\"delta\",\"text\":\"a\"}\n\n"]);
  };
  try {
    await assert.rejects(readEventStream("/api/chat/stream", {}, () => {
      throw new Error("consumer failed");
    }), /consumer failed/);
    assert.equal(calls, 1);
  } finally { globalThis.fetch = originalFetch; }
});
