import { api } from "../../../api";
import { notify } from "../../../lib/feedback/notify";

/** 入库任务挂在模块上，不跟物料页的生命周期。离开页面不会取消正在进行的请求。 */

export type MaterialUploadJob = {
  running: boolean;
  done: number;
  total: number;
  current: string;
  message: string;
};

const listeners = new Set<(job: MaterialUploadJob) => void>();

let pending: File[] = [];
let skipped = 0;
let running = false;
let done = 0;
let total = 0;
let fails: string[] = [];
let current = "";
let message = "";

function snapshot(): MaterialUploadJob {
  return { running, done, total, current, message };
}

function emit() {
  const job = snapshot();
  listeners.forEach((fn) => fn(job));
}

export function subscribeMaterialUpload(fn: (job: MaterialUploadJob) => void) {
  listeners.add(fn);
  fn(snapshot());
  return () => {
    listeners.delete(fn);
  };
}

export function enqueueMaterialUpload(files: File[], skippedCount = 0) {
  if (!files.length) {
    if (skippedCount) notify("没有可入库的 PDF / MD / TXT", "error");
    return;
  }
  pending.push(...files);
  skipped += skippedCount;
  total += files.length;
  emit();
  if (!running) void drain();
}

async function drain() {
  running = true;
  message = "";
  emit();
  while (pending.length) {
    const accepted: string[] = [];
    while (pending.length) {
      const file = pending.shift();
      if (!file) break;
      current = file.name;
      emit();
      try {
        const created = await api.uploadMaterial(file);
        accepted.push(created.id);
      } catch (err) {
        fails.push(`${file.name}: ${err instanceof Error ? err.message : "失败"}`);
      }
      done += 1;
      current = "";
      emit();
    }
    if (accepted.length) {
      current = "后台入库中";
      emit();
      fails.push(...(await waitUntilSettled(accepted)));
      current = "";
    }
  }
  const finished = done;
  const failed = [...fails];
  const skippedNow = skipped;
  const summary = failed.length
    ? failed.join("；")
    : skippedNow
      ? `已入库 ${finished} 份，跳过 ${skippedNow} 个非支持格式`
      : `已入库 ${finished} 份`;
  pending = [];
  skipped = 0;
  done = 0;
  total = 0;
  fails = [];
  current = "";
  running = false;
  message = failed.length ? `${failed.length} 个文件入库失败` : summary;
  emit();
  notify(summary, failed.length ? "error" : "ok");
}

async function waitUntilSettled(ids: string[]) {
  const waiting = new Set(ids);
  const failed: string[] = [];
  for (let i = 0; i < 600 && waiting.size; i += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 1500));
    let rows: { id: string; filename: string; status: string; error?: string | null }[] = [];
    try {
      rows = await api.materials();
    } catch {
      continue;
    }
    const byId = new Map(rows.map((row) => [row.id, row]));
    for (const id of [...waiting]) {
      const row = byId.get(id);
      if (!row || row.status !== "pending") {
        waiting.delete(id);
        if (row?.status === "failed") failed.push(`${row.filename}: ${row.error || "入库失败"}`);
      }
    }
  }
  if (waiting.size) failed.push(`${waiting.size} 个文件仍在入库，可稍后在列表查看`);
  return failed;
}
