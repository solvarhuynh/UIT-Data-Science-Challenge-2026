import { FOLLOW_UPS, MOCK_HITS, MOCK_RESPONSE } from "./mock-data";
import type { QueryRequest, QueryResponse, RetrievalHit } from "./types";

/**
 * Isolated API layer. Swap API_BASE / USE_MOCK to connect the real backend.
 * The UI never talks to fetch() directly.
 */
export const API_BASE = "/api/v1";
export const REQUEST_TIMEOUT_MS = 20_000;

export type PipelineStage =
  "analyzing" | "retrieving" | "matching" | "composing" | "streaming" | "done";

export const STAGE_LABELS: Record<Exclude<PipelineStage, "done" | "streaming">, string> = {
  analyzing: "Đang xác định lĩnh vực pháp lý…",
  retrieving: "Đang tra cứu kho văn bản Việt Nam…",
  matching: "Đang đối chiếu điều · khoản · điểm…",
  composing: "Đang soạn trả lời kèm căn cứ…",
};

export interface StreamHandlers {
  onStage: (stage: PipelineStage) => void;
  onHit: (hit: RetrievalHit) => void;
  onToken: (partialAnswer: string) => void;
  signal?: AbortSignal;
}

export function buildRequest(question: string, opts?: Partial<QueryRequest>): QueryRequest {
  return {
    question,
    top_k: 10,
    top_n: 5,
    filters: null,
    prompt_version: "legal_qa_v1",
    debug: false,
    ...opts,
  };
}

async function postQuery(req: QueryRequest, signal?: AbortSignal): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) throw new Error(`Backend error ${res.status}`);
  return (await res.json()) as QueryResponse;
}

const wait = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      reject(new DOMException("Aborted", "AbortError"));
    });
  });

function isOffline() {
  return typeof navigator !== "undefined" && navigator.onLine === false;
}

/**
 * Runs the retrieval pipeline. Attempts the real backend first, then falls
 * back to local mock data so the app is always demo-ready.
 * Ready to be swapped for SSE token streaming: replace the mock chunk loop
 * with an EventSource / ReadableStream reader emitting onToken().
 */
export async function runLegalQuery(
  req: QueryRequest,
  handlers: StreamHandlers,
): Promise<QueryResponse> {
  const { onStage, onHit, onToken, signal } = handlers;

  if (isOffline()) {
    const err = new Error("offline");
    err.name = "OfflineError";
    throw err;
  }

  onStage("analyzing");
  await wait(520, signal);

  let response: QueryResponse | null = null;
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2500);
    signal?.addEventListener("abort", () => controller.abort());
    response = await postQuery(req, controller.signal);
    clearTimeout(timer);
  } catch {
    response = null; // fall back to local mock corpus
  }

  onStage("retrieving");
  const hits = (response?.retrieval_hits?.length ? response.retrieval_hits : MOCK_HITS).slice(
    0,
    req.top_k,
  );
  for (const hit of hits.slice(0, req.top_n)) {
    await wait(230, signal);
    onHit(hit);
  }

  onStage("matching");
  await wait(560, signal);
  onStage("composing");
  await wait(420, signal);

  const final: QueryResponse = response ?? {
    ...MOCK_RESPONSE,
    prompt_version: req.prompt_version,
    mock: true,
  };

  onStage("streaming");
  const words = final.answer.split(" ");
  let acc = "";
  for (let i = 0; i < words.length; i += 4) {
    acc += (acc ? " " : "") + words.slice(i, i + 4).join(" ");
    onToken(acc);
    await wait(18, signal);
  }
  onToken(final.answer);
  onStage("done");
  return final;
}

export function suggestFollowUps(_question: string): string[] {
  return FOLLOW_UPS;
}
