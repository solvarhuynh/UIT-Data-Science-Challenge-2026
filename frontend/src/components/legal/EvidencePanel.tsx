import { useState } from "react";
import { Bookmark, ChevronDown, ExternalLink, FileText, Hash, X } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { savedSourceFromCitation, savedSourceFromHit } from "@/lib/legal/saved";
import type { Citation, QueryResponse, RetrievalHit, SavedLegalSource } from "@/lib/legal/types";

/* ------------------------------ evidence card ----------------------------- */

interface EvidenceCardProps {
  title: string;
  docNumber?: string;
  locator: string;
  excerpt: string;
  score: number;
  source: string;
  rank: number;
  active?: boolean;
  onLocate?: () => void;
  saved?: boolean;
  onToggleSaved?: () => void;
}

export function EvidenceCard({
  title,
  docNumber,
  locator,
  excerpt,
  score,
  source,
  rank,
  active,
  onLocate,
  saved = false,
  onToggleSaved,
}: EvidenceCardProps) {
  const [open, setOpen] = useState(false);

  return (
    <article
      className={cn(
        "legal-record relative overflow-hidden border bg-card p-3.5 pt-4 transition-colors",
        active
          ? "border-[var(--flag)] bg-[color-mix(in_oklab,var(--flag)_7%,var(--card))]"
          : "border-border hover:border-[color-mix(in_oklab,var(--brand)_50%,var(--border))]",
      )}
    >
      <div className="absolute inset-x-0 top-0 flex h-[3px]" aria-hidden>
        <span className="w-1/4 bg-[var(--flag)]" />
        <span className="flex-1 bg-[var(--brand)]" />
      </div>
      <p className="mb-2 text-[9.5px] font-semibold tracking-[0.18em] text-muted-foreground uppercase">
        Văn bản pháp luật Việt Nam
      </p>
      <div className="flex items-start gap-2">
        <span className="mt-0.5 inline-flex size-5 shrink-0 items-center justify-center rounded-[4px] bg-surface-2 font-mono text-[10.5px] font-semibold text-muted-foreground">
          {rank}
        </span>
        <div className="min-w-0 flex-1">
          <h4 className="font-serif text-[15px] leading-snug font-semibold text-foreground">
            {title}
          </h4>
          <p className="mt-0.5 text-[11.5px] text-muted-foreground">
            {locator}
            {docNumber ? ` · ${docNumber}` : ""}
          </p>
        </div>
        <span className="shrink-0 border border-border bg-background px-1.5 py-0.5 font-mono text-[10.5px] tabular-nums text-muted-foreground">
          {score.toFixed(2)}
        </span>
      </div>

      <blockquote
        className={cn(
          "mt-2.5 border-l-2 pl-3 font-serif text-[13px] leading-[1.7] text-foreground/85",
          active
            ? "border-[var(--flag)]"
            : "border-[color-mix(in_oklab,var(--brand)_70%,transparent)]",
          !open && "line-clamp-3",
        )}
      >
        {excerpt}
      </blockquote>

      <div className="mt-2.5 flex items-center gap-1.5">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="inline-flex items-center gap-1 text-[11.5px] text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
        >
          <ChevronDown
            className={cn("size-3.5 transition-transform", open && "rotate-180")}
            aria-hidden
          />
          {open ? "Thu gọn" : "Xem đầy đủ"}
        </button>

        {onToggleSaved && (
          <button
            type="button"
            onClick={onToggleSaved}
            aria-label={saved ? "Bỏ lưu nguồn" : "Lưu nguồn"}
            aria-pressed={saved}
            className={cn(
              "inline-flex size-7 items-center justify-center rounded-md transition-colors",
              saved
                ? "bg-[color-mix(in_oklab,var(--brand)_14%,transparent)] text-[var(--brand)]"
                : "text-muted-foreground hover:bg-accent hover:text-foreground",
            )}
          >
            <Bookmark className={cn("size-3.5", saved && "fill-current")} aria-hidden />
          </button>
        )}

        {onLocate && (
          <button
            type="button"
            onClick={onLocate}
            className="ml-auto inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-[11.5px] font-medium text-foreground transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
          >
            <ExternalLink className="size-3" aria-hidden />
            Xem trong câu trả lời
          </button>
        )}
      </div>

      <div className="mt-2 flex min-w-0 items-center gap-1.5 border-t border-border/70 pt-2">
        <span className="shrink-0 text-[9.5px] font-semibold tracking-[0.1em] text-muted-foreground uppercase">
          Nguồn
        </span>
        <p className="truncate font-mono text-[10px] text-muted-foreground/70" title={source}>
          {source}
        </p>
      </div>
    </article>
  );
}

/* ---------------------------- retrieval details --------------------------- */

export function RetrievalDetails({ response }: { response: QueryResponse | null }) {
  if (!response) {
    return (
      <p className="px-1 py-6 text-center text-[13px] text-muted-foreground">
        Chưa có thông tin xử lý.
      </p>
    );
  }
  const rows: [string, string][] = [
    ["prompt_version", response.prompt_version],
    ["trace_id", response.trace_id],
    ["cache_hit", String(response.cache_hit)],
    ["citations", String(response.citations.length)],
    ["retrieval_hits", String(response.retrieval_hits.length)],
    ["data_source", response.mock ? "local_mock_corpus" : "backend"],
  ];
  return (
    <div className="space-y-4">
      <div>
        <p className="mb-2 text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">
          Độ trễ pipeline
        </p>
        <div className="space-y-2">
          {Object.entries(response.latency_ms).map(([k, v]) => (
            <div key={k} className="space-y-1">
              <div className="flex justify-between text-[12px]">
                <span className="font-mono text-muted-foreground">{k}</span>
                <span className="font-mono tabular-nums text-foreground">{v} ms</span>
              </div>
              <div className="h-1 w-full overflow-hidden rounded-full bg-surface-2">
                <div
                  className="h-full rounded-full bg-[var(--brand)]"
                  style={{
                    width: `${Math.min(100, (v / response.latency_ms.total_ms) * 100)}%`,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="border-t border-border pt-3">
        <p className="mb-2 text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">
          Siêu dữ liệu
        </p>
        <dl className="space-y-1.5 text-[12px]">
          {rows.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-3">
              <dt className="font-mono text-muted-foreground">{k}</dt>
              <dd className="truncate font-mono text-foreground">{v}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}

/* ------------------------------ evidence panel ---------------------------- */

export type EvidenceTab = "citations" | "hits" | "meta";

interface EvidencePanelProps {
  citations: Citation[];
  hits: RetrievalHit[];
  response: QueryResponse | null;
  activeCitation: string | null;
  onSelectCitation?: (id: string) => void;
  onClose: () => void;
  savedSourceIds: ReadonlySet<string>;
  onToggleSaved: (source: SavedLegalSource) => void;
  activeTab: EvidenceTab;
  onTabChange: (tab: EvidenceTab) => void;
  advanced?: boolean;
}

export function EvidencePanelContent({
  citations,
  hits,
  response,
  activeCitation,
  onSelectCitation,
  onClose,
  savedSourceIds,
  onToggleSaved,
  activeTab,
  onTabChange,
  advanced = false,
}: EvidencePanelProps) {
  return (
    <div className="flex h-full flex-col bg-surface">
      <div className="flex h-14 shrink-0 items-center gap-2 border-b border-border px-4">
        <FileText className="size-4 text-[var(--brand)]" aria-hidden />
        <h2 className="text-[13.5px] font-semibold text-foreground">Căn cứ pháp lý</h2>
        <button
          type="button"
          onClick={onClose}
          aria-label="Đóng bảng căn cứ pháp lý"
          className="ml-auto inline-flex size-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
        >
          <X className="size-4" aria-hidden />
        </button>
      </div>

      <Tabs
        value={activeTab}
        onValueChange={(value) => onTabChange(value as EvidenceTab)}
        className="flex min-h-0 flex-1 flex-col"
      >
        <div className="px-3 pt-3">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="citations" className="whitespace-nowrap text-[12px]">
              Trích dẫn
            </TabsTrigger>
            <TabsTrigger value="hits" className="whitespace-nowrap text-[12px]">
              Truy hồi
            </TabsTrigger>
            <TabsTrigger value="meta" className="whitespace-nowrap text-[12px]">
              Xử lý
            </TabsTrigger>
          </TabsList>
        </div>

        <div className="scroll-slim min-h-0 flex-1 overflow-y-auto px-3 py-3">
          <TabsContent value="citations" className="mt-0 space-y-2.5">
            {citations.length === 0 ? (
              <p className="px-1 py-8 text-center text-[13px] text-muted-foreground">
                Chưa có nguồn trích dẫn. Hãy đặt một câu hỏi pháp luật.
              </p>
            ) : (
              citations.map((c) => {
                const savedSource = savedSourceFromCitation(c);
                return (
                  <div key={c.id} id={`evidence-${c.id}`}>
                    <EvidenceCard
                      title={c.law_name}
                      docNumber={c.doc_number}
                      locator={[c.article, c.clause, c.point].filter(Boolean).join(" · ")}
                      excerpt={c.excerpt}
                      score={c.score}
                      source={c.source}
                      rank={c.rank}
                      active={activeCitation === c.id}
                      saved={savedSourceIds.has(savedSource.id)}
                      onToggleSaved={() => onToggleSaved(savedSource)}
                      onLocate={onSelectCitation ? () => onSelectCitation(c.id) : undefined}
                    />
                  </div>
                );
              })
            )}
          </TabsContent>

          <TabsContent value="hits" className="mt-0 space-y-2.5">
            {hits.length === 0 ? (
              <p className="px-1 py-8 text-center text-[13px] text-muted-foreground">
                Chưa có kết quả truy hồi.
              </p>
            ) : (
              hits.map((h) => {
                const savedSource = savedSourceFromHit(h);
                return (
                  <div key={h.chunk_id} className="space-y-1.5">
                    <EvidenceCard
                      title={h.law_name}
                      docNumber={h.metadata.doc_number}
                      locator={[h.chapter, h.article, h.clause, h.point]
                        .filter(Boolean)
                        .join(" · ")}
                      excerpt={h.text}
                      score={h.final_score}
                      source={h.source}
                      rank={h.rank}
                      saved={savedSourceIds.has(savedSource.id)}
                      onToggleSaved={() => onToggleSaved(savedSource)}
                    />
                    {advanced && (
                      <div className="grid grid-cols-4 gap-1.5 px-1 font-mono text-[10px] text-muted-foreground">
                        <span>dense {h.dense_score.toFixed(2)}</span>
                        <span>sparse {h.sparse_score.toFixed(2)}</span>
                        <span>hybrid {h.hybrid_score.toFixed(2)}</span>
                        <span>rerank {h.rerank_score.toFixed(2)}</span>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </TabsContent>

          <TabsContent value="meta" className="mt-0">
            <RetrievalDetails response={response} />
          </TabsContent>
        </div>
      </Tabs>

      <div className="flex shrink-0 items-center gap-1.5 border-t border-border px-4 py-2.5 text-[11px] text-muted-foreground">
        <Hash className="size-3" aria-hidden />
        Mỗi kết luận đều gắn với điều, khoản cụ thể trong văn bản gốc.
      </div>
    </div>
  );
}
