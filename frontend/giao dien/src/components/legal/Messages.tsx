import { useMemo, useState } from "react";
import { motion } from "motion/react";
import {
  Bookmark,
  Check,
  ChevronDown,
  Copy,
  Gauge,
  Share2,
  ThumbsDown,
  ThumbsUp,
  Timer,
} from "lucide-react";
import { toast } from "sonner";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { Citation, Message, QueryResponse } from "@/lib/legal/types";
import { WarningBanner } from "./WarningBanner";

/* ------------------------------ user message ----------------------------- */

export function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex justify-start">
      <div className="w-full border-l-[3px] border-[var(--brand)] bg-surface/65 px-4 py-3.5 text-[14.5px] leading-relaxed text-foreground shadow-[inset_0_0_0_1px_var(--border)]">
        <p className="mb-1 text-[9.5px] font-semibold tracking-[0.16em] text-muted-foreground uppercase">
          Yêu cầu tra cứu
        </p>
        <p className="font-serif text-[17px] leading-snug font-medium">{content}</p>
      </div>
    </div>
  );
}

/* ----------------------------- citation link ----------------------------- */

interface CitationLinkProps {
  citation: Citation;
  messageId: string;
  anchorId: string;
  active?: boolean;
  onSelect: (id: string) => void;
}

export function CitationLink({
  citation,
  messageId,
  anchorId,
  active,
  onSelect,
}: CitationLinkProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          id={`cite-${messageId}-${citation.id}-${anchorId}`}
          data-citation-message={messageId}
          data-citation-id={citation.id}
          onClick={() => onSelect(citation.id)}
          className={cn(
            "mx-0.5 inline-flex items-baseline rounded-[3px] border px-1.5 py-px align-baseline font-sans text-[11.5px] font-medium whitespace-nowrap transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
            active
              ? "border-[var(--flag)] bg-[color-mix(in_oklab,var(--flag)_16%,transparent)] text-foreground"
              : "border-[color-mix(in_oklab,var(--brand)_55%,var(--border))] bg-[color-mix(in_oklab,var(--brand)_13%,transparent)] text-foreground hover:bg-[color-mix(in_oklab,var(--brand)_26%,transparent)]",
          )}
        >
          {citation.label}
        </button>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">
        <span className="line-clamp-3 text-[11.5px] leading-relaxed">{citation.excerpt}</span>
      </TooltipContent>
    </Tooltip>
  );
}

/* ---------------------------- markdown renderer --------------------------- */

function renderInline(
  text: string,
  citations: Citation[],
  activeCitation: string | null,
  onCitation: (id: string) => void,
  keyPrefix: string,
  messageId: string,
) {
  const nodes: React.ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*|\[\[c\d+\]\])/g;
  const parts = text.split(pattern).filter(Boolean);
  parts.forEach((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith("**") && part.endsWith("**")) {
      nodes.push(
        <strong key={key} className="font-semibold text-foreground">
          {part.slice(2, -2)}
        </strong>,
      );
    } else if (/^\[\[c\d+\]\]$/.test(part)) {
      const id = part.slice(2, -2);
      const citation = citations.find((c) => c.id === id);
      nodes.push(
        citation ? (
          <CitationLink
            key={key}
            citation={citation}
            messageId={messageId}
            anchorId={key}
            active={activeCitation === id}
            onSelect={onCitation}
          />
        ) : null,
      );
    } else {
      nodes.push(<span key={key}>{part}</span>);
    }
  });
  return nodes;
}

interface LegalMarkdownProps {
  content: string;
  citations: Citation[];
  messageId: string;
  activeCitation: string | null;
  onCitation: (id: string) => void;
}

function LegalMarkdown({
  content,
  citations,
  messageId,
  activeCitation,
  onCitation,
}: LegalMarkdownProps) {
  const blocks = useMemo(() => content.split("\n").filter((l) => l.trim().length > 0), [content]);

  const out: React.ReactNode[] = [];
  let list: string[] = [];

  const flush = (key: string) => {
    if (!list.length) return;
    out.push(
      <ul key={key}>
        {list.map((li, i) => (
          <li key={`${key}-${i}`}>
            {renderInline(li, citations, activeCitation, onCitation, `${key}-${i}`, messageId)}
          </li>
        ))}
      </ul>,
    );
    list = [];
  };

  blocks.forEach((line, idx) => {
    if (line.startsWith("- ")) {
      list.push(line.slice(2));
      return;
    }
    flush(`ul-${idx}`);
    if (line.startsWith("### ")) {
      out.push(<h3 key={idx}>{line.slice(4)}</h3>);
    } else if (idx === 0) {
      out.push(
        <div key={idx} className="mb-5 flex gap-3.5">
          <span className="rule-line w-[3px] shrink-0 rounded-full" aria-hidden />
          <p className="font-serif text-[17px] leading-[1.7] font-medium text-foreground">
            {renderInline(line, citations, activeCitation, onCitation, `lead-${idx}`, messageId)}
          </p>
        </div>,
      );
    } else {
      out.push(
        <p key={idx}>
          {renderInline(line, citations, activeCitation, onCitation, `p-${idx}`, messageId)}
        </p>,
      );
    }
  });
  flush("ul-end");

  return <div className="prose-legal text-[14.75px] text-foreground/90">{out}</div>;
}

/* ----------------------------- answer actions ---------------------------- */

interface AnswerActionsProps {
  content: string;
  onOpenEvidence: () => void;
  sourcesSaved: boolean;
  canSaveSources: boolean;
  onToggleSavedSources: () => void;
}

export function AnswerActions({
  content,
  onOpenEvidence,
  sourcesSaved,
  canSaveSources,
  onToggleSavedSources,
}: AnswerActionsProps) {
  const [copied, setCopied] = useState(false);
  const [vote, setVote] = useState<"up" | "down" | null>(null);

  const items = [
    {
      label: copied ? "Đã sao chép" : "Sao chép",
      icon: copied ? Check : Copy,
      onClick: async () => {
        await navigator.clipboard.writeText(content);
        setCopied(true);
        toast.success("Đã sao chép câu trả lời");
        setTimeout(() => setCopied(false), 1600);
      },
      active: copied,
    },
    ...(canSaveSources
      ? [
          {
            label: sourcesSaved ? "Đã lưu căn cứ" : "Lưu căn cứ",
            icon: Bookmark,
            onClick: onToggleSavedSources,
            active: sourcesSaved,
          },
        ]
      : []),
    {
      label: "Sao chép liên kết",
      icon: Share2,
      onClick: async () => {
        try {
          await navigator.clipboard.writeText(window.location.href);
          toast.success("Đã sao chép liên kết trang");
        } catch {
          toast.error("Không thể sao chép liên kết trên trình duyệt này");
        }
      },
      active: false,
    },
    {
      label: "Hữu ích",
      icon: ThumbsUp,
      onClick: () => {
        setVote("up");
        toast.success("Cảm ơn phản hồi của bạn");
      },
      active: vote === "up",
    },
    {
      label: "Chưa chính xác",
      icon: ThumbsDown,
      onClick: () => {
        setVote("down");
        onOpenEvidence();
        toast("Đã ghi nhận. Mời bạn đối chiếu căn cứ pháp lý.");
      },
      active: vote === "down",
    },
  ];

  return (
    <div className="flex flex-wrap items-center gap-1">
      {items.map((it) => (
        <Tooltip key={it.label}>
          <TooltipTrigger asChild>
            <button
              type="button"
              onClick={it.onClick}
              aria-label={it.label}
              aria-pressed={it.active}
              className={cn(
                "inline-flex size-8 items-center justify-center rounded-md transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
                it.active
                  ? "bg-[color-mix(in_oklab,var(--brand)_20%,transparent)] text-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              <it.icon className="size-4" aria-hidden />
            </button>
          </TooltipTrigger>
          <TooltipContent>{it.label}</TooltipContent>
        </Tooltip>
      ))}
    </div>
  );
}

/* ---------------------------- assistant answer ---------------------------- */

const CONFIDENCE_LABEL: Record<NonNullable<QueryResponse["confidence"]>, string> = {
  high: "Độ tin cậy cao",
  medium: "Độ tin cậy trung bình",
  low: "Độ tin cậy thấp",
};

interface AssistantAnswerProps {
  message: Message;
  streaming?: boolean;
  activeCitation: string | null;
  onCitation: (id: string) => void;
  onOpenEvidence: () => void;
  onFollowUp: (q: string) => void;
  sourcesSaved?: boolean;
  onToggleSavedSources?: () => void;
}

export function AssistantAnswer({
  message,
  streaming = false,
  activeCitation,
  onCitation,
  onOpenEvidence,
  onFollowUp,
  sourcesSaved = false,
  onToggleSavedSources = () => {},
}: AssistantAnswerProps) {
  const res = message.response;
  const citations = res?.citations ?? [];

  return (
    <motion.article
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: "easeOut" }}
      className="w-full"
      aria-live={streaming ? "polite" : undefined}
    >
      <LegalMarkdown
        content={message.content}
        citations={citations}
        messageId={message.id}
        activeCitation={activeCitation}
        onCitation={onCitation}
      />

      {streaming && (
        <span className="ml-0.5 inline-block h-4 w-[2px] animate-pulse bg-[var(--brand)] align-middle" />
      )}

      {!streaming && res && (
        <>
          {res.warnings.length > 0 && (
            <div className="mt-5 space-y-2">
              {res.warnings.map((w) => (
                <WarningBanner key={w} title={w} />
              ))}
            </div>
          )}

          {citations.length > 0 && (
            <section className="mt-7">
              <h4 className="mb-2.5 text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">
                Căn cứ trả lời
              </h4>
              <ul className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,240px),1fr))] gap-2">
                {citations.map((c) => (
                  <li key={c.id} className="h-full">
                    <button
                      type="button"
                      onClick={() => onCitation(c.id)}
                      className={cn(
                        "group h-full w-full rounded-md border bg-surface px-3 py-2.5 text-left transition-colors hover:border-[color-mix(in_oklab,var(--brand)_60%,var(--border))]",
                        activeCitation === c.id ? "border-[var(--flag)]" : "border-border",
                      )}
                    >
                      <p className="font-serif text-[13.5px] font-semibold text-foreground">
                        {c.law_name}
                      </p>
                      <p className="mt-0.5 text-[11.5px] text-muted-foreground">
                        {[c.article, c.clause, c.point].filter(Boolean).join(" · ")}
                      </p>
                      <p className="mt-1.5 line-clamp-2 text-[12px] leading-relaxed text-muted-foreground">
                        {c.excerpt}
                      </p>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}

          <div className="mt-5 flex flex-col gap-3 border-t border-border pt-3 sm:flex-row sm:items-center">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
              <span className="inline-flex items-center gap-1.5 text-[11.5px] font-medium text-foreground">
                <Gauge className="size-3.5 text-[var(--success)]" aria-hidden />
                {CONFIDENCE_LABEL[res.confidence ?? "medium"]}
              </span>
              <span className="text-[11.5px] text-muted-foreground">
                {res.retrieval_hits.length} nguồn được truy hồi
              </span>

              <Popover>
                <PopoverTrigger asChild>
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 text-[11.5px] text-muted-foreground underline-offset-4 transition-colors hover:text-foreground hover:underline focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                  >
                    <Timer className="size-3.5" aria-hidden />
                    Chi tiết kỹ thuật
                    <ChevronDown className="size-3" aria-hidden />
                  </button>
                </PopoverTrigger>
                <PopoverContent align="start" className="w-72 space-y-2 text-[12px]">
                  <p className="text-[12.5px] font-semibold text-foreground">Thông tin xử lý</p>
                  {Object.entries(res.latency_ms).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-muted-foreground">
                      <span className="font-mono">{k}</span>
                      <span className="font-mono tabular-nums text-foreground">{v} ms</span>
                    </div>
                  ))}
                  <div className="space-y-1 border-t border-border pt-2 text-muted-foreground">
                    <div className="flex justify-between">
                      <span>prompt_version</span>
                      <span className="font-mono text-foreground">{res.prompt_version}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>cache_hit</span>
                      <span className="font-mono text-foreground">{String(res.cache_hit)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>trace_id</span>
                      <span className="font-mono text-foreground">{res.trace_id}</span>
                    </div>
                    {res.mock && (
                      <div className="flex justify-between">
                        <span>data_source</span>
                        <span className="font-mono text-foreground">local_mock_corpus</span>
                      </div>
                    )}
                  </div>
                </PopoverContent>
              </Popover>
            </div>

            <div className="self-end sm:ml-auto sm:self-auto">
              <AnswerActions
                content={message.content}
                onOpenEvidence={onOpenEvidence}
                sourcesSaved={sourcesSaved}
                canSaveSources={citations.length > 0}
                onToggleSavedSources={onToggleSavedSources}
              />
            </div>
          </div>

          {message.followUps && message.followUps.length > 0 && (
            <div className="mt-5">
              <p className="mb-2 text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">
                Câu hỏi tiếp theo
              </p>
              <div className="flex flex-wrap gap-2">
                {message.followUps.map((f) => (
                  <button
                    key={f}
                    type="button"
                    onClick={() => onFollowUp(f)}
                    className="rounded-md border border-border bg-background px-3 py-1.5 text-[12.5px] text-foreground transition-colors hover:border-[color-mix(in_oklab,var(--brand)_60%,var(--border))] hover:bg-surface focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                  >
                    {f}
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </motion.article>
  );
}
