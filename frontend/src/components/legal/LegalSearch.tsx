import { useMemo, useState } from "react";
import { FileText, MessageSquareQuote, Search } from "lucide-react";
import { LEGAL_DOCS, type LegalDocEntry } from "@/lib/legal/mock-data";
import { cn } from "@/lib/utils";

interface LegalSearchProps {
  onOpenEvidence: (doc: LegalDocEntry) => void;
  onAsk: (question: string) => void;
}

const DOC_TYPES = ["Tất cả", "Bộ luật", "Luật", "Nghị định", "Thông tư"] as const;
const STATUSES = ["Tất cả", "Còn hiệu lực", "Sửa đổi bổ sung"] as const;

function highlight(text: string, q: string) {
  if (!q.trim()) return text;
  const parts = text.split(new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi"));
  return parts.map((p, i) =>
    p.toLowerCase() === q.toLowerCase() ? (
      <mark
        key={i}
        className="rounded-[2px] bg-[color-mix(in_oklab,var(--brand)_45%,transparent)] px-0.5 text-foreground"
      >
        {p}
      </mark>
    ) : (
      <span key={i}>{p}</span>
    ),
  );
}

export function LegalSearchResult({
  doc,
  query,
  onOpenEvidence,
  onAsk,
}: {
  doc: LegalDocEntry;
  query: string;
  onOpenEvidence: (d: LegalDocEntry) => void;
  onAsk: (q: string) => void;
}) {
  return (
    <article className="border-b border-border py-5 last:border-b-0">
      <nav aria-label="Cấu trúc văn bản" className="mb-1.5 flex flex-wrap items-center gap-1.5">
        {doc.hierarchy.map((h, i) => (
          <span key={h} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            {i > 0 && <span aria-hidden>›</span>}
            {h}
          </span>
        ))}
      </nav>

      <button
        type="button"
        onClick={() => onOpenEvidence(doc)}
        className="block text-left focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
      >
        <h3 className="font-serif text-[17px] leading-snug font-semibold text-foreground hover:underline">
          {highlight(doc.title, query)}
        </h3>
      </button>

      <p className="mt-1.5 max-w-[70ch] text-[13.5px] leading-relaxed text-muted-foreground">
        {highlight(doc.excerpt, query)}
      </p>

      <div className="mt-3 flex flex-wrap items-center gap-2 text-[11.5px]">
        <span className="rounded border border-border px-1.5 py-0.5 text-muted-foreground">
          {doc.doc_number}
        </span>
        <span
          className={cn(
            "rounded border px-1.5 py-0.5",
            doc.status === "Còn hiệu lực"
              ? "border-[color-mix(in_oklab,var(--success)_45%,var(--border))] text-[var(--success)]"
              : "border-border text-muted-foreground",
          )}
        >
          {doc.status}
        </span>
        <span className="text-muted-foreground">Ban hành {doc.issued_date}</span>

        <button
          type="button"
          onClick={() => onAsk(`${doc.title} được quy định như thế nào?`)}
          className="ml-auto inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 font-medium text-foreground transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
        >
          <MessageSquareQuote className="size-3.5" aria-hidden />
          Hỏi về văn bản này
        </button>
      </div>
    </article>
  );
}

export function LegalSearch({ onOpenEvidence, onAsk }: LegalSearchProps) {
  const [query, setQuery] = useState("");
  const [type, setType] = useState<(typeof DOC_TYPES)[number]>("Tất cả");
  const [status, setStatus] = useState<(typeof STATUSES)[number]>("Tất cả");
  const [year, setYear] = useState<string>("Tất cả");

  const years = useMemo(
    () => [
      "Tất cả",
      ...Array.from(new Set(LEGAL_DOCS.map((d) => String(d.year))))
        .sort()
        .reverse(),
    ],
    [],
  );

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    return LEGAL_DOCS.filter((d) => {
      if (type !== "Tất cả" && d.doc_type !== type) return false;
      if (status !== "Tất cả" && d.status !== status) return false;
      if (year !== "Tất cả" && String(d.year) !== year) return false;
      if (!q) return true;
      return (
        d.title.toLowerCase().includes(q) ||
        d.excerpt.toLowerCase().includes(q) ||
        d.law_name.toLowerCase().includes(q) ||
        (d.article ?? "").toLowerCase().includes(q)
      );
    });
  }, [query, type, status, year]);

  return (
    <div className="mx-auto w-full max-w-[880px] px-5 py-8">
      <header className="mb-6">
        <h2 className="font-serif text-[22px] font-semibold tracking-tight text-foreground">
          Tra cứu văn bản pháp luật
        </h2>
        <p className="mt-1 text-[13.5px] text-muted-foreground">
          Tìm theo tên luật, số hiệu, điều khoản trong kho dữ liệu đã được lập chỉ mục.
        </p>
      </header>

      <div className="relative">
        <Search
          className="pointer-events-none absolute top-1/2 left-3.5 size-4 -translate-y-1/2 text-muted-foreground"
          aria-hidden
        />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ví dụ: chấm dứt hợp đồng lao động, Điều 35, 45/2019/QH14…"
          aria-label="Tìm kiếm văn bản"
          className="w-full rounded-lg border border-border bg-card py-3 pr-4 pl-10 text-[14px] text-foreground placeholder:text-muted-foreground/70 focus-visible:border-[color-mix(in_oklab,var(--brand)_60%,var(--border))] focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
        />
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {[
          {
            label: "Loại",
            value: type,
            set: setType as (v: string) => void,
            options: DOC_TYPES as readonly string[],
          },
          {
            label: "Trạng thái",
            value: status,
            set: setStatus as (v: string) => void,
            options: STATUSES as readonly string[],
          },
          { label: "Năm", value: year, set: setYear as (v: string) => void, options: years },
        ].map((f) => (
          <label
            key={f.label}
            className="flex items-center gap-1.5 text-[12px] text-muted-foreground"
          >
            {f.label}
            <select
              value={f.value}
              onChange={(e) => f.set(e.target.value)}
              className="rounded-md border border-border bg-background px-2 py-1.5 text-[12px] text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
            >
              {f.options.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>

      <p className="mt-5 text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">
        {results.length} kết quả
      </p>

      <div className="mt-1">
        {results.length === 0 ? (
          <div className="paper flex flex-col items-center rounded-lg border border-border py-16 text-center">
            <FileText className="size-6 text-muted-foreground" aria-hidden />
            <p className="mt-3 text-[14px] font-medium text-foreground">Không tìm thấy văn bản</p>
            <p className="mt-1 text-[13px] text-muted-foreground">
              Thử rút gọn từ khóa hoặc bỏ bớt bộ lọc.
            </p>
          </div>
        ) : (
          results.map((d) => (
            <LegalSearchResult
              key={`${d.doc_id}-${d.article}`}
              doc={d}
              query={query}
              onOpenEvidence={onOpenEvidence}
              onAsk={onAsk}
            />
          ))
        )}
      </div>
    </div>
  );
}
