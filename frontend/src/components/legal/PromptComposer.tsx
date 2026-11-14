import { useEffect, useRef } from "react";
import {
  ArrowUp,
  Scale,
  SlidersHorizontal,
  BookOpen,
  GitCompare,
  MessagesSquare,
} from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { QueryMode, RetrievalSettings } from "@/lib/legal/types";

const QUERY_MODES: { id: QueryMode; label: string; icon: typeof Scale }[] = [
  { id: "qa", label: "Hỏi đáp", icon: MessagesSquare },
  { id: "lookup", label: "Tra cứu", icon: BookOpen },
  { id: "compare", label: "So sánh", icon: GitCompare },
];

interface QueryModeSelectorProps {
  value: QueryMode;
  onChange: (mode: QueryMode) => void;
  className?: string;
}

export function QueryModeSelector({ value, onChange, className }: QueryModeSelectorProps) {
  return (
    <div
      role="radiogroup"
      aria-label="Chế độ truy vấn"
      className={cn("inline-flex items-center gap-0.5 rounded-full bg-surface-2/80 p-1", className)}
    >
      {QUERY_MODES.map((m) => {
        const active = m.id === value;
        return (
          <button
            key={m.id}
            role="radio"
            aria-checked={active}
            aria-label={m.label}
            type="button"
            onClick={() => onChange(m.id)}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full px-2 py-1.5 text-[11.5px] font-medium transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none sm:px-2.5",
              active
                ? "bg-card text-foreground shadow-[0_1px_3px_oklch(0_0_0/0.08)]"
                : "text-muted-foreground hover:bg-card/55 hover:text-foreground",
            )}
          >
            <m.icon className="size-3.5" aria-hidden />
            <span className="hidden sm:inline">{m.label}</span>
          </button>
        );
      })}
    </div>
  );
}

interface PromptComposerProps {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  mode: QueryMode;
  onModeChange: (m: QueryMode) => void;
  settings: RetrievalSettings;
  onSettingsChange: (s: RetrievalSettings) => void;
  busy?: boolean;
  autoFocus?: boolean;
  inputRef?: React.RefObject<HTMLTextAreaElement | null>;
}

export function PromptComposer({
  value,
  onChange,
  onSubmit,
  mode,
  onModeChange,
  settings,
  onSettingsChange,
  busy = false,
  inputRef,
}: PromptComposerProps) {
  const localRef = useRef<HTMLTextAreaElement>(null);
  const ref = inputRef ?? localRef;
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    if (!value) {
      el.style.height = "48px";
      return;
    }
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value, ref]);

  return (
    <div
      className={cn(
        "rounded-[24px] border border-border/90 bg-card/95 shadow-[var(--shadow-lift)] transition-[border-color,box-shadow] focus-within:border-[color-mix(in_oklab,var(--brand)_62%,var(--border))] focus-within:shadow-[0_14px_38px_-24px_color-mix(in_oklab,var(--ink)_48%,transparent)]",
      )}
    >
      <label htmlFor="legal-prompt" className="sr-only">
        Câu hỏi pháp luật
      </label>
      <textarea
        id="legal-prompt"
        ref={ref}
        rows={1}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            onSubmit();
          }
        }}
        placeholder="Nhập câu hỏi pháp luật…"
        className="scroll-slim max-h-40 min-h-12 w-full resize-none overflow-x-hidden bg-transparent px-5 pt-4 pb-1.5 text-[15px] leading-relaxed text-foreground placeholder:text-muted-foreground/65 focus:outline-none"
      />

      <div className="flex items-center gap-1 px-2.5 pt-1 pb-2.5">
        <Popover>
          <PopoverTrigger asChild>
            <button
              type="button"
              aria-label="Cài đặt truy hồi"
              className="inline-flex size-9 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-surface-2 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
            >
              <SlidersHorizontal className="size-4" aria-hidden />
            </button>
          </PopoverTrigger>
          <PopoverContent
            align="start"
            className="scroll-slim w-[min(20rem,calc(100vw-2rem))] space-y-4 overflow-y-auto"
            style={{
              maxHeight: "min(calc(100dvh - 2rem), var(--radix-popover-content-available-height))",
            }}
          >
            <div>
              <p className="text-[13px] font-semibold text-foreground">Cài đặt truy hồi</p>
              <p className="mt-0.5 text-[11.5px] text-muted-foreground">
                Điều chỉnh phạm vi văn bản được sử dụng làm căn cứ.
              </p>
            </div>

            <div className="space-y-1.5">
              <span className="text-[12px] font-medium text-foreground">Số lượng nguồn</span>
              <div className="flex gap-1.5">
                {([5, 10, 20] as const).map((k) => (
                  <button
                    key={k}
                    type="button"
                    onClick={() => onSettingsChange({ ...settings, topK: k })}
                    className={cn(
                      "flex-1 rounded-md border px-2 py-1.5 text-[12px] font-medium transition-colors",
                      settings.topK === k
                        ? "border-[var(--brand)] bg-[color-mix(in_oklab,var(--brand)_18%,transparent)] text-foreground"
                        : "border-border text-muted-foreground hover:bg-accent",
                    )}
                  >
                    {k}
                  </button>
                ))}
              </div>
            </div>

            {(
              [
                ["lawName", "Lọc theo tên luật", "VD: Bộ luật Lao động 2019"],
                ["docId", "Lọc theo số văn bản", "VD: 45/2019/QH14"],
                ["article", "Lọc theo điều", "VD: Điều 35"],
              ] as const
            ).map(([key, label, ph]) => (
              <div key={key} className="space-y-1.5">
                <label
                  htmlFor={`filter-${key}`}
                  className="text-[12px] font-medium text-foreground"
                >
                  {label}
                </label>
                <input
                  id={`filter-${key}`}
                  value={settings[key]}
                  onChange={(e) => onSettingsChange({ ...settings, [key]: e.target.value })}
                  placeholder={ph}
                  className="w-full rounded-md border border-input bg-background px-2.5 py-1.5 text-[12.5px] text-foreground placeholder:text-muted-foreground/60 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                />
              </div>
            ))}

            <div className="flex items-center justify-between border-t border-border pt-3">
              <div>
                <p className="text-[12px] font-medium text-foreground">Chế độ nâng cao</p>
                <p className="text-[11px] text-muted-foreground">Hiển thị điểm hybrid & rerank</p>
              </div>
              <Switch
                checked={settings.advanced}
                onCheckedChange={(v) => onSettingsChange({ ...settings, advanced: v })}
                aria-label="Chế độ nâng cao"
              />
            </div>
          </PopoverContent>
        </Popover>

        <QueryModeSelector value={mode} onChange={onModeChange} />

        <button
          type="button"
          onClick={onSubmit}
          disabled={busy || !value.trim()}
          aria-label="Gửi câu hỏi"
          className="ml-auto inline-flex size-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-sm transition-[transform,filter,opacity] hover:-translate-y-0.5 hover:brightness-105 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none active:translate-y-0 disabled:cursor-not-allowed disabled:opacity-35 disabled:hover:translate-y-0"
        >
          <ArrowUp className="size-4" aria-hidden />
        </button>
      </div>
    </div>
  );
}
