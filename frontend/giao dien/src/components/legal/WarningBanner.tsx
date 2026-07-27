import { AlertTriangle, Info, RefreshCw, WifiOff } from "lucide-react";
import { cn } from "@/lib/utils";

export type WarningTone = "warning" | "info" | "error" | "offline";

interface WarningBannerProps {
  tone?: WarningTone;
  title: string;
  description?: string;
  onRetry?: () => void;
  className?: string;
}

const toneStyles: Record<WarningTone, string> = {
  warning:
    "border-[color-mix(in_oklab,var(--brand)_55%,var(--border))] bg-[color-mix(in_oklab,var(--brand)_12%,var(--surface))]",
  info: "border-border bg-surface",
  error:
    "border-[color-mix(in_oklab,var(--flag)_45%,var(--border))] bg-[color-mix(in_oklab,var(--flag)_10%,var(--surface))]",
  offline: "border-border bg-surface-2",
};

export function WarningBanner({
  tone = "warning",
  title,
  description,
  onRetry,
  className,
}: WarningBannerProps) {
  const Icon = tone === "offline" ? WifiOff : tone === "info" ? Info : AlertTriangle;
  return (
    <div
      role="status"
      className={cn(
        "flex items-start gap-3 rounded-md border px-3.5 py-3 text-sm",
        toneStyles[tone],
        className,
      )}
    >
      <Icon
        className={cn(
          "mt-0.5 size-4 shrink-0",
          tone === "error" ? "text-[var(--flag)]" : "text-muted-foreground",
        )}
        aria-hidden
      />
      <div className="min-w-0 flex-1">
        <p className="font-medium text-foreground">{title}</p>
        {description && <p className="mt-1 leading-relaxed text-muted-foreground">{description}</p>}
      </div>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
        >
          <RefreshCw className="size-3.5" aria-hidden />
          Thử lại
        </button>
      )}
    </div>
  );
}
