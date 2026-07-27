import { cn } from "@/lib/utils";
import shipcodeLogo from "../../../LOGO SHIPCODE UI.png";

interface BrandLogoProps {
  size?: number;
  className?: string;
  /** Reduces contrast for decorative placements (welcome state watermark). */
  subtle?: boolean;
}

/** Official HCMUTE-SHIPCODE logo. Never redraw, recolor or reproportion it. */
export function BrandLogo({ size = 34, className, subtle = false }: BrandLogoProps) {
  return (
    <img
      src={shipcodeLogo}
      alt="Logo đội HCMUTE-SHIPCODE"
      width={size}
      height={size}
      className={cn(
        "shrink-0 select-none rounded-[3px] bg-white object-contain",
        subtle && "opacity-95",
        className,
      )}
      style={{ width: size, height: size }}
      draggable={false}
    />
  );
}

interface VietnamFlagProps {
  className?: string;
  decorative?: boolean;
}

/** Compact, code-native rendering of the Vietnamese national flag. */
export function VietnamFlag({ className, decorative = false }: VietnamFlagProps) {
  return (
    <span
      className={cn("vietnam-flag", className)}
      role={decorative ? undefined : "img"}
      aria-label={decorative ? undefined : "Cờ Việt Nam"}
      aria-hidden={decorative || undefined}
    >
      <span className="vietnam-flag-star" aria-hidden />
    </span>
  );
}

interface BrandLockupProps {
  collapsed?: boolean;
  size?: number;
}

export function BrandLockup({ collapsed = false, size = 34 }: BrandLockupProps) {
  return (
    <div className={cn("brand-lockup flex min-w-0 items-center", collapsed && "justify-center")}>
      <span className="brand-logo-shell">
        <BrandLogo size={size} />
      </span>
      {!collapsed && (
        <div className="brand-lockup-copy min-w-0 flex-1">
          <div className="truncate text-[13px] leading-none font-bold tracking-[0.025em] text-foreground">
            HCMUTE-SHIPCODE
          </div>
          <div className="brand-lockup-subtitle mt-1.5 flex min-w-0 items-center gap-1.5">
            <span className="brand-lockup-rule" aria-hidden />
            <span className="truncate text-[9px] font-semibold tracking-[0.105em] text-muted-foreground uppercase">
              Pháp luật Việt Nam
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
