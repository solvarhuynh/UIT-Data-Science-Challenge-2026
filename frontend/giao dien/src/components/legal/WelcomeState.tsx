import { motion } from "motion/react";
import { ArrowUpRight, Scale } from "lucide-react";
import { BrandLogo, VietnamFlag } from "./BrandLogo";
import { SUGGESTED_PROMPTS } from "@/lib/legal/mock-data";

interface WelcomeStateProps {
  onPick: (prompt: string) => void;
}

export function WelcomeState({ onPick }: WelcomeStateProps) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22, ease: "easeOut" }}
      className="welcome-vietnam mx-auto h-full min-h-0 w-full max-w-[760px] px-5 text-center"
    >
      <div className="mx-auto flex w-fit flex-col items-center">
        <div className="logo-archive-frame">
          <div className="logo-archive-inner">
            <BrandLogo size={96} subtle className="welcome-logo" />
          </div>
          <VietnamFlag className="welcome-vietnam-flag" animated />
          <div
            className="vietnam-legal-mark"
            aria-label="Biểu trưng cán cân công lý và hoa sen Việt Nam"
            role="img"
          >
            <Scale className="vietnam-legal-scale" aria-hidden />
            <span className="vietnam-legal-lotus" aria-hidden>
              <i className="vietnam-lotus-petal vietnam-lotus-petal-left" />
              <i className="vietnam-lotus-petal vietnam-lotus-petal-center" />
              <i className="vietnam-lotus-petal vietnam-lotus-petal-right" />
            </span>
          </div>
        </div>
        <div className="viet-divider mt-4" aria-hidden>
          <span />
          <i />
          <span />
        </div>
        <p className="welcome-kicker mt-2.5 text-[10.5px] font-semibold tracking-[0.18em] text-[var(--flag)] uppercase">
          HCMUTE-SHIPCODE · Hỗ trợ tra cứu pháp lý
        </p>
      </div>

      <h2 className="welcome-title mt-3.5 font-serif text-[28px] leading-tight font-semibold tracking-[-0.02em] text-foreground sm:text-[36px]">
        Tra cứu pháp luật Việt Nam
      </h2>
      <p className="welcome-description mx-auto mt-2.5 max-w-full whitespace-nowrap text-[14px] leading-6 font-medium tracking-[0.01em] text-muted-foreground sm:text-[15px]">
        Chuẩn điều luật – Trọn niềm tin
      </p>

      <div className="welcome-provenance" aria-label="Phạm vi văn bản">
        <span>Luật</span>
        <i />
        <span>Nghị định</span>
        <i />
        <span>Thông tư</span>
        <i />
        <span>Điều · Khoản · Điểm</span>
      </div>

      <div className="welcome-suggestions mx-auto mt-5 max-w-[600px] text-left">
        <p className="mb-2 px-1 text-[10.5px] font-semibold tracking-[0.16em] text-muted-foreground uppercase">
          Tra cứu nhanh
        </p>
        <ul className="divide-y divide-border border-y border-border/90 bg-background/45">
          {SUGGESTED_PROMPTS.map((p, index) => (
            <li key={p}>
              <button
                type="button"
                onClick={() => onPick(p)}
                className="group flex w-full items-center gap-3 px-2 py-3 text-left text-[14px] text-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--brand)_7%,transparent)] focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
              >
                <span className="w-7 shrink-0 font-serif text-[13px] font-semibold tabular-nums text-[var(--flag)]">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span className="h-4 w-px bg-border transition-colors group-hover:bg-[var(--brand)]" />
                <span className="min-w-0 flex-1">{p}</span>
                <ArrowUpRight
                  className="size-4 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100"
                  aria-hidden
                />
              </button>
            </li>
          ))}
        </ul>
      </div>
    </motion.section>
  );
}
