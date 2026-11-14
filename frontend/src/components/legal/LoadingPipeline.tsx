import { motion } from "motion/react";
import { FileSearch } from "lucide-react";
import { STAGE_LABELS, type PipelineStage } from "@/lib/legal/api";
import type { RetrievalHit } from "@/lib/legal/types";

interface LoadingPipelineProps {
  stage: PipelineStage;
  hits: RetrievalHit[];
}

const ORDER: PipelineStage[] = ["analyzing", "retrieving", "matching", "composing"];

export function LoadingPipeline({ stage, hits }: LoadingPipelineProps) {
  const activeIndex = ORDER.indexOf(stage);

  return (
    <div className="rounded-lg border border-border bg-surface/70 p-4">
      <div className="relative mb-4 h-px w-full overflow-hidden bg-border">
        <motion.div
          className="absolute inset-y-0 w-1/3 rule-line"
          animate={{ x: ["-100%", "300%"] }}
          transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>

      <ul className="space-y-2">
        {ORDER.map((s, i) => {
          const done = activeIndex > i || stage === "streaming" || stage === "done";
          const active = activeIndex === i;
          return (
            <li
              key={s}
              className="flex items-center gap-2.5 text-[13px]"
              aria-current={active ? "step" : undefined}
            >
              <span
                className={
                  "size-1.5 rounded-full transition-colors " +
                  (done ? "bg-[var(--success)]" : active ? "bg-[var(--brand)]" : "bg-border")
                }
              />
              <span
                className={
                  active
                    ? "font-medium text-foreground"
                    : done
                      ? "text-muted-foreground"
                      : "text-muted-foreground/60"
                }
              >
                {STAGE_LABELS[s as keyof typeof STAGE_LABELS]}
              </span>
            </li>
          );
        })}
      </ul>

      {hits.length > 0 && (
        <div className="mt-4 space-y-1.5 border-t border-border pt-3">
          {hits.map((h) => (
            <motion.div
              key={h.chunk_id}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.18 }}
              className="flex items-center gap-2 text-xs text-muted-foreground"
            >
              <FileSearch className="size-3.5 shrink-0 text-[var(--brand)]" aria-hidden />
              <span className="truncate">
                {h.law_name} · {h.article}
                {h.clause ? ` · ${h.clause}` : ""}
              </span>
              <span className="ml-auto shrink-0 font-mono tabular-nums">
                {h.final_score.toFixed(2)}
              </span>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
