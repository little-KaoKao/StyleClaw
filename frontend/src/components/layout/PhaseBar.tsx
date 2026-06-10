import { Fragment } from "react";
import { Check, ChevronRight } from "lucide-react";

import { cn } from "@/lib/utils";
import type { Phase } from "@/lib/types";

interface PhaseBarProps {
  current: Phase | null;
  /** Optional click hook — when provided, each phase block becomes interactive. */
  onSelectPhase?: (phase: Phase) => void;
}

const PHASES: { phase: Phase; label: string }[] = [
  { phase: "INIT", label: "① 初始化 INIT" },
  { phase: "MODEL_SELECT", label: "② 选模型 MODEL_SELECT" },
  { phase: "STYLE_REFINE", label: "③ 精炼 STYLE_REFINE" },
  { phase: "BATCH_T2I", label: "④ 批量 T2I" },
  { phase: "BATCH_I2I", label: "⑤ 批量 I2I" },
  { phase: "COMPLETED", label: "✓ 完成" },
];

export function PhaseBar({ current, onSelectPhase }: PhaseBarProps) {
  const currentIndex = current
    ? PHASES.findIndex((p) => p.phase === current)
    : -1;
  const interactive = Boolean(onSelectPhase);

  return (
    <nav className="flex shrink-0 items-center gap-2 overflow-x-auto border-b-4 border-foreground bg-background px-4 py-3">
      {PHASES.map((p, i) => {
        const isCurrent = i === currentIndex;
        const isDone = currentIndex >= 0 && i < currentIndex;
        const isFuture = currentIndex < 0 || i > currentIndex;

        return (
          <Fragment key={p.phase}>
            {i > 0 && (
              <ChevronRight className="h-5 w-5 shrink-0 text-foreground/60" />
            )}
            <button
              type="button"
              onClick={() => onSelectPhase?.(p.phase)}
              className={cn(
                "flex shrink-0 items-center gap-2 border-2 border-foreground px-3 py-1.5",
                "text-sm font-bold uppercase tracking-wide whitespace-nowrap",
                "transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-foreground focus-visible:ring-offset-2",
                interactive ? "cursor-pointer" : "cursor-default",
                isCurrent && "bg-bauhaus-blue text-white",
                isDone && "bg-foreground text-white",
                isFuture && "bg-muted text-foreground/40"
              )}
            >
              {isDone && <Check className="h-5 w-5 shrink-0" />}
              {p.label}
            </button>
          </Fragment>
        );
      })}
    </nav>
  );
}
