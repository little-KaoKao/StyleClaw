import { Fragment } from "react";
import { Check } from "lucide-react";

import { cn } from "@/lib/utils";
import { PRIMARY_GRADIENT } from "@/lib/clay";
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
  { phase: "COMPLETED", label: "完成" },
];

export function PhaseBar({ current, onSelectPhase }: PhaseBarProps) {
  const currentIndex = current
    ? PHASES.findIndex((p) => p.phase === current)
    : -1;
  const interactive = Boolean(onSelectPhase);

  return (
    <nav className="m-4 mb-0 flex shrink-0 items-center gap-2 overflow-x-auto rounded-[32px] bg-white/70 px-4 py-3 backdrop-blur-xl shadow-clayCard">
      {PHASES.map((p, i) => {
        const isCurrent = i === currentIndex;
        const isDone = currentIndex >= 0 && i < currentIndex;
        const isFuture = currentIndex < 0 || i > currentIndex;

        return (
          <Fragment key={p.phase}>
            {i > 0 && (
              <span className="shrink-0 select-none px-0.5 text-muted">›</span>
            )}
            <button
              type="button"
              onClick={() => onSelectPhase?.(p.phase)}
              style={{ fontFamily: "Nunito, sans-serif" }}
              className={cn(
                "flex shrink-0 items-center gap-2 rounded-full px-4 py-2",
                "text-sm font-bold whitespace-nowrap",
                "transition-all duration-200 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-clay-accent/30",
                interactive ? "cursor-pointer hover:-translate-y-0.5" : "cursor-default",
                isCurrent && `${PRIMARY_GRADIENT} text-white shadow-clayButton`,
                isDone && "bg-white text-clay-accent shadow-clayButton",
                isFuture && "bg-clay-surface text-muted"
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
