import { Fragment } from "react";
import { Check, ChevronRight } from "lucide-react";

import { cn } from "@/lib/utils";
import type { Phase } from "@/lib/types";

interface PhaseBarProps {
  current: Phase | null;
}

const PHASES: { phase: Phase; label: string }[] = [
  { phase: "INIT", label: "① 初始化 INIT" },
  { phase: "MODEL_SELECT", label: "② 选模型 MODEL_SELECT" },
  { phase: "STYLE_REFINE", label: "③ 精炼 STYLE_REFINE" },
  { phase: "BATCH_T2I", label: "④ 批量 T2I" },
  { phase: "BATCH_I2I", label: "⑤ 批量 I2I" },
  { phase: "COMPLETED", label: "✓ 完成" },
];

export function PhaseBar({ current }: PhaseBarProps) {
  const currentIndex = current
    ? PHASES.findIndex((p) => p.phase === current)
    : -1;

  return (
    <nav className="flex shrink-0 items-center gap-1 overflow-x-auto border-b px-4 py-2">
      {PHASES.map((p, i) => {
        const isCurrent = i === currentIndex;
        const isDone = currentIndex >= 0 && i < currentIndex;
        const isFuture = currentIndex < 0 || i > currentIndex;

        return (
          <Fragment key={p.phase}>
            {i > 0 && (
              <ChevronRight className="size-3.5 shrink-0 text-zinc-300" />
            )}
            <span
              className={cn(
                "flex shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium whitespace-nowrap transition-colors",
                isCurrent && "bg-zinc-900 text-white",
                isDone && "text-zinc-500",
                isFuture && "text-zinc-300"
              )}
            >
              {isDone && <Check className="size-3" />}
              {p.label}
            </span>
          </Fragment>
        );
      })}
    </nav>
  );
}
