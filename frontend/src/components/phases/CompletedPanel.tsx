import { Sparkles } from "lucide-react";

import { GalleryGrid } from "@/components/gallery/GalleryGrid";
import { CLAY_CARD } from "@/lib/clay";
import { cn } from "@/lib/utils";
import type { PanelProps } from "./shared";

export function CompletedPanel({ gallery }: PanelProps) {
  return (
    <div className="space-y-6 p-6">
      <div
        className={cn(
          CLAY_CARD,
          "relative flex items-center gap-4 overflow-hidden bg-gradient-to-br from-[#A78BFA]/30 via-white/70 to-[#F472B6]/20 p-8"
        )}
      >
        <div className="flex size-14 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-[#A78BFA] to-[#7C3AED] text-white shadow-clayButton">
          <Sparkles className="h-7 w-7" />
        </div>
        <div className="flex flex-col gap-1">
          <h2
            className="text-3xl font-black tracking-tight text-foreground md:text-4xl"
            style={{ fontFamily: "Nunito, sans-serif" }}
          >
            项目已完成
          </h2>
          <p className="font-medium text-muted">
            风格探索流程已走完，下面是各阶段的产出。
          </p>
        </div>
      </div>
      <GalleryGrid
        groups={gallery?.groups ?? []}
        refImages={gallery?.ref_images}
      />
    </div>
  );
}
