import { CheckCircle2 } from "lucide-react";

import { GalleryGrid } from "@/components/gallery/GalleryGrid";
import type { PanelProps } from "./shared";

export function CompletedPanel({ gallery }: PanelProps) {
  return (
    <div className="space-y-6 p-6">
      <header className="flex items-center gap-2">
        <CheckCircle2 className="size-5 text-emerald-600" />
        <h2 className="text-base font-semibold text-zinc-900">项目已完成</h2>
      </header>
      <p className="text-sm text-muted-foreground">
        风格探索流程已走完，下面是各阶段的产出。
      </p>
      <GalleryGrid
        groups={gallery?.groups ?? []}
        refImages={gallery?.ref_images}
      />
    </div>
  );
}
