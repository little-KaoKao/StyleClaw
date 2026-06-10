import { GalleryGrid } from "@/components/gallery/GalleryGrid";
import { BORDER, SHADOW_MD } from "@/lib/bauhaus";
import { cn } from "@/lib/utils";
import type { PanelProps } from "./shared";

export function CompletedPanel({ gallery }: PanelProps) {
  return (
    <div className="space-y-6 p-6">
      <div
        className={cn(
          "flex flex-col gap-2 bg-bauhaus-yellow p-6 text-foreground",
          BORDER,
          SHADOW_MD
        )}
      >
        <h2 className="font-black uppercase tracking-tight text-2xl md:text-3xl">
          ✓ 项目已完成
        </h2>
        <p className="font-medium text-foreground/70">
          风格探索流程已走完，下面是各阶段的产出。
        </p>
      </div>
      <GalleryGrid
        groups={gallery?.groups ?? []}
        refImages={gallery?.ref_images}
      />
    </div>
  );
}
