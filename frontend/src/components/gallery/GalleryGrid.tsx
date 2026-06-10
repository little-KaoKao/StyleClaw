import { useState } from "react";

import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScoreCard } from "@/components/gallery/ScoreCard";
import { ORB_GRADIENTS } from "@/lib/clay";
import type { GalleryGroup } from "@/lib/types";

interface GalleryGridProps {
  groups: GalleryGroup[];
  refImages?: string[];
}

export function GalleryGrid({ groups, refImages }: GalleryGridProps) {
  // One controlled lightbox at grid level, driven by the selected image src.
  const [selectedSrc, setSelectedSrc] = useState<string | null>(null);

  const hasRefs = !!refImages && refImages.length > 0;

  if (groups.length === 0 && !hasRefs) {
    return (
      <div className="flex items-center justify-center rounded-[24px] py-12 font-bold text-muted">
        还没有生成结果
      </div>
    );
  }

  // Adaptive density: model/refine groups carry several images each (compare
  // side-by-side → fewer, wider cards); batch cases carry 1 image each (pack
  // many cards per row to use the full width instead of one column).
  const maxImgs = groups.reduce((m, g) => Math.max(m, g.images.length), 0);
  const gridCols =
    maxImgs >= 3
      ? "grid-cols-1 xl:grid-cols-2"
      : "grid-cols-2 md:grid-cols-3 xl:grid-cols-4";
  const innerCols = maxImgs >= 3 ? "grid-cols-2" : "grid-cols-1";

  const openLightbox = (src: string) => setSelectedSrc(src);

  return (
    <div className="flex flex-col gap-6">
      {hasRefs && (
        <section className="flex flex-col gap-3">
          <h3 className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-muted">
            <span className="h-4 w-4 shrink-0 rounded-full bg-gradient-to-br from-purple-400 to-purple-600" />
            参考图
          </h3>
          <div className="flex flex-wrap gap-3">
            {refImages!.map((src) => (
              <button
                key={src}
                type="button"
                onClick={() => openLightbox(src)}
                className="overflow-hidden rounded-2xl bg-clay-surface shadow-clayCard transition-all duration-300 hover:-translate-y-0.5 hover:shadow-clayCardHover"
              >
                <img src={src} alt="参考图" loading="lazy" className="size-20 object-cover" />
              </button>
            ))}
          </div>
        </section>
      )}

      <div className={`grid gap-4 ${gridCols}`}>
        {groups.map((group, groupIndex) => (
          <section
            key={group.label}
            className="flex flex-col gap-2 rounded-[24px] bg-white/70 p-4 shadow-clayCard backdrop-blur-xl transition-all duration-500 hover:-translate-y-1 hover:shadow-clayCardHover"
          >
            <div className="flex items-center gap-2">
              <span
                className={`h-4 w-4 shrink-0 rounded-full bg-gradient-to-br ${
                  ORB_GRADIENTS[groupIndex % ORB_GRADIENTS.length]
                }`}
              />
              <h3
                className="truncate text-sm font-bold text-foreground"
                style={{ fontFamily: "Nunito, sans-serif" }}
                title={group.label}
              >
                {group.label}
              </h3>
            </div>

            {group.prompt && (
              <p
                className="line-clamp-3 text-xs leading-snug text-muted"
                title={group.prompt}
              >
                {group.prompt}
              </p>
            )}

            {group.scores && <ScoreCard scores={group.scores} />}

            {group.images.length === 0 ? (
              <p className="py-4 text-center text-xs font-bold text-muted">
                暂无图片
              </p>
            ) : (
              <div className={`grid gap-2 ${innerCols}`}>
                {group.images.map((src) => (
                  <button
                    key={src}
                    type="button"
                    onClick={() => openLightbox(src)}
                    className="overflow-hidden rounded-[20px] bg-clay-surface shadow-clayCard transition-all duration-300 hover:-translate-y-0.5 hover:shadow-clayCardHover"
                  >
                    <img
                      src={src}
                      alt={group.label}
                      loading="lazy"
                      className="h-auto w-full object-contain"
                    />
                  </button>
                ))}
              </div>
            )}
          </section>
        ))}
      </div>

      <Dialog
        open={!!selectedSrc}
        onOpenChange={(open) => {
          if (!open) setSelectedSrc(null);
        }}
      >
        <DialogContent className="max-w-3xl p-2">
          <DialogTitle className="sr-only">预览图片</DialogTitle>
          {selectedSrc && (
            <img
              src={selectedSrc}
              alt="预览"
              className="max-h-[80vh] w-full rounded-[24px] object-contain"
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
