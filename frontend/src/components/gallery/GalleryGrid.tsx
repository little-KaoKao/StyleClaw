import { useState } from "react";

import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScoreCard } from "@/components/gallery/ScoreCard";
import { ACCENTS, SHADOW_SM } from "@/lib/bauhaus";
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
      <div className="flex items-center justify-center py-12 text-sm font-bold uppercase tracking-wide text-foreground/40">
        还没有生成结果
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-8">
      {hasRefs && (
        <section className="flex flex-col gap-3">
          <h3 className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest">
            <span
              className="h-3 w-3 shrink-0"
              style={{ backgroundColor: ACCENTS[0] }}
            />
            参考图
          </h3>
          <div className="flex flex-wrap gap-2">
            {refImages!.map((src) => (
              <button
                key={src}
                type="button"
                onClick={() => setSelectedSrc(src)}
                className="overflow-hidden rounded-none border-2 border-foreground transition-opacity hover:opacity-80"
              >
                <img
                  src={src}
                  alt="参考图"
                  loading="lazy"
                  className="size-20 object-cover"
                />
              </button>
            ))}
          </div>
        </section>
      )}

      {groups.map((group, groupIndex) => (
        <section key={group.label} className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <span
              className="h-5 w-5 shrink-0 border-2 border-foreground"
              style={{ backgroundColor: ACCENTS[groupIndex % ACCENTS.length] }}
            />
            <h3 className="font-bold uppercase tracking-wide">{group.label}</h3>
          </div>

          {group.scores && (
            <div className="max-w-xs">
              <ScoreCard scores={group.scores} />
            </div>
          )}

          {group.images.length === 0 ? (
            <p className="text-sm font-bold uppercase tracking-wide text-foreground/40">
              暂无图片
            </p>
          ) : (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
              {group.images.map((src) => (
                <button
                  key={src}
                  type="button"
                  onClick={() => setSelectedSrc(src)}
                  className={`overflow-hidden rounded-none border-2 border-foreground md:border-4 ${SHADOW_SM}`}
                >
                  <img
                    src={src}
                    alt={group.label}
                    loading="lazy"
                    className="aspect-square w-full object-cover grayscale transition-all duration-200 hover:grayscale-0"
                  />
                </button>
              ))}
            </div>
          )}
        </section>
      ))}

      <Dialog
        open={!!selectedSrc}
        onOpenChange={(open) => {
          if (!open) setSelectedSrc(null);
        }}
      >
        <DialogContent className="max-w-3xl border-4 border-foreground p-2">
          <DialogTitle className="sr-only">预览图片</DialogTitle>
          {selectedSrc && (
            <img
              src={selectedSrc}
              alt="预览"
              className="max-h-[80vh] w-full rounded-none object-contain"
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
