import { useState } from "react";

import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScoreCard } from "@/components/gallery/ScoreCard";
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
      <div className="flex items-center justify-center py-12 text-sm text-muted-foreground">
        还没有生成结果
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {hasRefs && (
        <section className="flex flex-col gap-2">
          <h3 className="text-xs font-medium tracking-wide text-muted-foreground">
            参考图
          </h3>
          <div className="flex flex-wrap gap-2">
            {refImages!.map((src) => (
              <button
                key={src}
                type="button"
                onClick={() => setSelectedSrc(src)}
                className="overflow-hidden rounded-md border transition-opacity hover:opacity-80"
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

      {groups.map((group) => (
        <section key={group.label} className="flex flex-col gap-2">
          <div className="flex items-start justify-between gap-3">
            <h3 className="font-mono text-sm font-medium text-zinc-900">
              {group.label}
            </h3>
          </div>

          {group.scores && (
            <div className="max-w-xs">
              <ScoreCard scores={group.scores} />
            </div>
          )}

          {group.images.length === 0 ? (
            <p className="text-xs text-muted-foreground">暂无图片</p>
          ) : (
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
              {group.images.map((src) => (
                <button
                  key={src}
                  type="button"
                  onClick={() => setSelectedSrc(src)}
                  className="overflow-hidden rounded-md border bg-zinc-50 transition-opacity hover:opacity-80"
                >
                  <img
                    src={src}
                    alt={group.label}
                    loading="lazy"
                    className="aspect-square w-full object-cover"
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
        <DialogContent className="max-w-3xl p-2">
          <DialogTitle className="sr-only">预览图片</DialogTitle>
          {selectedSrc && (
            <img
              src={selectedSrc}
              alt="预览"
              className="max-h-[80vh] w-full rounded-md object-contain"
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
