import { cn } from "@/lib/utils";

interface ScoreCardProps {
  scores: Record<string, number> | null;
}

// Fixed display order — JSON key order is not guaranteed.
const DIMENSIONS: { key: string; label: string }[] = [
  { key: "visual_style", label: "视觉风格" },
  { key: "color_science", label: "色彩" },
  { key: "lighting_quality", label: "光影" },
  { key: "material_texture", label: "质感" },
  { key: "post_processing", label: "后期" },
  { key: "spatial_perspective", label: "空间" },
  { key: "dynamic_state", label: "动态" },
];

export function ScoreCard({ scores }: ScoreCardProps) {
  if (!scores) return null;

  const total = scores.total;
  const rows = DIMENSIONS.filter((d) => d.key in scores);

  return (
    <div className="flex flex-col gap-1.5 rounded-md border bg-zinc-50/60 px-3 py-2">
      {typeof total === "number" && (
        <div className="flex items-baseline justify-between">
          <span className="text-[11px] font-medium text-muted-foreground">
            总分
          </span>
          <span className="text-lg leading-none font-semibold tabular-nums text-zinc-900">
            {total.toFixed(1)}
          </span>
        </div>
      )}

      {rows.length > 0 && (
        <div className="flex flex-col gap-1">
          {rows.map((d) => {
            const score = scores[d.key];
            const pct = Math.max(0, Math.min(100, score * 10));
            return (
              <div key={d.key} className="flex items-center gap-2">
                <span className="w-12 shrink-0 text-[11px] text-muted-foreground">
                  {d.label}
                </span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-200">
                  <div
                    className={cn(
                      "h-full rounded-full",
                      score >= 7 ? "bg-emerald-500" : "bg-zinc-400"
                    )}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="w-7 shrink-0 text-right text-[11px] tabular-nums text-zinc-600">
                  {score.toFixed(1)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
