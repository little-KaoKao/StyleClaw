import { SHADOW_SM } from "@/lib/bauhaus";

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
    <div
      className={`flex flex-col gap-2 rounded-none border-2 border-foreground bg-white p-3 ${SHADOW_SM}`}
    >
      {typeof total === "number" && (
        <div className="flex items-center gap-3">
          <span className="text-xs font-bold uppercase tracking-wider text-foreground">
            总分
          </span>
          <span className="flex h-10 w-10 shrink-0 items-center justify-center bg-bauhaus-blue text-2xl font-black tabular-nums text-white">
            {total.toFixed(1)}
          </span>
        </div>
      )}

      {rows.length > 0 && (
        <div className="flex flex-col gap-1.5">
          {rows.map((d) => {
            const score = scores[d.key];
            const pct = Math.max(0, Math.min(100, score * 10));
            const strong = score >= 7;
            return (
              <div key={d.key} className="flex items-center gap-2">
                <span className="w-16 shrink-0 text-xs font-bold uppercase tracking-wider">
                  {d.label}
                </span>
                <div className="h-3 flex-1 border-2 border-foreground bg-muted">
                  <div
                    className={`h-full ${
                      strong ? "bg-bauhaus-blue" : "bg-bauhaus-red"
                    }`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="w-8 shrink-0 text-right text-xs font-bold tabular-nums">
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
