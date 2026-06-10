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
    <div className="flex flex-col gap-3 rounded-[20px] bg-white/70 p-4 shadow-clayCard">
      {typeof total === "number" && (
        <div className="flex items-center gap-3">
          <span className="text-xs font-bold uppercase tracking-widest text-muted">
            总分
          </span>
          <span
            className="flex h-14 w-14 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-[#A78BFA] to-[#7C3AED] text-xl font-black tabular-nums text-white shadow-clayButton"
            style={{ fontFamily: "Nunito, sans-serif" }}
          >
            {total.toFixed(1)}
          </span>
        </div>
      )}

      {rows.length > 0 && (
        <div className="flex flex-col gap-2">
          {rows.map((d) => {
            const score = scores[d.key];
            const pct = Math.max(0, Math.min(100, score * 10));
            const strong = score >= 7;
            return (
              <div key={d.key} className="flex items-center gap-2">
                <span className="w-16 shrink-0 text-xs font-bold text-muted">
                  {d.label}
                </span>
                <div className="h-3 flex-1 overflow-hidden rounded-full bg-clay-surface shadow-clayPressed">
                  <div
                    className={`h-full rounded-full bg-gradient-to-r ${
                      strong
                        ? "from-[#A78BFA] to-[#7C3AED]"
                        : "from-[#F472B6] to-[#DB2777]"
                    }`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="w-8 shrink-0 text-right text-xs font-bold tabular-nums text-foreground">
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
