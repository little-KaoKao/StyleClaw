import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { ORB_GRADIENTS, PRIMARY_GRADIENT } from "@/lib/clay";

interface HistoryTileProps {
  label: string;
  /** Currently being viewed. */
  active?: boolean;
  /** The project's live slice. */
  current?: boolean;
  accentIndex?: number;
  onClick: () => void;
}

export function HistoryTile({
  label,
  active,
  current,
  accentIndex = 0,
  onClick,
}: HistoryTileProps) {
  const orb = ORB_GRADIENTS[accentIndex % ORB_GRADIENTS.length];
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-[20px] px-4 py-3 font-bold whitespace-nowrap transition-all hover:-translate-y-1",
        "focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-clay-accent/30",
        active
          ? cn(PRIMARY_GRADIENT, "text-white shadow-clayButton")
          : "bg-white/70 text-foreground shadow-clayCard"
      )}
      style={{ fontFamily: "Nunito, sans-serif" }}
    >
      <span className="flex items-center gap-2">
        <span
          className={cn(
            "h-3 w-3 shrink-0 rounded-full bg-gradient-to-br",
            orb
          )}
        />
        {label}
        {current && (
          <Badge variant="secondary" className="ml-1">
            当前
          </Badge>
        )}
      </span>
    </button>
  );
}
