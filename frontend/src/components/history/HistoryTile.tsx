import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { ACCENTS, SHADOW_SM, PRESS, LIFT } from "@/lib/bauhaus";

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
  const accent = ACCENTS[accentIndex % ACCENTS.length];
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "relative border-2 border-foreground rounded-none px-4 py-3",
        "font-bold uppercase tracking-wide whitespace-nowrap",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-foreground",
        SHADOW_SM,
        PRESS,
        LIFT,
        active ? "bg-bauhaus-blue text-white" : "bg-white text-foreground"
      )}
    >
      <span
        className="absolute left-1 top-1 h-2.5 w-2.5"
        style={{ backgroundColor: accent }}
      />
      <span className="flex items-center gap-2 pl-2">
        {label}
        {current && (
          <Badge variant="yellow" className="ml-1">
            当前
          </Badge>
        )}
      </span>
    </button>
  );
}
