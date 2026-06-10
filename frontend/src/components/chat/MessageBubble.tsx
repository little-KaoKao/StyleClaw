import { cn } from "@/lib/utils";
import { PRIMARY_GRADIENT } from "@/lib/clay";

interface MessageBubbleProps {
  role: "user" | "assistant";
  text: string;
}

/**
 * A single text message. User messages sit on the right (clay gradient bubble),
 * assistant text on the left (soft white clay bubble). Plan cards are rendered
 * separately by PlanPreview.
 */
export function MessageBubble({ role, text }: MessageBubbleProps) {
  const isUser = role === "user";
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] rounded-[20px] px-4 py-2 text-sm font-medium whitespace-pre-wrap break-words",
          isUser
            ? cn(PRIMARY_GRADIENT, "text-white shadow-clayButton")
            : "bg-white/80 text-foreground shadow-clayCard"
        )}
      >
        {text}
      </div>
    </div>
  );
}
