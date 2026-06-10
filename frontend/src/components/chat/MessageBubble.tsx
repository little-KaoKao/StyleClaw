import { cn } from "@/lib/utils";
import { SHADOW_SM } from "@/lib/bauhaus";

interface MessageBubbleProps {
  role: "user" | "assistant";
  text: string;
}

/**
 * A single text message. User messages sit on the right (bauhaus-blue block),
 * assistant text on the left (white block). Plan cards are rendered separately
 * by PlanPreview.
 */
export function MessageBubble({ role, text }: MessageBubbleProps) {
  const isUser = role === "user";
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] rounded-none border-2 border-foreground px-3 py-2 text-sm font-medium whitespace-pre-wrap break-words",
          isUser
            ? cn("bg-bauhaus-blue text-white", SHADOW_SM)
            : "bg-white text-foreground"
        )}
      >
        {text}
      </div>
    </div>
  );
}
