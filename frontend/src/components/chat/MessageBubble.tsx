import { cn } from "@/lib/utils";

interface MessageBubbleProps {
  role: "user" | "assistant";
  text: string;
}

/**
 * A single text message. User messages sit on the right (zinc-900), assistant
 * text on the left (muted). Plan cards are rendered separately by PlanPreview.
 */
export function MessageBubble({ role, text }: MessageBubbleProps) {
  const isUser = role === "user";
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap break-words",
          isUser
            ? "bg-zinc-900 text-white"
            : "bg-muted text-foreground"
        )}
      >
        {text}
      </div>
    </div>
  );
}
