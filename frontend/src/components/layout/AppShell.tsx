import type { ReactNode } from "react";
import { MessageCircle, ChevronLeft } from "lucide-react";

interface AppShellProps {
  main: ReactNode;
  /** Chat node, or null when no project is selected. */
  chat: ReactNode;
  /** When true the chat aside collapses to a slim reopen rail. */
  chatCollapsed?: boolean;
  /** Expand the collapsed chat rail. */
  onExpandChat?: () => void;
}

/**
 * Two-column work area. The main column is now its own frosted clay card so it
 * matches the header / phase-bar / chat surfaces — content no longer floats
 * loose against the page background, and its left edge lines up with the chrome.
 */
export function AppShell({ main, chat, chatCollapsed, onExpandChat }: AppShellProps) {
  const hasChat = chat !== null;
  return (
    <div className="flex min-h-0 flex-1 gap-4 p-4">
      <main className="min-h-0 flex-1 overflow-y-auto rounded-[32px] bg-white/55 backdrop-blur-xl shadow-clayCard">
        {main}
      </main>

      {hasChat &&
        (chatCollapsed ? (
          <button
            type="button"
            onClick={onExpandChat}
            aria-label="展开助手"
            className="group flex w-14 shrink-0 flex-col items-center gap-3 rounded-[32px] bg-white/70 py-5 backdrop-blur-xl shadow-clayCard transition-all duration-200 hover:-translate-y-0.5"
          >
            <span className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-purple-400 to-purple-600 text-white shadow-clayButton">
              <MessageCircle className="h-5 w-5" />
            </span>
            <span
              className="text-xs font-extrabold tracking-widest text-muted [writing-mode:vertical-rl]"
              style={{ fontFamily: "Nunito, sans-serif" }}
            >
              助手
            </span>
            <ChevronLeft className="h-4 w-4 text-muted transition-transform duration-200 group-hover:-translate-x-0.5" />
          </button>
        ) : (
          <aside className="flex min-h-0 w-[380px] shrink-0 flex-col overflow-hidden rounded-[32px] bg-white/70 backdrop-blur-xl shadow-clayCard">
            {chat}
          </aside>
        ))}
    </div>
  );
}
