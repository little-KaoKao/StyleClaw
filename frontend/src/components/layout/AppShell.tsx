import type { ReactNode } from "react";

interface AppShellProps {
  main: ReactNode;
  chat: ReactNode;
}

export function AppShell({ main, chat }: AppShellProps) {
  return (
    <div className="flex min-h-0 flex-1">
      <main className="min-h-0 flex-1 overflow-y-auto">{main}</main>
      {chat !== null && (
        <aside className="flex min-h-0 w-[380px] shrink-0 flex-col overflow-y-auto border-l">
          {chat}
        </aside>
      )}
    </div>
  );
}
