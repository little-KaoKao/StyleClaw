import { useState } from "react";
import { FolderPlus } from "lucide-react";

import { Header } from "@/components/layout/Header";
import { PhaseBar } from "@/components/layout/PhaseBar";
import { AppShell } from "@/components/layout/AppShell";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useProject } from "@/hooks/useProject";
import { useAppStore } from "@/store/app-store";

function EmptyState({ onNewProject }: { onNewProject: () => void }) {
  return (
    <div className="flex h-full items-center justify-center p-8">
      <Card className="w-full max-w-sm text-center">
        <CardContent className="flex flex-col items-center gap-4 py-4">
          <div className="flex size-12 items-center justify-center rounded-full bg-zinc-100">
            <FolderPlus className="size-6 text-zinc-500" />
          </div>
          <div className="space-y-1">
            <p className="font-medium">还没有项目</p>
            <p className="text-sm text-muted-foreground">
              从一组参考图新建项目，开始风格探索。
            </p>
          </div>
          <Button onClick={onNewProject}>
            <FolderPlus className="size-4" />
            新建项目
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

function PhaseWorkspacePlaceholder() {
  return (
    <div className="flex h-full items-center justify-center p-8 text-sm text-muted-foreground">
      阶段面板（Task 6）
    </div>
  );
}

function ChatPlaceholder() {
  return (
    <div className="flex h-full items-center justify-center p-8 text-sm text-muted-foreground">
      聊天助手（Task 9）
    </div>
  );
}

export default function App() {
  const currentProject = useAppStore((s) => s.currentProject);
  const { detail } = useProject();

  // Placeholder for the "new project" modal (Task 10). For now we only
  // track intent so the wiring is in place.
  const [, setShowNew] = useState(false);
  const onNewProject = () => setShowNew(true);

  return (
    <div className="flex h-screen flex-col">
      <Header onNewProject={onNewProject} />
      <PhaseBar current={detail?.state.phase ?? null} />
      <AppShell
        main={
          currentProject ? (
            <PhaseWorkspacePlaceholder />
          ) : (
            <EmptyState onNewProject={onNewProject} />
          )
        }
        chat={currentProject ? <ChatPlaceholder /> : null}
      />
    </div>
  );
}
