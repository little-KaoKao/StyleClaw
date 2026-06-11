import { useState } from "react";
import { FolderPlus } from "lucide-react";

import { Header } from "@/components/layout/Header";
import { PhaseBar } from "@/components/layout/PhaseBar";
import { AppShell } from "@/components/layout/AppShell";
import { PhasePanel } from "@/components/phases/PhasePanel";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { NewProjectModal } from "@/components/modals/NewProjectModal";
import { SelectModelModal } from "@/components/modals/SelectModelModal";
import { AddRefsModal } from "@/components/modals/AddRefsModal";
import { HistoryView } from "@/components/history/HistoryView";
import { ClayBackdrop } from "@/components/layout/ClayBackdrop";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useProject } from "@/hooks/useProject";
import { useHistory } from "@/hooks/useHistory";
import { useAppStore } from "@/store/app-store";

function EmptyState({ onNewProject }: { onNewProject: () => void }) {
  return (
    <div className="flex h-full items-center justify-center p-8">
      <Card className="w-full max-w-sm text-center">
        <CardContent className="flex flex-col items-center gap-4 py-4">
          <div className="flex size-14 items-center justify-center rounded-full bg-gradient-to-br from-[#A78BFA] to-[#7C3AED] shadow-clayButton">
            <FolderPlus className="size-6 text-white" />
          </div>
          <div className="space-y-1">
            <p className="font-bold text-foreground" style={{ fontFamily: "Nunito, sans-serif" }}>
              还没有项目
            </p>
            <p className="text-sm text-muted">
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

export default function App() {
  const currentProject = useAppStore((s) => s.currentProject);
  const viewing = useAppStore((s) => s.viewing);
  const setViewing = useAppStore((s) => s.setViewing);
  const { detail } = useProject();
  const { history, refresh: refreshHistory } = useHistory();

  // Confirm-action modals (Task 10).
  const [showNew, setShowNew] = useState(false);
  const onNewProject = () => setShowNew(true);

  const [showSelectModel, setShowSelectModel] = useState(false);
  const [showAddRefs, setShowAddRefs] = useState(false);
  const [chatCollapsed, setChatCollapsed] = useState(false);

  let mainContent: React.ReactNode;
  if (!currentProject) {
    mainContent = <EmptyState onNewProject={onNewProject} />;
  } else if (viewing) {
    mainContent = history ? (
      <HistoryView
        viewing={viewing}
        history={history}
        onHistoryChange={refreshHistory}
      />
    ) : (
      <div className="flex h-full items-center justify-center p-8 text-sm font-bold tracking-wide text-muted">
        加载历史…
      </div>
    );
  } else {
    mainContent = (
      <PhasePanel
        onSelectModel={() => setShowSelectModel(true)}
        onAddRefs={() => setShowAddRefs(true)}
      />
    );
  }

  return (
    <div className="relative flex min-h-screen h-screen flex-col bg-background text-foreground">
      <ClayBackdrop />
      <Header onNewProject={onNewProject} />
      <PhaseBar
        current={detail?.state.phase ?? null}
        onSelectPhase={
          currentProject
            ? (phase) =>
                // Clicking the CURRENT phase returns to the live working panel
                // (with its action buttons); clicking any OTHER phase opens that
                // phase's read-only history. Without this, clicking the current
                // phase to "go back" just reopened history and the actions
                // appeared to vanish.
                phase === detail?.state.phase
                  ? setViewing(null)
                  : setViewing({ phase })
            : undefined
        }
      />
      <AppShell
        main={mainContent}
        chat={
          currentProject ? (
            <ChatPanel key={currentProject} onCollapse={() => setChatCollapsed(true)} />
          ) : null
        }
        chatCollapsed={chatCollapsed}
        onExpandChat={() => setChatCollapsed(false)}
      />

      <NewProjectModal open={showNew} onOpenChange={setShowNew} />
      <SelectModelModal
        open={showSelectModel}
        onOpenChange={setShowSelectModel}
        projectName={currentProject}
      />
      <AddRefsModal
        open={showAddRefs}
        onOpenChange={setShowAddRefs}
        projectName={currentProject}
      />
    </div>
  );
}
