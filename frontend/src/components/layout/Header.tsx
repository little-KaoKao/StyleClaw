import { useEffect, useState } from "react";
import { ChevronDown, Plus, Check } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ORB_GRADIENTS } from "@/lib/clay";
import { listProjects } from "@/lib/api";
import type { ProjectSummary } from "@/lib/types";
import { useAppStore } from "@/store/app-store";

interface HeaderProps {
  onNewProject?: () => void;
}

/** Three soft clay gradient orbs in a row. */
function ClayLogo() {
  return (
    <div className="flex items-center gap-1.5">
      <span className={`h-5 w-5 shrink-0 rounded-full bg-gradient-to-br ${ORB_GRADIENTS[0]} shadow-clayButton`} />
      <span className={`h-5 w-5 shrink-0 rounded-full bg-gradient-to-br ${ORB_GRADIENTS[1]} shadow-clayButton`} />
      <span className={`h-5 w-5 shrink-0 rounded-full bg-gradient-to-br ${ORB_GRADIENTS[2]} shadow-clayButton`} />
    </div>
  );
}

/** Short Chinese label per phase, for the header status chip. */
const PHASE_LABELS: Record<string, string> = {
  INIT: "初始化",
  MODEL_SELECT: "选模型",
  STYLE_REFINE: "精炼",
  BATCH_T2I: "批量 T2I",
  BATCH_I2I: "批量 I2I",
  COMPLETED: "完成",
};

/** "2026-06-11T08:30:00Z" → "06-11 08:30" (best-effort, falls back to raw). */
function shortTime(iso: string): string {
  const m = /(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(iso);
  return m ? `${m[2]}-${m[3]} ${m[4]}:${m[5]}` : iso;
}

export function Header({ onNewProject }: HeaderProps) {
  const currentProject = useAppStore((s) => s.currentProject);
  const setCurrentProject = useAppStore((s) => s.setCurrentProject);

  const [projects, setProjects] = useState<ProjectSummary[]>([]);

  const summary = projects.find((p) => p.name === currentProject) ?? null;
  const roundOrBatch = summary
    ? summary.phase === "STYLE_REFINE"
      ? `轮次 ${summary.current_round}`
      : summary.phase === "BATCH_T2I" || summary.phase === "BATCH_I2I"
        ? `批次 ${summary.current_batch}`
        : null
    : null;

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const { projects } = await listProjects();
        if (!cancelled) setProjects(projects);
      } catch {
        // Backend may be down / no projects yet — degrade to an empty list.
        if (!cancelled) setProjects([]);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <header className="m-4 mb-0 flex h-16 shrink-0 items-center justify-between rounded-[32px] bg-white/70 px-6 backdrop-blur-xl shadow-clayCard">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3">
          <ClayLogo />
          <span
            className="text-xl font-black tracking-tight text-foreground"
            style={{ fontFamily: "Nunito, sans-serif" }}
          >
            StyleClaw
          </span>
        </div>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="secondary" size="sm" className="gap-2">
              <span className="max-w-[180px] truncate">
                {currentProject ?? "选择项目"}
              </span>
              <ChevronDown className="h-5 w-5 shrink-0 opacity-60" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="min-w-[220px]">
            {projects.length === 0 ? (
              <div className="px-2 py-3 text-center text-xs text-muted">
                暂无项目
              </div>
            ) : (
              projects.map((p) => (
                <DropdownMenuItem
                  key={p.name}
                  onSelect={() => setCurrentProject(p.name)}
                  className="justify-between rounded-xl hover:bg-clay-accent/10"
                >
                  <span className="flex items-center gap-2 truncate">
                    {p.name === currentProject && (
                      <Check className="h-5 w-5 shrink-0 text-clay-accent" />
                    )}
                    <span className="truncate">{p.name}</span>
                  </span>
                  <span className="ml-2 shrink-0 text-[10px] font-bold tracking-wider text-muted">
                    {p.phase}
                  </span>
                </DropdownMenuItem>
              ))
            )}
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onSelect={() => onNewProject?.()}
              className="rounded-xl hover:bg-clay-accent/10"
            >
              <Plus className="h-5 w-5 shrink-0" />
              新建项目
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {summary && (
        <div className="hidden items-center gap-2 text-xs font-bold text-muted md:flex">
          <span className="rounded-full bg-clay-surface px-3 py-1 text-clay-accent">
            {PHASE_LABELS[summary.phase] ?? summary.phase}
          </span>
          {roundOrBatch && (
            <span className="rounded-full bg-clay-surface px-3 py-1">{roundOrBatch}</span>
          )}
          <span className="text-muted/70">更新 {shortTime(summary.last_updated)}</span>
        </div>
      )}

      <div className="flex items-center gap-3">
        <Badge variant="secondary">本地</Badge>
      </div>
    </header>
  );
}
