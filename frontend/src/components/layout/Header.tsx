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
import { listProjects } from "@/lib/api";
import type { ProjectSummary } from "@/lib/types";
import { useAppStore } from "@/store/app-store";

interface HeaderProps {
  onNewProject?: () => void;
}

/** Three geometric Bauhaus shapes in a row: red circle, blue square, yellow triangle. */
function GeometricLogo() {
  return (
    <div className="flex items-center gap-1.5">
      <span className="h-4 w-4 shrink-0 rounded-full bg-bauhaus-red" />
      <span className="h-4 w-4 shrink-0 bg-bauhaus-blue" />
      <span
        className="h-4 w-4 shrink-0 bg-bauhaus-yellow"
        style={{ clipPath: "polygon(50% 0, 0 100%, 100% 100%)" }}
      />
    </div>
  );
}

export function Header({ onNewProject }: HeaderProps) {
  const currentProject = useAppStore((s) => s.currentProject);
  const setCurrentProject = useAppStore((s) => s.setCurrentProject);

  const [projects, setProjects] = useState<ProjectSummary[]>([]);

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
    <header className="flex h-16 shrink-0 items-center justify-between border-b-4 border-foreground bg-background px-4">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-3">
          <GeometricLogo />
          <span className="font-black uppercase tracking-tight text-xl text-foreground">
            STYLECLAW
          </span>
        </div>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="default" className="gap-2">
              <span className="max-w-[180px] truncate">
                {currentProject ?? "选择项目"}
              </span>
              <ChevronDown className="h-5 w-5 shrink-0 opacity-60" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="min-w-[220px]">
            {projects.length === 0 ? (
              <div className="px-2 py-3 text-center text-xs text-muted-foreground">
                暂无项目
              </div>
            ) : (
              projects.map((p) => (
                <DropdownMenuItem
                  key={p.name}
                  onSelect={() => setCurrentProject(p.name)}
                  className="justify-between"
                >
                  <span className="flex items-center gap-2 truncate">
                    {p.name === currentProject && (
                      <Check className="h-5 w-5 shrink-0 text-foreground" />
                    )}
                    <span className="truncate">{p.name}</span>
                  </span>
                  <span className="ml-2 shrink-0 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                    {p.phase}
                  </span>
                </DropdownMenuItem>
              ))
            )}
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => onNewProject?.()}>
              <Plus className="h-5 w-5 shrink-0" />
              新建项目
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <div className="flex items-center gap-3">
        <Badge variant="outline">本地</Badge>
      </div>
    </header>
  );
}
