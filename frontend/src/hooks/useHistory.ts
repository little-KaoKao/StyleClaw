import { useCallback, useEffect, useState } from "react";
import { getHistory } from "@/lib/api";
import { useAppStore } from "@/store/app-store";
import type { History } from "@/lib/types";

export function useHistory() {
  const project = useAppStore((s) => s.currentProject);
  const [history, setHistory] = useState<History | null>(null);
  const refresh = useCallback(async () => {
    if (!project) {
      setHistory(null);
      return;
    }
    try {
      setHistory(await getHistory(project));
    } catch {
      setHistory(null);
    }
  }, [project]);
  useEffect(() => {
    refresh();
  }, [refresh]);
  return { history, refresh };
}
