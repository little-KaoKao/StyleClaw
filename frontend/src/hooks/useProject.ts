import { useCallback, useEffect } from "react";
import { getGallery, getProject } from "@/lib/api";
import { useAppStore } from "@/store/app-store";

export function useProject() {
  const project = useAppStore((s) => s.currentProject);
  const detail = useAppStore((s) => s.detail);
  const gallery = useAppStore((s) => s.gallery);
  const setDetail = useAppStore((s) => s.setDetail);
  const setGallery = useAppStore((s) => s.setGallery);

  const refresh = useCallback(async () => {
    if (!project) return;
    const [d, g] = await Promise.all([getProject(project), getGallery(project)]);
    setDetail(d);
    setGallery(g);
  }, [project, setDetail, setGallery]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { project, detail, gallery, refresh };
}
