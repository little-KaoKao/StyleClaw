import { useCallback, useEffect } from "react";
import { getGallery, getProject } from "@/lib/api";
import { useAppStore } from "@/store/app-store";

/** True when a ViewSlice names a concrete slice (not just a phase). */
function hasConcreteSlice(v: { pass?: number; round?: number; batch?: number }) {
  return v.pass != null || v.round != null || v.batch != null;
}

export function useProject() {
  const project = useAppStore((s) => s.currentProject);
  const viewing = useAppStore((s) => s.viewing);
  const detail = useAppStore((s) => s.detail);
  const gallery = useAppStore((s) => s.gallery);
  const setDetail = useAppStore((s) => s.setDetail);
  const setGallery = useAppStore((s) => s.setGallery);

  const refresh = useCallback(async () => {
    if (!project) return;
    // Detail always reflects current project state — fetch it independently so a
    // failing slice gallery (e.g. a phase with no data on disk) can never drop it.
    try {
      setDetail(await getProject(project));
    } catch {
      // leave previous detail in place on transient failure
    }

    // When viewing a specific slice, fetch that slice's gallery; otherwise current.
    const params =
      viewing && hasConcreteSlice(viewing)
        ? {
            phase: viewing.phase,
            pass: viewing.pass,
            round: viewing.round,
            batch: viewing.batch,
          }
        : viewing
        ? { phase: viewing.phase }
        : undefined;
    try {
      setGallery(await getGallery(project, params));
    } catch {
      setGallery(null);
    }
  }, [project, viewing, setDetail, setGallery]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { project, detail, gallery, refresh };
}
