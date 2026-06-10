import { useState } from "react";
import { ArrowLeft } from "lucide-react";

import { Button } from "@/components/ui/button";
import { GalleryGrid } from "@/components/gallery/GalleryGrid";
import { HistoryPanel } from "./HistoryPanel";
import { useProject } from "@/hooks/useProject";
import { useRun } from "@/hooks/useRun";
import { useAppStore } from "@/store/app-store";
import { rollback as apiRollback } from "@/lib/api";
import { SHADOW_SM } from "@/lib/bauhaus";
import type { History, ViewSlice } from "@/lib/types";

interface HistoryViewProps {
  viewing: ViewSlice;
  history: History;
  /** Refresh App's history instance so tiles/current reflect post-redo state. */
  onHistoryChange?: () => void;
}

/** True when a ViewSlice names a concrete slice (not just a phase). */
function hasConcreteSlice(v: ViewSlice) {
  return v.pass != null || v.round != null || v.batch != null;
}

function sliceLabel(v: ViewSlice): string {
  const parts: string[] = [];
  if (v.pass != null) parts.push(`pass-${String(v.pass).padStart(3, "0")}`);
  if (v.round != null) parts.push(`round-${String(v.round).padStart(3, "0")}`);
  if (v.batch != null) parts.push(`batch-${String(v.batch).padStart(3, "0")}`);
  return parts.join(" / ");
}

/** Whether the viewed slice differs from the project's live current slice. */
function differsFromCurrent(v: ViewSlice, history: History): boolean {
  const cur = history.current;
  if (v.phase !== cur.phase) return true;
  if (v.pass != null && v.pass !== cur.pass) return true;
  if (v.round != null && v.round !== cur.round) return true;
  if (v.batch != null && v.batch !== cur.batch) return true;
  return false;
}

export function HistoryView({ viewing, history, onHistoryChange }: HistoryViewProps) {
  const project = useAppStore((s) => s.currentProject);
  const setViewing = useAppStore((s) => s.setViewing);
  const gallery = useAppStore((s) => s.gallery);
  const { refresh } = useProject();
  const { run } = useRun();

  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const concrete = hasConcreteSlice(viewing);
  const showBanner = concrete && differsFromCurrent(viewing, history);

  async function afterRedo() {
    setViewing(null);
    await refresh();
    onHistoryChange?.();
  }

  async function doRun(name: string, args: Record<string, unknown>) {
    if (!project) return;
    setBusy(true);
    setMsg(null);
    try {
      await run([{ name, args }]);
      await afterRedo();
    } catch (err) {
      setMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function doRollback(round?: number) {
    if (!project) return;
    setBusy(true);
    setMsg(null);
    try {
      const res = await apiRollback(project, "STYLE_REFINE", round);
      // Rollback only flips the state pointer (no run to watch), so stay in the
      // history view and show the returned message rather than auto-leaving —
      // setViewing(null) here would unmount before the banner paints.
      setMsg(res.message);
      await refresh();
      onHistoryChange?.();
    } catch (err) {
      setMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function renderRedoButtons() {
    switch (viewing.phase) {
      case "MODEL_SELECT":
        return (
          <>
            <Button
              variant="red"
              size="sm"
              disabled={busy}
              onClick={() => doRun("set-pass", { pass_num: viewing.pass })}
            >
              切换到此 pass 继续
            </Button>
            <Button
              variant="blue"
              size="sm"
              disabled={busy}
              onClick={() => doRun("retest-models", {})}
            >
              重测模型(开新 pass)
            </Button>
          </>
        );
      case "STYLE_REFINE":
        return (
          <Button
            variant="red"
            size="sm"
            disabled={busy}
            onClick={() => doRollback(viewing.round)}
          >
            回到此轮重做
          </Button>
        );
      case "BATCH_T2I":
      case "BATCH_I2I":
        return (
          <Button
            variant="red"
            size="sm"
            disabled={busy}
            onClick={() => doRun("design-cases", {})}
          >
            重新设计案例
          </Button>
        );
      default:
        return null;
    }
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      <div className="flex items-center gap-3">
        <Button variant="outline" size="sm" onClick={() => setViewing(null)}>
          <ArrowLeft />
          回到当前
        </Button>
        <h1 className="font-black uppercase tracking-tight">{viewing.phase}</h1>
      </div>

      <HistoryPanel
        phase={viewing.phase}
        history={history}
        onPick={(slice) => {
          setMsg(null);
          setViewing(slice);
        }}
        activeSlice={viewing}
      />

      {showBanner && (
        <div
          className={`flex flex-wrap items-center gap-3 border-2 border-foreground bg-bauhaus-yellow p-3 font-bold ${SHADOW_SM}`}
        >
          <span>
            正在查看历史 — {viewing.phase} {sliceLabel(viewing)}（只读）
          </span>
          {renderRedoButtons()}
        </div>
      )}

      {msg && (
        <div className="border-2 border-foreground bg-white p-3 text-sm font-bold">
          {msg}
        </div>
      )}

      {concrete ? (
        gallery ? (
          <GalleryGrid groups={gallery.groups} refImages={gallery.ref_images} />
        ) : (
          <p className="text-sm font-bold uppercase tracking-wide text-foreground/40">
            加载图库…
          </p>
        )
      ) : (
        <p className="text-sm font-bold uppercase tracking-wide text-foreground/40">
          选择一份查看
        </p>
      )}
    </div>
  );
}
