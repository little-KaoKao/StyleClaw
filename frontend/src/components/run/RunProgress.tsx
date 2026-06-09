import { useEffect, useRef } from "react";
import { Check, Loader2, X } from "lucide-react";

import { cn } from "@/lib/utils";
import type { WsEvent } from "@/lib/types";

interface RunProgressProps {
  events: WsEvent[];
  llmBuffer: string;
  status: string;
}

interface DerivedStep {
  index: number;
  name: string;
  description: string;
  done: boolean;
  ok: boolean;
  summary: string;
}

/**
 * Pair step_start / step_done events on their `index` (robust to ordering)
 * into an ordered list of steps. A step with a start but no done is "running".
 */
function deriveSteps(events: WsEvent[]): DerivedStep[] {
  const byIndex = new Map<number, DerivedStep>();
  for (const ev of events) {
    if (ev.type === "step_start") {
      const existing = byIndex.get(ev.index);
      byIndex.set(ev.index, {
        index: ev.index,
        name: ev.name,
        description: ev.description,
        done: existing?.done ?? false,
        ok: existing?.ok ?? false,
        summary: existing?.summary ?? "",
      });
    } else if (ev.type === "step_done") {
      const existing = byIndex.get(ev.index);
      byIndex.set(ev.index, {
        index: ev.index,
        name: existing?.name ?? ev.name,
        description: existing?.description ?? "",
        done: true,
        ok: ev.status === "ok",
        summary: ev.summary,
      });
    }
  }
  return [...byIndex.values()].sort((a, b) => a.index - b.index);
}

export function RunProgress({ events, llmBuffer, status }: RunProgressProps) {
  const bufferRef = useRef<HTMLDivElement>(null);

  // Auto-scroll the token stream to the bottom as deltas arrive.
  useEffect(() => {
    const el = bufferRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [llmBuffer]);

  const steps = deriveSteps(events);
  const errorEvent = events.find((e) => e.type === "error");

  if (events.length === 0) {
    return status === "running" ? (
      <p className="text-xs text-muted-foreground">准备中…</p>
    ) : null;
  }

  return (
    <div className="flex flex-col gap-3">
      {steps.length > 0 && (
        <ol className="flex flex-col gap-1.5">
          {steps.map((step) => {
            const running = !step.done;
            return (
              <li key={step.index} className="flex items-start gap-2 text-sm">
                <span className="mt-0.5 shrink-0">
                  {running ? (
                    <Loader2 className="size-3.5 animate-spin text-zinc-400" />
                  ) : step.ok ? (
                    <Check className="size-3.5 text-emerald-600" />
                  ) : (
                    <X className="size-3.5 text-destructive" />
                  )}
                </span>
                <div className="min-w-0">
                  <span
                    className={cn(
                      "font-medium",
                      running && "text-zinc-900",
                      step.done && !step.ok && "text-destructive"
                    )}
                  >
                    {step.name}
                  </span>
                  {step.done && step.summary ? (
                    <span className="ml-1.5 text-muted-foreground">
                      {step.summary}
                    </span>
                  ) : step.description ? (
                    <span className="ml-1.5 text-muted-foreground">
                      {step.description}
                    </span>
                  ) : null}
                </div>
              </li>
            );
          })}
        </ol>
      )}

      {llmBuffer && (
        <div className="rounded-md border bg-zinc-50">
          <div className="border-b px-2.5 py-1 text-[10px] font-medium tracking-wide text-muted-foreground uppercase">
            LLM 输出
          </div>
          <div
            ref={bufferRef}
            className="max-h-40 overflow-y-auto px-2.5 py-1.5 font-mono text-xs leading-relaxed whitespace-pre-wrap text-zinc-600"
          >
            {llmBuffer}
            {status === "running" && (
              <span className="ml-0.5 inline-block animate-pulse text-zinc-400">
                ↓
              </span>
            )}
          </div>
        </div>
      )}

      {status === "error" && errorEvent?.type === "error" && (
        <div className="rounded-md border border-destructive/40 bg-destructive/5 px-2.5 py-2 text-sm text-destructive">
          <p className="font-medium">运行出错</p>
          <p className="mt-0.5 text-xs break-words">{errorEvent.message}</p>
          {errorEvent.detail && (
            <p className="mt-0.5 text-xs break-words opacity-70">
              {errorEvent.detail}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
