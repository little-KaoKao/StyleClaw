import { useEffect, useRef } from "react";
import { Check, Loader2, X } from "lucide-react";

import { ACCENTS, SHADOW_SM } from "@/lib/bauhaus";
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
      <p className="text-xs font-bold uppercase tracking-wide text-foreground/60">
        准备中…
      </p>
    ) : null;
  }

  return (
    <div className="flex flex-col gap-4">
      {steps.length > 0 && (
        <ol className="flex flex-col gap-2">
          {steps.map((step) => {
            const running = !step.done;
            return (
              <li key={step.index} className="flex items-center gap-3 text-sm">
                <span
                  className="flex h-7 w-7 shrink-0 items-center justify-center rounded-none border-2 border-foreground text-sm font-black"
                  style={{
                    backgroundColor: ACCENTS[step.index % ACCENTS.length],
                  }}
                >
                  {step.index + 1}
                </span>
                <span className="font-bold uppercase tracking-wide">
                  {step.name}
                </span>
                {running ? (
                  <Loader2 className="h-5 w-5 shrink-0 animate-spin text-foreground/50" />
                ) : step.ok ? (
                  <Check className="h-5 w-5 shrink-0 text-bauhaus-blue" />
                ) : (
                  <X className="h-5 w-5 shrink-0 text-bauhaus-red" />
                )}
                {step.done && step.summary ? (
                  <span className="text-sm font-medium text-foreground/70">
                    {step.summary}
                  </span>
                ) : step.description ? (
                  <span className="text-sm font-medium text-foreground/70">
                    {step.description}
                  </span>
                ) : null}
              </li>
            );
          })}
        </ol>
      )}

      {llmBuffer && (
        <div className="rounded-none border-2 border-foreground bg-[#FFF9C4] p-3">
          <div className="text-xs font-bold uppercase tracking-widest text-foreground">
            LLM 输出
          </div>
          <div
            ref={bufferRef}
            className="mt-2 max-h-40 overflow-y-auto font-mono text-sm leading-relaxed whitespace-pre-wrap text-foreground"
          >
            {llmBuffer}
            {status === "running" && (
              <span className="ml-0.5 inline-block animate-pulse">▌</span>
            )}
          </div>
        </div>
      )}

      {status === "error" && errorEvent?.type === "error" && (
        <div
          className={`rounded-none border-2 border-foreground bg-bauhaus-red p-3 font-bold text-white ${SHADOW_SM}`}
        >
          <p className="uppercase tracking-wide">运行出错</p>
          <p className="mt-1 text-sm break-words">{errorEvent.message}</p>
          {errorEvent.detail && (
            <p className="mt-1 text-sm break-words opacity-80">
              {errorEvent.detail}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
