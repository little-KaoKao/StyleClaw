import { useEffect, useRef } from "react";
import { Check, Loader2, X } from "lucide-react";

import { ORB_GRADIENTS } from "@/lib/clay";
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
      <p className="text-xs font-bold uppercase tracking-widest text-muted">
        准备中…
      </p>
    ) : null;
  }

  return (
    <div className="flex flex-col gap-4">
      {steps.length > 0 && (
        <ol className="flex flex-col gap-3">
          {steps.map((step) => {
            const running = !step.done;
            return (
              <li key={step.index} className="flex items-center gap-3 text-sm">
                <span
                  className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br text-sm font-black text-white shadow-clayButton ${
                    ORB_GRADIENTS[step.index % ORB_GRADIENTS.length]
                  }`}
                  style={{ fontFamily: "Nunito, sans-serif" }}
                >
                  {step.index + 1}
                </span>
                <span
                  className="font-bold text-foreground"
                  style={{ fontFamily: "Nunito, sans-serif" }}
                >
                  {step.name}
                </span>
                {running ? (
                  <Loader2 className="h-5 w-5 shrink-0 animate-spin text-clay-accent" />
                ) : step.ok ? (
                  <Check className="h-5 w-5 shrink-0 text-clay-emerald" />
                ) : (
                  <X className="h-5 w-5 shrink-0 text-clay-accent-alt" />
                )}
                {step.done && step.summary ? (
                  <span className="text-sm text-muted">{step.summary}</span>
                ) : step.description ? (
                  <span className="text-sm text-muted">{step.description}</span>
                ) : null}
              </li>
            );
          })}
        </ol>
      )}

      {llmBuffer && (
        <div className="rounded-[20px] bg-clay-surface p-4 shadow-clayPressed">
          <div className="text-xs font-bold uppercase tracking-widest text-muted">
            LLM 输出
          </div>
          <div
            ref={bufferRef}
            className="mt-2 max-h-40 overflow-y-auto whitespace-pre-wrap font-mono text-sm leading-relaxed text-foreground"
          >
            {llmBuffer}
            {status === "running" && (
              <span className="ml-0.5 inline-block animate-pulse">▌</span>
            )}
          </div>
        </div>
      )}

      {status === "error" && errorEvent?.type === "error" && (
        <div className="rounded-[20px] bg-gradient-to-br from-[#FCE7F3] to-[#FBCFE8] p-4 font-bold text-clay-accent-alt shadow-clayCard">
          <p>运行出错</p>
          <p className="mt-1 break-words text-sm">{errorEvent.message}</p>
          {errorEvent.detail && (
            <p className="mt-1 break-words text-sm opacity-80">
              {errorEvent.detail}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
