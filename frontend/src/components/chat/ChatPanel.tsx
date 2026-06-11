import { useCallback, useEffect, useRef } from "react";
import { useState } from "react";
import { MessageCircle, Send, ChevronRight } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { RunProgress } from "@/components/run/RunProgress";
import { startRun as apiStartRun } from "@/lib/api";
import { useChat } from "@/hooks/useChat";
import { useProject } from "@/hooks/useProject";
import { useAppStore } from "@/store/app-store";
import type { ActionStep } from "@/lib/types";

import { MessageBubble } from "./MessageBubble";
import { PlanPreview } from "./PlanPreview";

export function ChatPanel({ onCollapse }: { onCollapse?: () => void }) {
  const { messages, pending, send, dismissPlan } = useChat();
  const { refresh } = useProject();

  // ChatPanel deliberately does NOT call useRun(): useRun() mounts its own
  // useWebSocket, and PhasePanel already mounts one feeding the shared store.
  // A second mount would double every event (and double the shared llmBuffer,
  // breaking PhasePanel too). Instead we read run state straight from the store
  // and start runs via a thin inline helper. Both panels render the same run.
  const currentProject = useAppStore((s) => s.currentProject);
  const runStatus = useAppStore((s) => s.runStatus);
  const runEvents = useAppStore((s) => s.runEvents);
  const llmBuffer = useAppStore((s) => s.llmBuffer);
  const startRunStore = useAppStore((s) => s.startRun);

  const running = runStatus === "running";
  const disabled = pending || running;

  const [intent, setIntent] = useState("");
  const listRef = useRef<HTMLDivElement>(null);

  // Start a run from a plan's steps. Carries the loop + summary through so a
  // looped plan actually iterates (the backend rebuilds LoopConfig from `loop`).
  const run = useCallback(
    async (steps: ActionStep[], opts?: { loop?: object; summary?: string }) => {
      if (!currentProject) return;
      const { run_id } = await apiStartRun(currentProject, steps, opts);
      startRunStore(run_id);
    },
    [currentProject, startRunStore]
  );

  // Refresh project detail + gallery once each run finishes so the chat-driven
  // changes show up in the main panel. PhasePanel does this too; it's idempotent.
  useEffect(() => {
    if (runStatus === "done") void refresh();
  }, [runStatus, refresh]);

  // Auto-scroll the transcript to the bottom as messages / run output grow.
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, runEvents, llmBuffer]);

  const onSend = () => {
    if (disabled) return;
    const trimmed = intent.trim();
    if (!trimmed) return;
    setIntent("");
    void send(trimmed);
  };

  if (!currentProject) {
    return (
      <div className="flex h-full items-center justify-center p-8 font-bold text-muted">
        请先选择或新建项目
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-2 px-4 py-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-purple-400 to-purple-600 text-white shadow-clayButton">
          <MessageCircle className="h-5 w-5 shrink-0" />
        </span>
        <span
          className="font-extrabold text-foreground"
          style={{ fontFamily: "Nunito, sans-serif" }}
        >
          助手
        </span>
        {onCollapse && (
          <button
            type="button"
            onClick={onCollapse}
            aria-label="收起助手"
            className="ml-auto flex h-8 w-8 items-center justify-center rounded-full text-muted transition-all duration-200 hover:bg-clay-accent/10 hover:text-clay-accent"
          >
            <ChevronRight className="h-5 w-5" />
          </button>
        )}
      </header>
      <div className="mx-4 h-px bg-clay-surface" />

      <div ref={listRef} className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && (
          <p className="text-sm text-muted">
            用自然语言描述你想做什么，例如「帮我精炼一轮再评分」。
          </p>
        )}

        {messages.map((msg) => {
          if (msg.kind === "plan" && msg.plan) {
            return (
              <PlanPreview
                key={msg.id}
                plan={msg.plan}
                disabled={disabled}
                onConfirm={() =>
                  void run(msg.plan!.steps, {
                    loop: msg.plan!.loop ?? undefined,
                    summary: msg.plan!.summary,
                  })
                }
                onCancel={() => dismissPlan(msg.id)}
              />
            );
          }
          return (
            <MessageBubble key={msg.id} role={msg.role} text={msg.text ?? ""} />
          );
        })}

        {pending && (
          <p className="text-xs text-muted">正在生成计划…</p>
        )}

        {/* Live execution stream, shared with the phase panel via the store. */}
        <RunProgress events={runEvents} llmBuffer={llmBuffer} status={runStatus} />
      </div>

      <div className="mx-4 h-px bg-clay-surface" />
      <div className="flex items-center gap-2 p-3">
        <Input
          className="flex-1"
          value={intent}
          onChange={(e) => setIntent(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.nativeEvent.isComposing) {
              e.preventDefault();
              onSend();
            }
          }}
          placeholder="想做什么？"
          disabled={disabled}
        />
        <Button
          variant="primary"
          size="icon"
          className="rounded-2xl"
          onClick={onSend}
          disabled={disabled || !intent.trim()}
          aria-label="发送"
        >
          <Send className="h-5 w-5" />
        </Button>
      </div>
    </div>
  );
}
