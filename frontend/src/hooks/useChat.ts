import { useCallback, useRef, useState } from "react";

import { previewPlan } from "@/lib/api";
import { useAppStore } from "@/store/app-store";
import type { ActionPlan } from "@/lib/types";

export type ChatRole = "user" | "assistant";
export type ChatKind = "text" | "plan";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  kind: ChatKind;
  text?: string;
  plan?: ActionPlan;
}

/**
 * Manages the right-column chat transcript: user intents, the assistant's
 * plan previews, and inline error messages. Run execution itself lives in
 * the shared zustand store — this hook only owns the message list and the
 * in-flight `previewPlan` call.
 */
export function useChat() {
  const project = useAppStore((s) => s.currentProject);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState(false);

  // Monotonic id source — avoids any non-secure-context randomUUID edge case
  // and keeps ids stable/predictable for React keys.
  const counter = useRef(0);
  const nextId = useCallback(() => {
    counter.current += 1;
    return `m${counter.current}`;
  }, []);

  const append = useCallback((msg: ChatMessage) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const send = useCallback(
    async (intent: string) => {
      const trimmed = intent.trim();
      if (!trimmed || !project || pending) return;

      append({ id: nextId(), role: "user", kind: "text", text: trimmed });
      setPending(true);
      try {
        const plan = await previewPlan(project, trimmed);
        append({ id: nextId(), role: "assistant", kind: "plan", plan });
      } catch (err) {
        const text = err instanceof Error ? err.message : String(err);
        append({
          id: nextId(),
          role: "assistant",
          kind: "text",
          text: `无法生成计划：${text}`,
        });
      } finally {
        setPending(false);
      }
    },
    [project, pending, append, nextId]
  );

  // Drop a plan card the user dismissed via "取消".
  const dismissPlan = useCallback((id: string) => {
    setMessages((prev) => prev.filter((m) => m.id !== id));
  }, []);

  return { messages, pending, send, dismissPlan };
}
