import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ACCENTS } from "@/lib/bauhaus";
import type { ActionPlan } from "@/lib/types";

interface PlanPreviewProps {
  plan: ActionPlan;
  onConfirm: () => void;
  onCancel: () => void;
  disabled?: boolean;
}

/** Render a step's args as a compact `key=value` line, or null when empty. */
function argsLine(args?: Record<string, unknown>): string | null {
  if (!args) return null;
  const parts = Object.entries(args)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : String(v)}`);
  return parts.length > 0 ? parts.join(", ") : null;
}

/**
 * A plan-preview card: shows the planner's summary, the ordered steps, an
 * optional loop line, and the "停在哪" stop summary, with confirm / cancel
 * buttons. `disabled` blocks confirm while any run is in flight (only one run
 * is allowed at a time).
 */
export function PlanPreview({
  plan,
  onConfirm,
  onCancel,
  disabled,
}: PlanPreviewProps) {
  return (
    <Card className="relative gap-4 py-5">
      {/* Geometric corner accent */}
      <span
        aria-hidden
        className="absolute right-3 top-3 h-4 w-4 border-2 border-foreground"
        style={{ backgroundColor: ACCENTS[1] }}
      />
      <CardContent className="space-y-4 px-5">
        <p className="font-black uppercase tracking-tight">{plan.summary}</p>

        <ol className="space-y-2">
          {plan.steps.map((step, i) => {
            const args = argsLine(step.args);
            return (
              <li key={i} className="flex items-center gap-2">
                <span
                  className="flex h-5 w-5 shrink-0 items-center justify-center border-2 border-foreground text-xs font-black text-white"
                  style={{ backgroundColor: ACCENTS[i % ACCENTS.length] }}
                >
                  {i + 1}
                </span>
                <span className="min-w-0">
                  <span className="font-bold uppercase">{step.name}</span>
                  {step.description && (
                    <span className="ml-1.5 font-medium text-foreground/70">
                      {step.description}
                    </span>
                  )}
                  {args && (
                    <span className="ml-1.5 font-mono text-xs text-foreground/70">
                      ({args})
                    </span>
                  )}
                </span>
              </li>
            );
          })}
        </ol>

        {plan.loop && (
          <p className="font-medium text-foreground/70">
            循环：步骤 {plan.loop.start_step + 1}–{plan.loop.end_step + 1}，最多{" "}
            {plan.loop.max_iterations} 轮
          </p>
        )}

        {plan.stop_summary && (
          <p className="font-medium text-foreground/70">
            停在哪：{plan.stop_summary}
          </p>
        )}

        <div className="flex items-center gap-3 pt-1">
          <Button variant="red" size="sm" onClick={onConfirm} disabled={disabled}>
            确认执行
          </Button>
          <Button variant="outline" size="sm" onClick={onCancel}>
            取消
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
