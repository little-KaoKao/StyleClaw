import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
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
    <Card className="gap-3 py-4">
      <CardContent className="space-y-3">
        <p className="text-sm font-semibold text-zinc-900">{plan.summary}</p>

        <ol className="space-y-1.5">
          {plan.steps.map((step, i) => {
            const args = argsLine(step.args);
            return (
              <li key={i} className="text-sm">
                <span className="font-medium text-zinc-900">
                  {i + 1}. {step.name}
                </span>
                {step.description && (
                  <span className="ml-1.5 text-muted-foreground">
                    {step.description}
                  </span>
                )}
                {args && (
                  <span className="ml-1.5 font-mono text-xs text-muted-foreground">
                    ({args})
                  </span>
                )}
              </li>
            );
          })}
        </ol>

        {plan.loop && (
          <p className="text-xs text-muted-foreground">
            循环：步骤 {plan.loop.start_step + 1}–{plan.loop.end_step + 1}，最多{" "}
            {plan.loop.max_iterations} 轮
          </p>
        )}

        {plan.stop_summary && (
          <p className="text-xs text-muted-foreground">
            停在哪：{plan.stop_summary}
          </p>
        )}

        <div className="flex gap-2 pt-1">
          <Button size="sm" onClick={onConfirm} disabled={disabled}>
            确认执行
          </Button>
          <Button size="sm" variant="ghost" onClick={onCancel}>
            取消
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
