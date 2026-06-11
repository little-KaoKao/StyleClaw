import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ORB_GRADIENTS } from "@/lib/clay";
import { cn } from "@/lib/utils";
import type { ActionPlan } from "@/lib/types";

const NUNITO = { fontFamily: "Nunito, sans-serif" } as const;

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
    <Card className="gap-4 rounded-[24px] bg-white/70 p-5 backdrop-blur-xl shadow-clayCard">
      <CardContent className="space-y-4 px-0">
        <p className="font-extrabold" style={NUNITO}>
          {plan.summary}
        </p>

        <ol className="space-y-2">
          {plan.steps.map((step, i) => {
            const args = argsLine(step.args);
            return (
              <li key={i} className="flex items-center gap-2">
                <span
                  className={cn(
                    "flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-gradient-to-br text-xs font-extrabold text-white shadow-clayButton",
                    ORB_GRADIENTS[i % ORB_GRADIENTS.length]
                  )}
                  style={NUNITO}
                >
                  {i + 1}
                </span>
                <span className="min-w-0">
                  <span className="font-bold">{step.name}</span>
                  {step.description && (
                    <span className="ml-1.5 font-medium text-muted">
                      {step.description}
                    </span>
                  )}
                  {args && (
                    <span className="ml-1.5 font-mono text-xs text-muted">
                      ({args})
                    </span>
                  )}
                </span>
              </li>
            );
          })}
        </ol>

        {plan.loop && (
          <p className="font-medium text-muted">
            循环：步骤 {plan.loop.start_step + 1}–{plan.loop.end_step + 1}，最多{" "}
            {plan.loop.max_iterations} 轮
          </p>
        )}

        {plan.stop_summary && (
          <p className="font-medium text-muted">停在哪：{plan.stop_summary}</p>
        )}

        <div className="flex items-center gap-3 pt-1">
          <Button variant="primary" size="sm" onClick={onConfirm} disabled={disabled}>
            确认执行
          </Button>
          <Button variant="soft" size="sm" onClick={onCancel}>
            取消
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
