import { useState } from "react";

import { Input } from "@/components/ui/input";
import {
  ActionButton,
  PanelShell,
  makeRunHandler,
  type PanelProps,
} from "./shared";

export function StyleRefinePanel(panel: PanelProps) {
  const { run, runStatus } = panel;
  const running = runStatus === "running";
  const [direction, setDirection] = useState("");

  const onRefine = () => {
    const trimmed = direction.trim();
    makeRunHandler(run, "refine", trimmed ? { direction: trimmed } : undefined)();
  };

  return (
    <PanelShell
      panel={panel}
      title="③ 精炼 STYLE_REFINE"
      description="逐轮精炼 trigger phrase：精炼 → 生成 → 轮询 → 评分，直到五个维度都达标，再批准进入批量测试。"
      extra={
        <div className="max-w-md space-y-1">
          <label className="text-xs font-medium text-muted-foreground">
            精炼方向（可选）
          </label>
          <Input
            value={direction}
            onChange={(e) => setDirection(e.target.value)}
            placeholder="例如：提高对比度，加入半调网点"
            disabled={running}
          />
        </div>
      }
      actions={
        <>
          <ActionButton
            label="精炼一轮"
            variant="default"
            disabled={running}
            onClick={onRefine}
          />
          <ActionButton
            label="生成"
            disabled={running}
            onClick={makeRunHandler(run, "generate")}
          />
          <ActionButton
            label="轮询"
            disabled={running}
            onClick={makeRunHandler(run, "poll")}
          />
          <ActionButton
            label="评分"
            disabled={running}
            onClick={makeRunHandler(run, "evaluate")}
          />
          <ActionButton
            label="批准进入批量测试"
            disabled={running}
            onClick={makeRunHandler(run, "approve", { target: "batch-t2i" })}
          />
        </>
      }
    />
  );
}
