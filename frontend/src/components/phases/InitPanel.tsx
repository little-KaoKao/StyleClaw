import { Sparkles } from "lucide-react";

import {
  ActionButton,
  PanelShell,
  makeRunHandler,
  type PanelProps,
} from "./shared";

export function InitPanel(panel: PanelProps) {
  const { run, runStatus } = panel;
  const running = runStatus === "running";

  return (
    <PanelShell
      panel={panel}
      title="① 初始化 INIT"
      description="分析参考图，提取风格并生成初始 trigger phrase。"
      actions={
        <ActionButton
          label="分析风格"
          variant="red"
          icon={<Sparkles />}
          disabled={running}
          onClick={makeRunHandler(run, "analyze")}
        />
      }
    />
  );
}
