import {
  ActionButton,
  PanelShell,
  makeRunHandler,
  type PanelProps,
} from "./shared";

export function ModelSelectPanel(panel: PanelProps) {
  const { run, runStatus, onSelectModel } = panel;
  const running = runStatus === "running";

  return (
    <PanelShell
      panel={panel}
      title="② 选模型 MODEL_SELECT"
      description="在 prompt-sref 与 prompt-only 两种条件下对比各模型，挑选最能还原风格的模型与变体。"
      actions={
        <>
          <ActionButton
            label="生成图片"
            variant="default"
            disabled={running}
            onClick={makeRunHandler(run, "generate")}
          />
          <ActionButton
            label="轮询结果"
            variant="secondary"
            disabled={running}
            onClick={makeRunHandler(run, "poll")}
          />
          <ActionButton
            label="评分"
            variant="secondary"
            disabled={running}
            onClick={makeRunHandler(run, "evaluate")}
          />
          <ActionButton
            label="确认选择模型"
            variant="outline"
            disabled={running}
            onClick={() => onSelectModel?.()}
          />
          <ActionButton
            label="重测模型"
            variant="soft"
            disabled={running}
            onClick={makeRunHandler(run, "retest-models")}
          />
        </>
      }
    />
  );
}
