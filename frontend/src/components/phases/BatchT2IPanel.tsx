import {
  ActionButton,
  PanelShell,
  makeRunHandler,
  type PanelProps,
} from "./shared";

export function BatchT2IPanel(panel: PanelProps) {
  const { run, runStatus, onAddRefs } = panel;
  const running = runStatus === "running";

  return (
    <PanelShell
      panel={panel}
      title="④ 批量 T2I"
      description="用 100 个多样化案例测试风格触发词的泛化能力（10 类各 10 个），生成报告后人工复核。"
      actions={
        <>
          <ActionButton
            label="设计 100 案例"
            variant="default"
            disabled={running}
            onClick={makeRunHandler(run, "design-cases")}
          />
          <ActionButton
            label="提交批量生成"
            variant="secondary"
            disabled={running}
            onClick={makeRunHandler(run, "batch-submit")}
          />
          <ActionButton
            label="轮询"
            variant="secondary"
            disabled={running}
            onClick={makeRunHandler(run, "poll")}
          />
          <ActionButton
            label="生成报告"
            variant="secondary"
            disabled={running}
            onClick={makeRunHandler(run, "report")}
          />
          <ActionButton
            label="添加 i2i 参考图"
            variant="outline"
            disabled={running}
            onClick={() => onAddRefs?.()}
          />
        </>
      }
    />
  );
}
