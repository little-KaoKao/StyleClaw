import {
  ActionButton,
  PanelShell,
  makeRunHandler,
  type PanelProps,
} from "./shared";

export function BatchI2IPanel(panel: PanelProps) {
  const { run, runStatus } = panel;
  const running = runStatus === "running";

  return (
    <PanelShell
      panel={panel}
      title="⑤ 批量 I2I"
      description="用 image-to-image 测试风格触发词：在已有图像上施加风格，生成报告后标记完成。"
      actions={
        <>
          <ActionButton
            label="提交 i2i 批量"
            variant="blue"
            disabled={running}
            onClick={makeRunHandler(run, "batch-submit")}
          />
          <ActionButton
            label="轮询"
            variant="yellow"
            disabled={running}
            onClick={makeRunHandler(run, "poll")}
          />
          <ActionButton
            label="生成报告"
            variant="blue"
            disabled={running}
            onClick={makeRunHandler(run, "report")}
          />
          <ActionButton
            label="标记完成"
            variant="red"
            disabled={running}
            onClick={makeRunHandler(run, "approve", { target: "completed" })}
          />
          <ActionButton
            label="返回 T2I"
            variant="outline"
            disabled={running}
            onClick={makeRunHandler(run, "back-to-t2i")}
          />
        </>
      }
    />
  );
}
