import { ArrowRight, Loader2, RefreshCw, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";

interface RunControlsProps {
  status: string;
  onRefresh: () => void;
  onReset?: () => void;
  nextLabel?: string;
  onNext?: () => void;
}

export function RunControls({
  status,
  onRefresh,
  onReset,
  nextLabel,
  onNext,
}: RunControlsProps) {
  if (status === "idle") return null;

  if (status === "running") {
    return (
      <Button size="sm" disabled>
        <Loader2 className="animate-spin" />
        运行中…
      </Button>
    );
  }

  if (status === "error") {
    return (
      <div className="flex items-center gap-3">
        {onReset && (
          <Button size="sm" variant="destructive" onClick={() => onReset()}>
            <RotateCcw />
            重试
          </Button>
        )}
        <Button size="sm" variant="outline" onClick={onRefresh}>
          <RefreshCw />
          刷新数据
        </Button>
      </div>
    );
  }

  // status === "done"
  return (
    <div className="flex items-center gap-3">
      {onNext && (
        <Button size="sm" variant="primary" onClick={() => onNext()}>
          {nextLabel ?? "继续"}
          <ArrowRight />
        </Button>
      )}
      <Button size="sm" variant="secondary" onClick={onRefresh}>
        <RefreshCw />
        刷新数据
      </Button>
    </div>
  );
}
