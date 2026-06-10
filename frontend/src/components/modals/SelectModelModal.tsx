import { useEffect, useState } from "react";
import { Check, Loader2 } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { selectModel } from "@/lib/api";
import { useProject } from "@/hooks/useProject";
import { cn } from "@/lib/utils";

interface SelectModelModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectName: string | null;
}

const MODELS = ["mj-v7", "niji7", "nb2", "seedream", "gpt-image-2", "n-pro"];

type Variant = "prompt-sref" | "prompt-only";

const VARIANTS: { value: Variant; label: string; hint: string }[] = [
  {
    value: "prompt-sref",
    label: "prompt-sref",
    hint: "trigger + 参考图，风格更贴近",
  },
  {
    value: "prompt-only",
    label: "prompt-only",
    hint: "只有 trigger，更灵活",
  },
];

export function SelectModelModal({
  open,
  onOpenChange,
  projectName,
}: SelectModelModalProps) {
  const { refresh } = useProject();

  const [selected, setSelected] = useState<string[]>([]);
  const [variant, setVariant] = useState<Variant>("prompt-sref");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setSelected([]);
      setVariant("prompt-sref");
      setSubmitting(false);
      setError(null);
    }
  }, [open]);

  function toggle(model: string) {
    setSelected((prev) =>
      prev.includes(model)
        ? prev.filter((m) => m !== model)
        : [...prev, model]
    );
  }

  const canSubmit = !!projectName && selected.length > 0 && !submitting;

  async function handleSubmit() {
    if (!projectName || selected.length === 0) return;
    setSubmitting(true);
    setError(null);
    try {
      const result = await selectModel(projectName, selected.join(","), variant);
      if (result.ok) {
        await refresh();
        onOpenChange(false);
      } else {
        setError(result.message || "选择失败，请重试");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "选择失败，请重试");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>选择模型</DialogTitle>
          <DialogDescription>
            选择用于风格精修的图像生成模型与构图方式。
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-bold uppercase tracking-wider">
              模型 <span className="text-bauhaus-red">*</span>
            </label>
            <div className="grid grid-cols-2 gap-2">
              {MODELS.map((model) => {
                const active = selected.includes(model);
                return (
                  <button
                    key={model}
                    type="button"
                    disabled={submitting}
                    onClick={() => toggle(model)}
                    className={cn(
                      "flex items-center justify-between gap-2 rounded-none border-2 border-foreground px-3 py-2 text-sm font-bold uppercase tracking-wider transition-colors disabled:pointer-events-none disabled:opacity-50",
                      active
                        ? "bg-bauhaus-blue text-white"
                        : "bg-white text-foreground hover:bg-muted"
                    )}
                  >
                    <span className="truncate">{model}</span>
                    {active && <Check className="h-5 w-5 shrink-0" />}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-bold uppercase tracking-wider">
              构图方式 variant
            </label>
            <div className="grid grid-cols-2 gap-2">
              {VARIANTS.map((v) => {
                const active = variant === v.value;
                return (
                  <button
                    key={v.value}
                    type="button"
                    disabled={submitting}
                    onClick={() => setVariant(v.value)}
                    className={cn(
                      "flex flex-col gap-1 rounded-none border-2 border-foreground px-3 py-2 text-left transition-colors disabled:pointer-events-none disabled:opacity-50",
                      active
                        ? "bg-bauhaus-blue text-white"
                        : "bg-white text-foreground hover:bg-muted"
                    )}
                  >
                    <span className="flex items-center gap-2">
                      <span
                        className={cn(
                          "flex h-5 w-5 shrink-0 items-center justify-center rounded-full border-2 border-current"
                        )}
                      >
                        {active && (
                          <span className="h-2 w-2 rounded-full bg-current" />
                        )}
                      </span>
                      <span className="text-sm font-bold uppercase tracking-wider">
                        {v.label}
                      </span>
                    </span>
                    <span
                      className={cn(
                        "text-xs font-medium",
                        active ? "text-white/80" : "text-foreground/70"
                      )}
                    >
                      {v.hint}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {error && (
            <p className="bg-bauhaus-red px-2 py-1 font-bold text-white">
              {error}
            </p>
          )}
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={submitting}
          >
            取消
          </Button>
          <Button variant="red" onClick={handleSubmit} disabled={!canSubmit}>
            {submitting && <Loader2 className="animate-spin" />}
            确认选择
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
