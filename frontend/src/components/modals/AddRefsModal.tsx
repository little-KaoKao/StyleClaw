import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { addRefs } from "@/lib/api";
import { useProject } from "@/hooks/useProject";

import { Dropzone } from "./Dropzone";

interface AddRefsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectName: string | null;
}

export function AddRefsModal({
  open,
  onOpenChange,
  projectName,
}: AddRefsModalProps) {
  const { refresh } = useProject();

  const [files, setFiles] = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setFiles([]);
      setSubmitting(false);
      setError(null);
    }
  }, [open]);

  const canSubmit = !!projectName && files.length > 0 && !submitting;

  async function handleSubmit() {
    if (!projectName || files.length === 0) return;
    setSubmitting(true);
    setError(null);

    const form = new FormData();
    for (const f of files) form.append("files", f);

    try {
      const result = await addRefs(projectName, form);
      if (result.ok) {
        await refresh();
        onOpenChange(false);
      } else {
        setError(result.message || "上传失败，请重试");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "上传失败，请重试");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>添加 i2i 参考图</DialogTitle>
          <DialogDescription>
            上传图片用于 image-to-image 批量测试。
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <Dropzone files={files} onFilesChange={setFiles} disabled={submitting} />
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={submitting}
          >
            取消
          </Button>
          <Button onClick={handleSubmit} disabled={!canSubmit}>
            {submitting && <Loader2 className="animate-spin" />}
            上传并添加
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
