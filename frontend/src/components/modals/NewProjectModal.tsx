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
import { Input } from "@/components/ui/input";
import { createProject } from "@/lib/api";
import { useProject } from "@/hooks/useProject";
import { useAppStore } from "@/store/app-store";

import { Dropzone } from "./Dropzone";

interface NewProjectModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function NewProjectModal({ open, onOpenChange }: NewProjectModalProps) {
  const setCurrentProject = useAppStore((s) => s.setCurrentProject);
  const { refresh } = useProject();

  const [name, setName] = useState("");
  const [ipInfo, setIpInfo] = useState("");
  const [description, setDescription] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Modals stay mounted, so reset every field whenever the dialog opens.
  useEffect(() => {
    if (open) {
      setName("");
      setIpInfo("");
      setDescription("");
      setFiles([]);
      setSubmitting(false);
      setError(null);
    }
  }, [open]);

  const canSubmit = name.trim().length > 0 && files.length > 0 && !submitting;

  async function handleSubmit() {
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);

    const form = new FormData();
    form.append("name", name.trim());
    form.append("ip_info", ipInfo.trim());
    form.append("description", description.trim());
    form.append("force", "false");
    for (const f of files) form.append("files", f);

    try {
      const result = await createProject(form);
      if (result.ok) {
        setCurrentProject(name.trim());
        await refresh();
        onOpenChange(false);
      } else {
        setError(result.message || "创建失败，请重试");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "创建失败，请重试");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>新建项目</DialogTitle>
          <DialogDescription>
            从一组参考图创建项目，开始风格探索。
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-sm font-bold text-muted">
              项目名 <span className="text-clay-accent-alt">*</span>
            </label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="spider-verse"
              disabled={submitting}
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-bold text-muted">IP 信息</label>
            <Input
              value={ipInfo}
              onChange={(e) => setIpInfo(e.target.value)}
              placeholder="Spider-Verse 动画风格"
              disabled={submitting}
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-bold text-muted">描述</label>
            <Input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="测试 Spider-Verse 视觉风格提取"
              disabled={submitting}
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-bold text-muted">
              参考图 <span className="text-clay-accent-alt">*</span>
            </label>
            <Dropzone
              files={files}
              onFilesChange={setFiles}
              disabled={submitting}
            />
          </div>

          {error && (
            <p className="rounded-2xl bg-gradient-to-br from-[#FCE7F3] to-[#FBCFE8] px-3 py-2 font-bold text-clay-accent-alt">
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
          <Button variant="primary" onClick={handleSubmit} disabled={!canSubmit}>
            {submitting && <Loader2 className="animate-spin" />}
            创建项目
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
