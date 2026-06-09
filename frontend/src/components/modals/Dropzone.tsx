import { useRef, useState } from "react";
import { ImagePlus, X } from "lucide-react";

import { cn } from "@/lib/utils";

interface DropzoneProps {
  /** Currently selected files (controlled by the parent). */
  files: File[];
  /** Called with the next file list whenever the selection changes. */
  onFilesChange: (files: File[]) => void;
  disabled?: boolean;
}

/**
 * A small drag-and-drop image picker that also supports click-to-browse.
 * The parent owns the `File[]` state; this component is purely presentational
 * plus the file-collection plumbing.
 */
export function Dropzone({ files, onFilesChange, disabled }: DropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function addFiles(list: FileList | null) {
    if (!list) return;
    const incoming = Array.from(list).filter((f) =>
      f.type.startsWith("image/")
    );
    if (incoming.length === 0) return;
    onFilesChange([...files, ...incoming]);
  }

  function removeAt(index: number) {
    onFilesChange(files.filter((_, i) => i !== index));
  }

  return (
    <div className="space-y-2">
      <div
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (disabled) return;
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          // Required — without preventDefault the browser handles the drop and
          // onDrop never fires.
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (disabled) return;
          addFiles(e.dataTransfer.files);
        }}
        className={cn(
          "flex flex-col items-center justify-center gap-1.5 rounded-md border border-dashed px-4 py-6 text-center transition-colors",
          disabled
            ? "cursor-not-allowed opacity-60"
            : "cursor-pointer hover:bg-accent/50",
          dragging && "border-ring bg-accent/50"
        )}
      >
        <ImagePlus className="size-5 text-zinc-400" />
        <p className="text-sm">
          拖拽图片到此，或<span className="text-zinc-900">点击选择</span>
        </p>
        <p className="text-xs text-muted-foreground">支持多张，仅图片格式</p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            addFiles(e.target.files);
            // Reset so re-selecting the same file fires onChange again.
            e.target.value = "";
          }}
        />
      </div>

      {files.length > 0 && (
        <ul className="space-y-1">
          <li className="text-xs text-muted-foreground">
            已选 {files.length} 张
          </li>
          {files.map((f, i) => (
            <li
              key={`${f.name}-${i}`}
              className="flex items-center justify-between gap-2 rounded-md bg-zinc-50 px-2 py-1 text-xs"
            >
              <span className="truncate">{f.name}</span>
              <button
                type="button"
                disabled={disabled}
                onClick={() => removeAt(i)}
                className="shrink-0 rounded p-0.5 text-zinc-400 hover:bg-zinc-200 hover:text-zinc-700 disabled:pointer-events-none"
                aria-label={`移除 ${f.name}`}
              >
                <X className="size-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
