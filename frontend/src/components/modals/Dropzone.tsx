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
          "flex flex-col items-center justify-center gap-2 rounded-[24px] bg-clay-surface p-6 text-center shadow-clayPressed transition-all",
          disabled
            ? "cursor-not-allowed opacity-60"
            : "cursor-pointer hover:bg-white",
          dragging && "bg-white outline-2 outline-dashed outline-clay-accent/40"
        )}
      >
        <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-purple-400 to-purple-600 text-white shadow-clayButton">
          <ImagePlus className="h-6 w-6 shrink-0" />
        </span>
        <p className="font-bold text-muted">拖拽图片到此，或点击选择</p>
        <p className="text-xs font-medium text-muted">支持多张，仅图片格式</p>
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
        <ul className="space-y-1.5">
          <li className="text-xs font-bold text-muted">已选 {files.length} 张</li>
          {files.map((f, i) => (
            <li
              key={`${f.name}-${i}`}
              className="flex items-center gap-2 rounded-2xl bg-white/70 px-3 py-2 text-xs font-medium shadow-clayCard"
            >
              <span className="flex-1 truncate">{f.name}</span>
              <button
                type="button"
                disabled={disabled}
                onClick={() => removeAt(i)}
                className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-clay-surface text-muted shadow-clayPressed transition-all hover:bg-clay-accent-alt/10 hover:text-clay-accent-alt disabled:pointer-events-none"
                aria-label={`移除 ${f.name}`}
              >
                <X className="h-4 w-4 shrink-0" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
