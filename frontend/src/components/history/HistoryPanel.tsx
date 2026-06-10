import { HistoryTile } from "./HistoryTile";
import type { History, Phase, ViewSlice } from "@/lib/types";

interface HistoryPanelProps {
  phase: Phase;
  history: History;
  onPick: (slice: ViewSlice) => void;
  activeSlice: ViewSlice | null;
}

const pad = (n: number) => String(n).padStart(3, "0");

function MutedNote({ children }: { children: React.ReactNode }) {
  return <p className="text-sm font-bold text-muted">{children}</p>;
}

export function HistoryPanel({
  phase,
  history,
  onPick,
  activeSlice,
}: HistoryPanelProps) {
  const cur = history.current;

  function renderBody() {
    switch (phase) {
      case "MODEL_SELECT": {
        const passes = history.model_select.passes;
        if (passes.length === 0) return <MutedNote>无历史记录</MutedNote>;
        return (
          <div className="flex flex-wrap gap-3">
            {passes.map((p, i) => (
              <HistoryTile
                key={p}
                label={`pass-${pad(p)}`}
                accentIndex={i}
                active={
                  activeSlice?.phase === "MODEL_SELECT" && activeSlice.pass === p
                }
                current={cur.phase === "MODEL_SELECT" && cur.pass === p}
                onClick={() => onPick({ phase: "MODEL_SELECT", pass: p })}
              />
            ))}
          </div>
        );
      }
      case "STYLE_REFINE": {
        const entries = Object.entries(history.style_refine.passes);
        if (entries.length === 0) return <MutedNote>无历史记录</MutedNote>;
        return (
          <div className="flex flex-col gap-5">
            {entries.map(([passStr, rounds]) => {
              const passNum = Number(passStr);
              return (
                <div key={passStr} className="flex flex-col gap-2">
                  <span className="text-xs font-bold text-muted">
                    pass-{pad(passNum)}
                  </span>
                  {rounds.length === 0 ? (
                    <MutedNote>无轮次</MutedNote>
                  ) : (
                    <div className="flex flex-wrap gap-3">
                      {rounds.map((r, i) => (
                        <HistoryTile
                          key={r}
                          label={`round-${pad(r)}`}
                          accentIndex={i}
                          active={
                            activeSlice?.phase === "STYLE_REFINE" &&
                            activeSlice.pass === passNum &&
                            activeSlice.round === r
                          }
                          current={
                            cur.phase === "STYLE_REFINE" &&
                            cur.pass === passNum &&
                            cur.round === r
                          }
                          onClick={() =>
                            onPick({
                              phase: "STYLE_REFINE",
                              pass: passNum,
                              round: r,
                            })
                          }
                        />
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        );
      }
      case "BATCH_T2I": {
        const batches = history.batch_t2i.batches;
        if (batches.length === 0) return <MutedNote>无历史记录</MutedNote>;
        return (
          <div className="flex flex-wrap gap-3">
            {batches.map((b, i) => (
              <HistoryTile
                key={b}
                label={`batch-${pad(b)}`}
                accentIndex={i}
                active={
                  activeSlice?.phase === "BATCH_T2I" && activeSlice.batch === b
                }
                current={cur.phase === "BATCH_T2I" && cur.batch === b}
                onClick={() => onPick({ phase: "BATCH_T2I", batch: b })}
              />
            ))}
          </div>
        );
      }
      case "BATCH_I2I": {
        const batches = history.batch_i2i.batches;
        if (batches.length === 0) return <MutedNote>无历史记录</MutedNote>;
        return (
          <div className="flex flex-wrap gap-3">
            {batches.map((b, i) => (
              <HistoryTile
                key={b}
                label={`batch-${pad(b)}`}
                accentIndex={i}
                active={
                  activeSlice?.phase === "BATCH_I2I" && activeSlice.batch === b
                }
                current={cur.phase === "BATCH_I2I" && cur.batch === b}
                onClick={() => onPick({ phase: "BATCH_I2I", batch: b })}
              />
            ))}
          </div>
        );
      }
      default:
        // INIT / COMPLETED
        return <MutedNote>无历史记录</MutedNote>;
    }
  }

  return (
    <section className="flex flex-col gap-4">
      <h2
        className="font-extrabold text-foreground"
        style={{ fontFamily: "Nunito, sans-serif" }}
      >
        {phase} 历史
      </h2>
      {renderBody()}
    </section>
  );
}
