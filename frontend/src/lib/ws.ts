import { useCallback, useEffect, useRef, useState } from "react";
import type { WsEvent } from "./types";

interface UseWsOptions {
  project: string;
  runId: string | null;
  onEvent?: (ev: WsEvent) => void;
}

export function useWebSocket({ project, runId, onEvent }: UseWsOptions) {
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const connect = useCallback(() => {
    if (!runId || !project) return;
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/api/projects/${project}/events?run_id=${runId}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (msg) => {
      try {
        const ev: WsEvent = JSON.parse(msg.data);
        onEventRef.current?.(ev);
      } catch {
        /* ignore parse errors */
      }
    };
  }, [project, runId]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  return { connected };
}
