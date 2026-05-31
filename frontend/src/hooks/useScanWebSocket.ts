import { useEffect, useRef, useCallback, useState } from 'react';

export interface ScanResult {
  text: string;
  confidence: number;
  alignment: {
    is_aligned: boolean;
    direction: string;
    coverage_percent: number;
  };
  status: string;
}

export interface UseScanWebSocketOptions {
  url?: string;
  onResult?: (result: ScanResult) => void;
  onError?: (error: string) => void;
}

export function useScanWebSocket({
  url = '/ws/scan',
  onResult,
  onError,
}: UseScanWebSocketOptions = {}) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastResult, setLastResult] = useState<ScanResult | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = url.startsWith('ws')
      ? url
      : `${protocol}//${window.location.host}${url}`;

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const result: ScanResult = JSON.parse(event.data);
        setLastResult(result);
        onResult?.(result);
      } catch {
        onError?.('Failed to parse scan result');
      }
    };

    ws.onerror = () => {
      onError?.('WebSocket connection error');
    };

    ws.onclose = () => {
      setConnected(false);
      // Attempt reconnect after 3 seconds
      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, 3000);
    };

    wsRef.current = ws;
  }, [url, onResult, onError]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setConnected(false);
    }
  }, []);

  const sendFrame = useCallback((base64Frame: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(base64Frame);
    }
  }, []);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    connected,
    lastResult,
    connect,
    disconnect,
    sendFrame,
  };
}
