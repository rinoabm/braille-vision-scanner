import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useScanWebSocket } from './useScanWebSocket';

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  url: string;
  readyState: number = 0; // CONNECTING
  onopen: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;
  sentMessages: string[] = [];

  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sentMessages.push(data);
  }

  close() {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }

  simulateOpen() {
    this.readyState = MockWebSocket.OPEN;
    if (this.onopen) {
      this.onopen(new Event('open'));
    }
  }

  simulateMessage(data: string) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data }));
    }
  }

  simulateError() {
    if (this.onerror) {
      this.onerror(new Event('error'));
    }
  }
}

describe('useScanWebSocket', () => {
  const originalWebSocket = globalThis.WebSocket;

  beforeEach(() => {
    MockWebSocket.instances = [];
    // Replace global WebSocket with mock
    (globalThis as any).WebSocket = MockWebSocket;
    vi.useFakeTimers();
  });

  afterEach(() => {
    (globalThis as any).WebSocket = originalWebSocket;
    vi.useRealTimers();
  });

  it('starts disconnected', () => {
    const { result } = renderHook(() => useScanWebSocket());
    expect(result.current.connected).toBe(false);
    expect(result.current.lastResult).toBeNull();
  });

  it('connects when connect is called', () => {
    const { result } = renderHook(() => useScanWebSocket({ url: 'ws://localhost:8000/ws/scan' }));

    act(() => {
      result.current.connect();
    });

    expect(MockWebSocket.instances.length).toBe(1);
    expect(MockWebSocket.instances[0].url).toBe('ws://localhost:8000/ws/scan');
  });

  it('sets connected to true on open', () => {
    const { result } = renderHook(() => useScanWebSocket({ url: 'ws://localhost:8000/ws/scan' }));

    act(() => {
      result.current.connect();
    });

    act(() => {
      MockWebSocket.instances[0].simulateOpen();
    });

    expect(result.current.connected).toBe(true);
  });

  it('parses scan results from messages', () => {
    const onResult = vi.fn();
    const { result } = renderHook(() =>
      useScanWebSocket({ url: 'ws://localhost:8000/ws/scan', onResult })
    );

    act(() => {
      result.current.connect();
    });

    act(() => {
      MockWebSocket.instances[0].simulateOpen();
    });

    const scanResult = {
      text: 'hello',
      confidence: 0.85,
      alignment: { is_aligned: true, direction: 'centered', coverage_percent: 75 },
      status: 'scanning',
    };

    act(() => {
      MockWebSocket.instances[0].simulateMessage(JSON.stringify(scanResult));
    });

    expect(result.current.lastResult).toEqual(scanResult);
    expect(onResult).toHaveBeenCalledWith(scanResult);
  });

  it('calls onError for invalid JSON', () => {
    const onError = vi.fn();
    const { result } = renderHook(() =>
      useScanWebSocket({ url: 'ws://localhost:8000/ws/scan', onError })
    );

    act(() => {
      result.current.connect();
    });

    act(() => {
      MockWebSocket.instances[0].simulateOpen();
    });

    act(() => {
      MockWebSocket.instances[0].simulateMessage('not json');
    });

    expect(onError).toHaveBeenCalledWith('Failed to parse scan result');
  });

  it('sends base64 frames when connected', () => {
    const { result } = renderHook(() => useScanWebSocket({ url: 'ws://localhost:8000/ws/scan' }));

    act(() => {
      result.current.connect();
    });

    act(() => {
      MockWebSocket.instances[0].simulateOpen();
    });

    act(() => {
      result.current.sendFrame('base64data');
    });

    expect(MockWebSocket.instances[0].sentMessages).toContain('base64data');
  });

  it('disconnects cleanly', () => {
    const { result } = renderHook(() => useScanWebSocket({ url: 'ws://localhost:8000/ws/scan' }));

    act(() => {
      result.current.connect();
    });

    act(() => {
      MockWebSocket.instances[0].simulateOpen();
    });

    act(() => {
      result.current.disconnect();
    });

    expect(result.current.connected).toBe(false);
  });
});
