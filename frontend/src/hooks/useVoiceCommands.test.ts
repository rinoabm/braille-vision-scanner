import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { matchCommand, speakFeedback, UNRECOGNIZED_MESSAGE } from './useVoiceCommands';

describe('matchCommand', () => {
  it('matches "start scan" to start_scan', () => {
    expect(matchCommand('start scan')).toBe('start_scan');
  });

  it('matches "scan" to start_scan', () => {
    expect(matchCommand('scan')).toBe('start_scan');
  });

  it('matches "stop scan" to stop_scan', () => {
    expect(matchCommand('stop scan')).toBe('stop_scan');
  });

  it('matches "stop" to stop_scan', () => {
    expect(matchCommand('stop')).toBe('stop_scan');
  });

  it('matches "repeat" to repeat', () => {
    expect(matchCommand('repeat')).toBe('repeat');
  });

  it('matches "faster" to faster', () => {
    expect(matchCommand('faster')).toBe('faster');
  });

  it('matches "slower" to slower', () => {
    expect(matchCommand('slower')).toBe('slower');
  });

  it('is case-insensitive', () => {
    expect(matchCommand('Start Scan')).toBe('start_scan');
    expect(matchCommand('STOP SCAN')).toBe('stop_scan');
    expect(matchCommand('Repeat')).toBe('repeat');
    expect(matchCommand('FASTER')).toBe('faster');
    expect(matchCommand('SLOWER')).toBe('slower');
  });

  it('handles leading/trailing whitespace', () => {
    expect(matchCommand('  start scan  ')).toBe('start_scan');
    expect(matchCommand('  repeat  ')).toBe('repeat');
  });

  it('matches commands embedded in longer phrases', () => {
    expect(matchCommand('please start scan now')).toBe('start_scan');
    expect(matchCommand('can you repeat that')).toBe('repeat');
  });

  it('returns null for unrecognized commands', () => {
    expect(matchCommand('hello world')).toBeNull();
    expect(matchCommand('play music')).toBeNull();
    expect(matchCommand('')).toBeNull();
  });
});

describe('speakFeedback', () => {
  let mockSpeak: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockSpeak = vi.fn();
    Object.defineProperty(window, 'speechSynthesis', {
      value: { speak: mockSpeak },
      writable: true,
      configurable: true,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('calls speechSynthesis.speak with an utterance', () => {
    speakFeedback('Hello');
    expect(mockSpeak).toHaveBeenCalledTimes(1);
    const utterance = mockSpeak.mock.calls[0][0];
    expect(utterance.text).toBe('Hello');
    expect(utterance.rate).toBe(1.0);
    expect(utterance.pitch).toBe(1.0);
  });

  it('does not throw when speechSynthesis is unavailable', () => {
    Object.defineProperty(window, 'speechSynthesis', {
      value: undefined,
      writable: true,
      configurable: true,
    });
    expect(() => speakFeedback('test')).not.toThrow();
  });
});

describe('UNRECOGNIZED_MESSAGE', () => {
  it('lists all available commands', () => {
    expect(UNRECOGNIZED_MESSAGE).toContain('start scan');
    expect(UNRECOGNIZED_MESSAGE).toContain('stop scan');
    expect(UNRECOGNIZED_MESSAGE).toContain('repeat');
    expect(UNRECOGNIZED_MESSAGE).toContain('faster');
    expect(UNRECOGNIZED_MESSAGE).toContain('slower');
    expect(UNRECOGNIZED_MESSAGE).toContain('Command not recognized');
  });
});
