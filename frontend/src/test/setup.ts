import '@testing-library/jest-dom';

// Mock SpeechSynthesisUtterance for jsdom
class MockSpeechSynthesisUtterance {
  text: string;
  rate: number = 1;
  pitch: number = 1;
  volume: number = 1;
  lang: string = '';
  voice: null = null;

  constructor(text?: string) {
    this.text = text || '';
  }
}

(globalThis as unknown as { SpeechSynthesisUtterance: typeof MockSpeechSynthesisUtterance }).SpeechSynthesisUtterance = MockSpeechSynthesisUtterance;
