"""
SignSpeakAI - Text-to-Speech Module

Provides text-to-speech functionality using pyttsx3 with
configurable voice properties and non-blocking async speech.
"""

import threading
import pyttsx3


class TextToSpeech:
    """
    Text-to-Speech engine with configurable rate, volume,
    and support for non-blocking asynchronous speech.
    """

    def __init__(self, rate=150, volume=0.9):
        """
        Initialize the TTS engine.

        Args:
            rate: Speech rate in words per minute (default 150).
            volume: Volume level from 0.0 to 1.0 (default 0.9).
        """
        self.rate = rate
        self.volume = volume
        self._speaking = False
        self._lock = threading.Lock()

        # Initialize engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', self.rate)
        self.engine.setProperty('volume', self.volume)

        # Try to set a clear voice
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            # Prefer a female voice for clarity (usually index 1)
            self.engine.setProperty('voice', voices[1].id)

    def speak(self, text):
        """
        Speak the given text (blocking).

        Args:
            text: The text to speak.
        """
        if not text or not text.strip():
            return

        self.engine.say(text.strip())
        self.engine.runAndWait()

    def speak_async(self, text):
        """
        Speak the given text in a background thread (non-blocking).

        Args:
            text: The text to speak.
        """
        if not text or not text.strip():
            return

        if self._speaking:
            return  # Don't overlap speech

        thread = threading.Thread(
            target=self._speak_worker,
            args=(text.strip(),),
            daemon=True
        )
        thread.start()

    def _speak_worker(self, text):
        """Background worker for async speech."""
        with self._lock:
            self._speaking = True
            try:
                # Create a new engine instance for thread safety
                engine = pyttsx3.init()
                engine.setProperty('rate', self.rate)
                engine.setProperty('volume', self.volume)

                voices = engine.getProperty('voices')
                if len(voices) > 1:
                    engine.setProperty('voice', voices[1].id)

                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[TTS Error] {e}")
            finally:
                self._speaking = False

    @property
    def is_speaking(self):
        """Check if the engine is currently speaking."""
        return self._speaking

    def close(self):
        """Cleanup TTS engine resources."""
        try:
            self.engine.stop()
        except Exception:
            pass