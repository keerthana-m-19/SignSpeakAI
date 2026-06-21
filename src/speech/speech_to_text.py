"""
SignSpeakAI - Speech-to-Text Module

Provides speech recognition functionality using the
SpeechRecognition library with Google's free API.
This enables bidirectional communication — hearing users
can speak back to the sign language user.
"""

import threading

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("[Warning] speech_recognition not installed. "
          "Run: pip install SpeechRecognition")


class SpeechToText:
    """
    Speech-to-Text engine using Google's free speech recognition.
    Listens to microphone input and converts speech to text.
    """

    def __init__(self, energy_threshold=300, pause_threshold=1.0):
        """
        Initialize the STT engine.

        Args:
            energy_threshold: Minimum audio energy to consider for recording.
            pause_threshold: Seconds of non-speaking audio before a phrase
                             is considered complete.
        """
        if not SR_AVAILABLE:
            self.available = False
            return

        self.available = True
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self._listening = False
        self._last_text = ""
        self._lock = threading.Lock()

    def listen(self):
        """
        Listen to microphone and return recognized text (blocking).

        Returns:
            Recognized text string, or empty string on failure.
        """
        if not self.available:
            print("[STT] speech_recognition library not available.")
            return ""

        try:
            with sr.Microphone() as source:
                print("[STT] Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("[STT] Listening...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

            print("[STT] Recognizing...")
            text = self.recognizer.recognize_google(audio)
            self._last_text = text
            return text

        except sr.WaitTimeoutError:
            print("[STT] No speech detected (timeout).")
            return ""
        except sr.UnknownValueError:
            print("[STT] Could not understand audio.")
            return ""
        except sr.RequestError as e:
            print(f"[STT] API error: {e}")
            return ""
        except Exception as e:
            print(f"[STT] Error: {e}")
            return ""

    def listen_async(self, callback=None):
        """
        Listen to microphone in a background thread (non-blocking).

        Args:
            callback: Optional function to call with recognized text.
                      Signature: callback(text: str) -> None
        """
        if not self.available:
            return

        if self._listening:
            return

        thread = threading.Thread(
            target=self._listen_worker,
            args=(callback,),
            daemon=True
        )
        thread.start()

    def _listen_worker(self, callback=None):
        """Background worker for async listening."""
        with self._lock:
            self._listening = True
            try:
                text = self.listen()
                if callback and text:
                    callback(text)
            finally:
                self._listening = False

    @property
    def is_listening(self):
        """Check if currently listening."""
        return self._listening

    @property
    def last_text(self):
        """Get the last recognized text."""
        return self._last_text
