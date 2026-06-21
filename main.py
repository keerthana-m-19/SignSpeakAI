"""
SignSpeakAI - Real-Time Sign Language to Speech Translator

Main application that orchestrates hand detection, sign prediction,
word formation, and text-to-speech for real-time ASL translation.

Usage:
    python main.py

Controls:
    S → Speak accumulated text
    C → Clear text
    Q → Quit
"""

import sys
import cv2
import numpy as np
import time

from src.utils.hand_detector import HandDetector
from src.prediction.realtime_prediction import SignPredictor
from src.speech.text_to_speech import TextToSpeech


# ============================================================
# UI RENDERING HELPERS
# ============================================================

class UIRenderer:
    """Renders a professional HUD overlay on the webcam frame."""

    # Color palette
    COLOR_BG = (30, 30, 30)
    COLOR_BG_SEMI = (40, 40, 40)
    COLOR_ACCENT = (255, 165, 0)       # Orange
    COLOR_GREEN = (0, 220, 100)
    COLOR_YELLOW = (0, 220, 255)
    COLOR_RED = (0, 80, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GRAY = (160, 160, 160)
    COLOR_DARK_GRAY = (80, 80, 80)
    COLOR_TEXT_BOX = (50, 50, 50)

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

    @staticmethod
    def draw_overlay(frame, overlay, x, y, w, h, alpha=0.6):
        """Draw a semi-transparent rectangle."""
        sub = frame[y:y+h, x:x+w]
        rect = np.full_like(sub, overlay)
        blended = cv2.addWeighted(sub, 1 - alpha, rect, alpha, 0)
        frame[y:y+h, x:x+w] = blended

    @staticmethod
    def get_confidence_color(confidence):
        """Get color based on confidence level."""
        if confidence >= 0.80:
            return UIRenderer.COLOR_GREEN
        elif confidence >= 0.50:
            return UIRenderer.COLOR_YELLOW
        else:
            return UIRenderer.COLOR_RED

    @staticmethod
    def draw_header(frame):
        """Draw the title bar."""
        h, w = frame.shape[:2]
        UIRenderer.draw_overlay(frame, UIRenderer.COLOR_BG, 0, 0, w, 55, alpha=0.75)

        cv2.putText(
            frame, "SignSpeakAI",
            (15, 38),
            UIRenderer.FONT_BOLD, 1.1,
            UIRenderer.COLOR_ACCENT, 2
        )

        # Subtitle
        cv2.putText(
            frame, "Real-Time ASL Translator",
            (230, 35),
            UIRenderer.FONT, 0.55,
            UIRenderer.COLOR_GRAY, 1
        )

    @staticmethod
    def draw_prediction_panel(frame, label, confidence, top5, hand_detected):
        """Draw the prediction info panel on the left side."""
        h, w = frame.shape[:2]

        # Panel background
        panel_x = 0
        panel_y = 60
        panel_w = 280
        panel_h = 220
        UIRenderer.draw_overlay(frame, UIRenderer.COLOR_BG, panel_x, panel_y,
                                panel_w, panel_h, alpha=0.7)

        if not hand_detected:
            cv2.putText(
                frame, "No Hand Detected",
                (15, panel_y + 40),
                UIRenderer.FONT, 0.65,
                UIRenderer.COLOR_GRAY, 1
            )
            cv2.putText(
                frame, "Show your hand to the camera",
                (15, panel_y + 70),
                UIRenderer.FONT, 0.45,
                UIRenderer.COLOR_DARK_GRAY, 1
            )
            return

        # Detected letter - large display
        color = UIRenderer.get_confidence_color(confidence)

        cv2.putText(
            frame, "Detected:",
            (15, panel_y + 25),
            UIRenderer.FONT, 0.5,
            UIRenderer.COLOR_GRAY, 1
        )

        cv2.putText(
            frame, str(label),
            (15, panel_y + 75),
            UIRenderer.FONT_BOLD, 1.8,
            color, 3
        )

        # Confidence percentage
        conf_text = f"{confidence * 100:.1f}%"
        cv2.putText(
            frame, conf_text,
            (100, panel_y + 75),
            UIRenderer.FONT, 0.9,
            color, 2
        )

        # Confidence bar
        bar_x = 15
        bar_y_pos = panel_y + 90
        bar_w = 250
        bar_h = 12

        cv2.rectangle(frame, (bar_x, bar_y_pos),
                       (bar_x + bar_w, bar_y_pos + bar_h),
                       UIRenderer.COLOR_DARK_GRAY, -1)
        filled_w = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y_pos),
                       (bar_x + filled_w, bar_y_pos + bar_h),
                       color, -1)

        # Top predictions
        cv2.putText(
            frame, "Top Predictions:",
            (15, panel_y + 125),
            UIRenderer.FONT, 0.45,
            UIRenderer.COLOR_GRAY, 1
        )

        if top5:
            for i, (lbl, conf) in enumerate(top5[:3]):
                y_offset = panel_y + 148 + (i * 22)
                text = f"{lbl}: {conf*100:.1f}%"
                t_color = color if i == 0 else UIRenderer.COLOR_GRAY
                cv2.putText(
                    frame, text,
                    (25, y_offset),
                    UIRenderer.FONT, 0.45,
                    t_color, 1
                )

    @staticmethod
    def draw_text_panel(frame, text):
        """Draw the text accumulation panel."""
        h, w = frame.shape[:2]

        panel_y = h - 130
        panel_h = 65
        UIRenderer.draw_overlay(frame, UIRenderer.COLOR_TEXT_BOX,
                                0, panel_y, w, panel_h, alpha=0.75)

        cv2.putText(
            frame, "Text:",
            (15, panel_y + 20),
            UIRenderer.FONT, 0.5,
            UIRenderer.COLOR_GRAY, 1
        )

        # Show text (truncate if too long for display)
        display_text = text if text else "_"
        max_chars = w // 14  # Approximate characters that fit
        if len(display_text) > max_chars:
            display_text = "..." + display_text[-(max_chars - 3):]

        cv2.putText(
            frame, display_text,
            (15, panel_y + 50),
            UIRenderer.FONT_BOLD, 0.8,
            UIRenderer.COLOR_WHITE, 2
        )

    @staticmethod
    def draw_controls(frame, is_speaking=False):
        """Draw the controls instruction bar at the bottom."""
        h, w = frame.shape[:2]

        panel_y = h - 60
        panel_h = 60
        UIRenderer.draw_overlay(frame, UIRenderer.COLOR_BG,
                                0, panel_y, w, panel_h, alpha=0.8)

        controls = [
            ("S", "Speak"),
            ("C", "Clear"),
            ("Q", "Quit")
        ]

        x_offset = 15
        for key, action in controls:
            # Key badge
            cv2.rectangle(
                frame,
                (x_offset, panel_y + 15),
                (x_offset + 30, panel_y + 43),
                UIRenderer.COLOR_ACCENT, -1
            )
            cv2.putText(
                frame, key,
                (x_offset + 7, panel_y + 37),
                UIRenderer.FONT_BOLD, 0.6,
                UIRenderer.COLOR_BG, 2
            )

            # Action label
            cv2.putText(
                frame, action,
                (x_offset + 38, panel_y + 37),
                UIRenderer.FONT, 0.55,
                UIRenderer.COLOR_WHITE, 1
            )

            x_offset += 120

        # Speaking indicator
        if is_speaking:
            cv2.putText(
                frame, "Speaking...",
                (w - 150, panel_y + 37),
                UIRenderer.FONT, 0.6,
                UIRenderer.COLOR_GREEN, 2
            )

    @staticmethod
    def draw_fps(frame, fps):
        """Draw FPS counter."""
        h, w = frame.shape[:2]
        cv2.putText(
            frame, f"FPS: {fps:.0f}",
            (w - 120, 40),
            UIRenderer.FONT, 0.55,
            UIRenderer.COLOR_GRAY, 1
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application entry point."""

    print("\n" + "=" * 55)
    print("     SignSpeakAI - Real-Time ASL Translator")
    print("=" * 55)
    print("  Loading components...\n")

    # ----------------------------------------------------------
    # Initialize Components
    # ----------------------------------------------------------

    try:
        detector = HandDetector(
            max_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
            roi_padding=40
        )
        print("  [OK] Hand Detector (MediaPipe)")
    except Exception as e:
        print(f"  [FAIL] Hand Detector: {e}")
        sys.exit(1)

    try:
        predictor = SignPredictor(
            model_path="models/sign_model.keras",
            class_names_path="models/class_names.npy",
            confidence_threshold=0.60,
            stabilizer_buffer=15,
            stabilizer_threshold=10
        )
        print("  [OK] Sign Predictor (MobileNetV2)")
    except FileNotFoundError as e:
        print(f"  [FAIL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  [FAIL] Sign Predictor: {e}")
        sys.exit(1)

    try:
        tts = TextToSpeech(rate=150, volume=0.9)
        print("  [OK] Text-to-Speech (pyttsx3)")
    except Exception as e:
        print(f"  [WARN] Text-to-Speech: {e}")
        tts = None

    # ----------------------------------------------------------
    # Open Webcam
    # ----------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("\n  [FAIL] Cannot open webcam.")
        print("  Please check your camera connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("  [OK] Webcam")

    print("\n" + "-" * 55)
    print("  Controls:")
    print("    S → Speak accumulated text")
    print("    C → Clear text")
    print("    Q → Quit")
    print("-" * 55)
    print("  Running... Show your hand to the camera!\n")

    # ----------------------------------------------------------
    # UI Renderer
    # ----------------------------------------------------------

    ui = UIRenderer()

    # ----------------------------------------------------------
    # Main Loop
    # ----------------------------------------------------------

    fps = 0
    frame_count = 0
    fps_start_time = time.time()

    current_label = ""
    current_confidence = 0.0
    current_top5 = []
    hand_detected = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] Failed to read from webcam.")
                break

            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # --------------------------------------------------
            # Hand Detection
            # --------------------------------------------------

            annotated, roi, bbox = detector.detect(frame)

            hand_detected = roi is not None

            # --------------------------------------------------
            # Sign Prediction
            # --------------------------------------------------

            if hand_detected:
                try:
                    raw_label, raw_conf, stable_label, top5 = \
                        predictor.get_stable_prediction(roi)

                    current_label = raw_label
                    current_confidence = raw_conf
                    current_top5 = top5

                    # Update text with stable predictions
                    if stable_label is not None:
                        predictor.update_text(stable_label)
                        print(f"  >> {stable_label}  |  "
                              f"Text: \"{predictor.get_text()}\"")

                except Exception as e:
                    print(f"[Prediction Error] {e}")

            # --------------------------------------------------
            # Draw UI
            # --------------------------------------------------

            ui.draw_header(annotated)
            ui.draw_prediction_panel(
                annotated, current_label, current_confidence,
                current_top5, hand_detected
            )
            ui.draw_text_panel(annotated, predictor.get_text())
            ui.draw_controls(
                annotated,
                is_speaking=(tts.is_speaking if tts else False)
            )

            # FPS calculation
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            ui.draw_fps(annotated, fps)

            # --------------------------------------------------
            # Display
            # --------------------------------------------------

            cv2.imshow("SignSpeakAI", annotated)

            # --------------------------------------------------
            # Keyboard Controls
            # --------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') or key == ord('S'):
                # Speak accumulated text
                text = predictor.get_text()
                if text.strip() and tts:
                    print(f"\n  [Speaking] \"{text.strip()}\"\n")
                    tts.speak_async(text.strip())
                elif not text.strip():
                    print("  [Speak] No text to speak.")

            elif key == ord('c') or key == ord('C'):
                # Clear text
                predictor.clear_text()
                current_label = ""
                current_confidence = 0.0
                current_top5 = []
                print("  [Cleared] Text buffer cleared.")

            elif key == ord('q') or key == ord('Q'):
                # Quit
                print("\n  [Quit] Shutting down...")
                break

    except KeyboardInterrupt:
        print("\n  [Interrupted] Shutting down...")

    finally:
        # --------------------------------------------------
        # Cleanup
        # --------------------------------------------------
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        predictor.close()
        if tts:
            tts.close()

        print("\n" + "=" * 55)
        print("  SignSpeakAI — Session Ended")
        print("=" * 55 + "\n")


if __name__ == "__main__":
    main()