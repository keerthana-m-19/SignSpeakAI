"""
SignSpeakAI - Hand Detector Module

Uses MediaPipe Hands to detect hand landmarks, draw them on the frame,
and extract a padded ROI around the detected hand for classification.
"""

import cv2
import numpy as np
import mediapipe as mp


class HandDetector:
    """
    Detects a single hand using MediaPipe Hands and extracts the
    hand region of interest (ROI) for sign language classification.
    """

    def __init__(
        self,
        max_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        roi_padding=40
    ):
        """
        Initialize the MediaPipe Hands detector.

        Args:
            max_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for hand tracking.
            roi_padding: Padding (in pixels) around the detected hand bounding box.
        """
        self.roi_padding = roi_padding

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, frame):
        """
        Detect hand in the frame and extract ROI.

        Args:
            frame: BGR image from webcam (numpy array).

        Returns:
            annotated_frame: Frame with hand landmarks drawn.
            roi: Cropped hand region (BGR), or None if no hand detected.
            bbox: Tuple (x1, y1, x2, y2) of the hand bounding box, or None.
        """
        h, w, _ = frame.shape
        annotated_frame = frame.copy()

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Process with MediaPipe
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return annotated_frame, None, None

        # Use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks on the frame
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )

        # Calculate bounding box from landmarks
        x_coords = []
        y_coords = []

        for landmark in hand_landmarks.landmark:
            x_coords.append(int(landmark.x * w))
            y_coords.append(int(landmark.y * h))

        x_min = max(0, min(x_coords) - self.roi_padding)
        y_min = max(0, min(y_coords) - self.roi_padding)
        x_max = min(w, max(x_coords) + self.roi_padding)
        y_max = min(h, max(y_coords) + self.roi_padding)

        # Make the bounding box square for better classification
        box_w = x_max - x_min
        box_h = y_max - y_min
        box_size = max(box_w, box_h)

        # Center the square box
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        x1 = max(0, cx - box_size // 2)
        y1 = max(0, cy - box_size // 2)
        x2 = min(w, cx + box_size // 2)
        y2 = min(h, cy + box_size // 2)

        # Draw bounding box
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Extract ROI
        roi = frame[y1:y2, x1:x2]

        # Validate ROI
        if roi.size == 0:
            return annotated_frame, None, None

        return annotated_frame, roi, (x1, y1, x2, y2)

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False