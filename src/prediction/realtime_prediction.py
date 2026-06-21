"""
SignSpeakAI - Real-Time Prediction Engine

Provides the SignPredictor class with prediction stabilization,
confidence-based filtering, and word formation logic.
"""

import os
import numpy as np
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


class PredictionStabilizer:
    """
    Stabilizes predictions by requiring N consistent predictions
    before emitting a result. Prevents flickering output.
    """

    def __init__(self, buffer_size=15, threshold=10):
        """
        Args:
            buffer_size: Number of recent predictions to keep.
            threshold: Minimum count of the same prediction in the
                       buffer before it is emitted.
        """
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.buffer = deque(maxlen=buffer_size)
        self.last_emitted = None

    def update(self, label, confidence):
        """
        Add a new prediction to the buffer.

        Args:
            label: Predicted class label.
            confidence: Prediction confidence score.

        Returns:
            The stabilized label if threshold is met and it's
            different from the last emitted label, else None.
        """
        self.buffer.append(label)

        # Count occurrences of each label in the buffer
        if len(self.buffer) < self.threshold:
            return None

        # Find the most frequent label
        label_counts = {}
        for item in self.buffer:
            label_counts[item] = label_counts.get(item, 0) + 1

        most_common = max(label_counts, key=label_counts.get)
        count = label_counts[most_common]

        if count >= self.threshold and most_common != self.last_emitted:
            self.last_emitted = most_common
            self.buffer.clear()
            return most_common

        return None

    def reset(self):
        """Reset the stabilizer state."""
        self.buffer.clear()
        self.last_emitted = None


class SignPredictor:
    """
    ASL Sign Language predictor using a trained MobileNetV2 model.
    Includes prediction stabilization and word formation.
    """

    def __init__(
        self,
        model_path="models/sign_model.keras",
        class_names_path="models/class_names.npy",
        confidence_threshold=0.60,
        stabilizer_buffer=15,
        stabilizer_threshold=10
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the trained Keras model.
            class_names_path: Path to the saved class names.
            confidence_threshold: Minimum confidence to accept a prediction.
            stabilizer_buffer: Size of the prediction stabilization buffer.
            stabilizer_threshold: Required count for stable prediction.
        """
        self.confidence_threshold = confidence_threshold
        self.img_size = (224, 224)

        # Load model
        print(f"[SignPredictor] Loading model from: {model_path}")

        # Try .keras first, fall back to .h5
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        elif os.path.exists(model_path.replace('.keras', '.h5')):
            fallback = model_path.replace('.keras', '.h5')
            print(f"[SignPredictor] .keras not found, using: {fallback}")
            self.model = tf.keras.models.load_model(fallback)
            # Check if we need to use old 64x64 size
            input_shape = self.model.input_shape
            if input_shape and input_shape[1] == 64:
                self.img_size = (64, 64)
                print(f"[SignPredictor] Using legacy input size: {self.img_size}")
        else:
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                f"Train the model first: python -m src.training.train_model"
            )

        # Load class names
        print(f"[SignPredictor] Loading classes from: {class_names_path}")
        self.class_names = np.load(class_names_path, allow_pickle=True)
        print(f"[SignPredictor] Loaded {len(self.class_names)} classes")

        # Initialize stabilizer
        self.stabilizer = PredictionStabilizer(
            buffer_size=stabilizer_buffer,
            threshold=stabilizer_threshold
        )

        # Word formation state
        self._text = ""

    def predict(self, roi):
        """
        Predict the ASL sign from a hand ROI image.

        Args:
            roi: Cropped hand region (BGR numpy array).

        Returns:
            label: Predicted class label.
            confidence: Prediction confidence (0-1).
            top5: List of (label, confidence) tuples for top 5 predictions.
        """
        import cv2

        # Resize to model input size
        img = cv2.resize(roi, self.img_size)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to float32
        img = img.astype(np.float32)

        # Apply MobileNetV2 preprocessing (or simple normalization for legacy)
        if self.img_size == (224, 224):
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        else:
            img = img / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = self.model.predict(img, verbose=0)

        # Get top prediction
        idx = np.argmax(prediction[0])
        label = str(self.class_names[idx])
        confidence = float(np.max(prediction[0]))

        # Get top 5 predictions
        top5_indices = np.argsort(prediction[0])[-5:][::-1]
        top5 = [
            (str(self.class_names[i]), float(prediction[0][i]))
            for i in top5_indices
        ]

        return label, confidence, top5

    def get_stable_prediction(self, roi):
        """
        Get a stabilized prediction. Only emits when the same
        label is predicted consistently.

        Args:
            roi: Cropped hand region (BGR numpy array).

        Returns:
            raw_label: The raw (unstabilized) predicted label.
            raw_confidence: The raw confidence score.
            stable_label: The stabilized label, or None if not yet stable.
            top5: Top 5 predictions.
        """
        label, confidence, top5 = self.predict(roi)

        stable_label = None
        if confidence >= self.confidence_threshold:
            stable_label = self.stabilizer.update(label, confidence)

        return label, confidence, stable_label, top5

    def update_text(self, label):
        """
        Update the text buffer based on the predicted label.

        Args:
            label: The stabilized predicted label.
        """
        if label is None:
            return

        if label == "space":
            self._text += " "
        elif label == "del":
            if len(self._text) > 0:
                self._text = self._text[:-1]
        elif label != "nothing":
            self._text += label

    def get_text(self):
        """Get the current accumulated text."""
        return self._text

    def clear_text(self):
        """Clear the accumulated text."""
        self._text = ""
        self.stabilizer.reset()

    def close(self):
        """Cleanup resources."""
        del self.model