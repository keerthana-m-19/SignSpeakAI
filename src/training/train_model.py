"""
SignSpeakAI - Model Training Pipeline

Trains a MobileNetV2-based model for ASL alphabet classification
using transfer learning with a two-phase approach:
  Phase 1: Feature extraction (frozen base)
  Phase 2: Fine-tuning (top layers unfrozen)

Usage:
    python -m src.training.train_model
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info messages

import tensorflow as tf
from tensorflow.keras import layers, callbacks

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
FINE_TUNE_EPOCHS = 3
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-4
FINE_TUNE_LAYERS = 30  # Number of top base layers to unfreeze

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "train")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "sign_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.npy")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")
# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("  SignSpeakAI - Model Training Pipeline")
print("=" * 60)
print(f"\n  Image Size:    {IMG_SIZE}")
print(f"  Batch Size:    {BATCH_SIZE}")
print(f"  Phase 1:       {EPOCHS} epochs (feature extraction)")
print(f"  Phase 2:       {FINE_TUNE_EPOCHS} epochs (fine-tuning)")
print(f"  Dataset:       {DATASET_DIR}")
print("=" * 60 + "\n")

print("[1/6] Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

class_names = train_ds.class_names
num_classes = len(class_names)

print(f"\n  Classes ({num_classes}):")
print(f"  {class_names}\n")

# Save class names
os.makedirs("models", exist_ok=True)
np.save(CLASS_NAMES_PATH, class_names)
print(f"  Class names saved to: {CLASS_NAMES_PATH}")

# ============================================================
# OPTIMIZE DATASET PIPELINE
# ============================================================

print("\n[2/6] Optimizing data pipeline...")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(500).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================

print("[3/6] Building data augmentation layers...")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
], name="data_augmentation")

# ============================================================
# BUILD MODEL
# ============================================================

print("[4/6] Building MobileNetV2 model...")

# Load MobileNetV2 base (pretrained on ImageNet)
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze all base model layers for Phase 1
base_model.trainable = False

print(f"  Base model layers: {len(base_model.layers)}")
print(f"  Base model trainable: {base_model.trainable}")

# Build the full model
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Augmentation (only during training)
x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# Feature extraction
x = base_model(x, training=False)

# Classification head
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs, name="SignSpeakAI_MobileNetV2")

model.summary()

# ============================================================
# PHASE 1: FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 60)
print("  Phase 1: Feature Extraction (base frozen)")
print("=" * 60 + "\n")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
training_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=training_callbacks
)

# ============================================================
# PHASE 2: FINE-TUNING
# ============================================================

print("\n" + "=" * 60)
print("  Phase 2: Fine-Tuning (top layers unfrozen)")
print("=" * 60 + "\n")

# Unfreeze the top layers of the base model
base_model.trainable = True

# Freeze all layers except the top FINE_TUNE_LAYERS
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"  Total base layers: {len(base_model.layers)}")
print(f"  Trainable layers:  {trainable_count}")

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning callbacks
fine_tune_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Continue training from where Phase 1 left off
total_epochs = EPOCHS + FINE_TUNE_EPOCHS

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history1.epoch[-1] + 1,
    callbacks=fine_tune_callbacks
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n[5/6] Saving final model...")

model.save(MODEL_SAVE_PATH)
print(f"  Model saved to: {MODEL_SAVE_PATH}")

# ============================================================
# PLOT TRAINING HISTORY
# ============================================================

print("[6/6] Generating training history plot...")


def plot_training_history(h1, h2, save_path):
    """Plot training and validation accuracy/loss curves."""
    # Combine histories
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    epochs_range = range(1, len(acc) + 1)
    phase1_end = len(h1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7,
                label='Fine-tuning Start')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Loss plot
    ax2.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax2.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7,
                label='Fine-tuning Start')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('SignSpeakAI - Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Training plot saved to: {save_path}")


plot_training_history(history1, history2, HISTORY_PLOT_PATH)

# ============================================================
# FINAL RESULTS
# ============================================================

final_val_loss, final_val_acc = model.evaluate(val_ds, verbose=0)

print("\n" + "=" * 60)
print("  Training Complete!")
print("=" * 60)
print(f"  Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"  Final Validation Loss:     {final_val_loss:.4f}")
print(f"  Model saved to:            {MODEL_SAVE_PATH}")
print(f"  Class names saved to:      {CLASS_NAMES_PATH}")
print(f"  Training plot saved to:    {HISTORY_PLOT_PATH}")
print("=" * 60 + "\n")
print("=" * 60 + "\n")
