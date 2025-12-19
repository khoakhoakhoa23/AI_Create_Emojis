from pathlib import Path

import cv2
import numpy as np
try:
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
except ImportError:
    from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
TRAIN_DIR = BASE_DIR / "archive" / "train"
VAL_DIR = BASE_DIR / "archive" / "test"
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 50
WEIGHTS_PATH = SCRIPT_DIR / "emotion_model.weights.h5"

if not TRAIN_DIR.exists() or not VAL_DIR.exists():
    raise FileNotFoundError(
        "Expected archive/train and archive/test directories next to the project root."
    )

# Data Augmentation cho training - giúp model học tốt hơn
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,          # Xoay ảnh ±15 độ
    width_shift_range=0.1,      # Dịch ngang 10%
    height_shift_range=0.1,     # Dịch dọc 10%
    shear_range=0.1,            # Biến dạng
    zoom_range=0.1,             # Zoom ±10%
    horizontal_flip=True,       # Lật ngang
    fill_mode='nearest'         # Điền pixel khi transform
)

# Chỉ rescale cho validation - không augment
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
)

# Model architecture được cải thiện với BatchNormalization
emotion_model = Sequential(
    [
        # Block 1
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(128, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3 - Thêm một layer để tăng capacity
        Conv2D(256, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Dense layers
        Flatten(),
        Dense(1024, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation="softmax"),
    ]
)

cv2.ocl.setUseOpenCL(False)

# Compile với learning rate cao hơn một chút
emotion_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=2e-4),  # Tăng learning rate
    metrics=["accuracy"],
)

# Tính số steps
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

print(f"\n{'='*50}")
print(f"Training Configuration:")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Validation steps: {validation_steps}")
print(f"  Total training samples: {train_generator.samples}")
print(f"  Total validation samples: {validation_generator.samples}")
print(f"{'='*50}\n")

# Callbacks để cải thiện training
callbacks = [
    # Dừng sớm nếu không cải thiện
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Lưu model tốt nhất
    ModelCheckpoint(
        filepath=str(WEIGHTS_PATH).replace('.h5', '_best.weights.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    # Giảm learning rate nếu không cải thiện
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

history = emotion_model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1,
)

# Lưu model cuối cùng
emotion_model.save_weights(WEIGHTS_PATH)

# Copy best model thành file chính nếu có
import shutil
best_model_path = SCRIPT_DIR / "emotion_model_best.weights.h5"
if best_model_path.exists():
    shutil.copy(best_model_path, WEIGHTS_PATH)
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best model copied to: {WEIGHTS_PATH}")
    print(f"Original best model: {best_model_path}")
else:
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Final model saved to: {WEIGHTS_PATH}")

print(f"Class mapping: {train_generator.class_indices}")

# In kết quả cuối cùng
if hasattr(history, 'history'):
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"{'='*50}\n")
