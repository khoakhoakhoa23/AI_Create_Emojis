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

# Data Augmentation cho training - giúp model học tốt hơn và chính xác hơn
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,          # Tăng lên ±20 độ để học tốt hơn với góc nghiêng
    width_shift_range=0.15,     # Tăng lên 15% để học tốt hơn với vị trí
    height_shift_range=0.15,    # Tăng lên 15%
    shear_range=0.15,           # Tăng lên 15% để học tốt hơn với biến dạng
    zoom_range=0.15,            # Tăng lên 15% để học tốt hơn với khoảng cách
    horizontal_flip=True,       # Lật ngang
    brightness_range=[0.8, 1.2], # Thay đổi độ sáng để học tốt hơn với ánh sáng khác nhau
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

# Model architecture được cải thiện với BatchNormalization và tối ưu hơn
emotion_model = Sequential(
    [
        # Block 1 - Tăng số filters để học tốt hơn
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2 - Thêm một Conv layer để tăng capacity
        Conv2D(128, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3 - Tăng filters và thêm layer
        Conv2D(256, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        Conv2D(256, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Dense layers - Tối ưu số lượng neurons
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

# Compile với learning rate được tối ưu
emotion_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999),  # Learning rate cao hơn cho training ban đầu
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

# Callbacks để cải thiện training - tối ưu hơn
callbacks = [
    # Dừng sớm nếu không cải thiện - tăng patience để học tốt hơn
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,  # Tăng từ 10 lên 15 để cho model học lâu hơn
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    # Lưu model tốt nhất
    ModelCheckpoint(
        filepath=str(WEIGHTS_PATH).replace('.h5', '_best.weights.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
        mode='max'
    ),
    # Giảm learning rate nếu không cải thiện - điều chỉnh để tối ưu hơn
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # Giảm mạnh hơn (từ 0.5 xuống 0.3) để học tốt hơn
        patience=4,  # Giảm từ 5 xuống 4 để phản ứng nhanh hơn
        min_lr=1e-6,  # Tăng min_lr từ 1e-7 lên 1e-6
        verbose=1,
        mode='min'
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
