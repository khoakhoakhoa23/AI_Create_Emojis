from pathlib import Path
import tkinter as tk
from tkinter import *
from collections import deque, Counter
import time

import cv2
import numpy as np
from PIL import Image, ImageTk
try:
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
except ImportError:
    from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
    from keras.models import Sequential
    from keras.optimizers import Adam

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
WEIGHTS_PATH = SCRIPT_DIR / "emotion_model.weights.h5"
# Thử load best model trước, nếu không có thì dùng model thường
BEST_WEIGHTS_PATH = SCRIPT_DIR / "emotion_model_best.weights.h5"
CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
EMOJI_DIR = BASE_DIR / "emojis" / "emojis"
IMG_SIZE = (48, 48)

# Tối ưu hóa: Prediction smoothing và frame skipping
PREDICTION_SMOOTHING_SIZE = 5  # Số lượng predictions để làm mượt
PREDICTION_INTERVAL = 3  # Chỉ predict mỗi N frames để tăng tốc
MIN_FACE_SIZE = 30  # Kích thước khuôn mặt tối thiểu
CONFIDENCE_THRESHOLD = 0.3  # Ngưỡng confidence tối thiểu

# Hàm tạo model với architecture cũ (tương thích với weights cũ)
def create_old_model():
    """Tạo model với architecture cũ (ít layers hơn)"""
    model = Sequential(
        [
            # Block 1
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
            BatchNormalization(),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 2 - Architecture cũ (ít layers hơn)
            Conv2D(128, kernel_size=(3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 3 - Architecture cũ
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
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-4, decay=1e-6),
        metrics=["accuracy"],
    )
    return model

# Hàm tạo model với architecture mới (đã cải thiện)
def create_new_model():
    """Tạo model với architecture mới (nhiều layers hơn)"""
    model = Sequential(
        [
            # Block 1
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
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-4, decay=1e-6),
        metrics=["accuracy"],
    )
    return model

# Load weights if they exist, otherwise show error
# Ưu tiên load best model nếu có
weights_to_load = BEST_WEIGHTS_PATH if BEST_WEIGHTS_PATH.exists() else WEIGHTS_PATH

if not weights_to_load.exists():
    print(f"ERROR: Model weights not found at {weights_to_load}")
    print("Please run train.py first to train the model.")
    exit(1)

# Thử load với architecture mới trước, nếu fail thì dùng architecture cũ
emotion_model = None
model_version = None

try:
    print(f"Attempting to load with NEW architecture from {weights_to_load}...")
    emotion_model = create_new_model()
    emotion_model.load_weights(weights_to_load)
    model_version = "NEW"
    print(f"✓ Successfully loaded NEW model architecture from {weights_to_load}")
except (ValueError, Exception) as e:
    print(f"✗ Failed to load with NEW architecture: {str(e)[:100]}...")
    print(f"Attempting to load with OLD architecture from {weights_to_load}...")
    try:
        emotion_model = create_old_model()
        emotion_model.load_weights(weights_to_load)
        model_version = "OLD"
        print(f"✓ Successfully loaded OLD model architecture from {weights_to_load}")
        print("⚠ WARNING: Using OLD model architecture. For better accuracy, please retrain with:")
        print("   python train.py")
    except Exception as e2:
        print(f"✗ ERROR: Failed to load model weights: {str(e2)}")
        print("Please check if the weights file is corrupted or run train.py to retrain.")
        exit(1)

print(f"Model version: {model_version}")
print("Optimizations enabled: frame skipping, prediction smoothing, histogram equalization")

cv2.ocl.setUseOpenCL(False)

emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

emoji_dist = {
    0: EMOJI_DIR / "angry.png",
    1: EMOJI_DIR / "disgusted.png",
    2: EMOJI_DIR / "fearful.png",
    3: EMOJI_DIR / "happy.png",
    4: EMOJI_DIR / "neutral.png",
    5: EMOJI_DIR / "sad.png",
    6: EMOJI_DIR / "surpriced.png",
}

face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))
cap1 = cv2.VideoCapture(0)
if not cap1.isOpened():
    print("Cannot open the camera.")

# Tối ưu hóa camera settings
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap1.set(cv2.CAP_PROP_FPS, 30)

show_text = [0]
frame_count = [0]  # Đếm frame để skip prediction
prediction_history = deque(maxlen=PREDICTION_SMOOTHING_SIZE)  # Lưu lịch sử predictions
last_prediction_time = [time.time()]  # Thời gian prediction cuối cùng


def show_vid():
    if not cap1.isOpened():
        lmain.after(500, show_vid)
        return

    flag1, frame1 = cap1.read()
    if not flag1:
        lmain.after(10, show_vid)
        return

    frame1 = cv2.resize(frame1, (600, 500))
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # Tối ưu face detection: tăng scaleFactor và điều chỉnh minNeighbors để nhanh hơn
    # minSize giúp bỏ qua các khuôn mặt quá nhỏ
    num_faces = face_detector.detectMultiScale(
        gray_frame, 
        scaleFactor=1.2,  # Giảm từ 1.3 xuống 1.2 để phát hiện tốt hơn
        minNeighbors=4,   # Giảm từ 5 xuống 4 để nhanh hơn nhưng vẫn chính xác
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),  # Bỏ qua khuôn mặt quá nhỏ
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Chỉ predict mỗi N frames để tăng tốc độ
    should_predict = (frame_count[0] % PREDICTION_INTERVAL == 0)
    frame_count[0] += 1
    
    current_time = time.time()
    # Đảm bảo không predict quá thường xuyên (tối thiểu 0.1 giây)
    if should_predict and (current_time - last_prediction_time[0]) > 0.1:
        for (x, y, w, h) in num_faces:
            # Chỉ xử lý khuôn mặt lớn nhất nếu có nhiều khuôn mặt
            if len(num_faces) > 1:
                # Sắp xếp theo diện tích và lấy khuôn mặt lớn nhất
                areas = [w * h for (_, _, w, h) in num_faces]
                largest_idx = np.argmax(areas)
                (x, y, w, h) = num_faces[largest_idx]
            
            # Vẽ rectangle đẹp hơn với màu gradient (simulated với 2 màu)
            cv2.rectangle(frame1, (x, y - 50), (x + w, y + h + 10), (233, 69, 96), 3)  # Pink/red border
            cv2.rectangle(frame1, (x + 2, y - 48), (x + w - 2, y + h + 8), (83, 52, 131), 2)  # Purple inner border
            
            roi_gray_frame = gray_frame[y : y + h, x : x + w]
            
            # Cải thiện preprocessing: thêm histogram equalization để tăng độ tương phản
            roi_gray_frame = cv2.equalizeHist(roi_gray_frame)
            
            cropped_img = cv2.resize(roi_gray_frame, IMG_SIZE)
            cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0) / 255.0
            
            # Sử dụng predict_on_batch thay vì predict để nhanh hơn
            prediction = emotion_model.predict_on_batch(cropped_img)
            maxindex = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            
            # Chỉ cập nhật nếu confidence đủ cao
            if confidence >= CONFIDENCE_THRESHOLD:
                prediction_history.append(maxindex)
                # Làm mượt: lấy mode (giá trị xuất hiện nhiều nhất) của lịch sử
                if len(prediction_history) > 0:
                    most_common = Counter(prediction_history).most_common(1)[0][0]
                    show_text[0] = most_common
                else:
                    show_text[0] = maxindex
            else:
                # Nếu confidence thấp, giữ nguyên prediction cũ
                if len(prediction_history) > 0:
                    show_text[0] = prediction_history[-1]
            
            last_prediction_time[0] = current_time
            
            # Hiển thị emotion và confidence với style đẹp hơn
            emotion_text = emotion_dict.get(show_text[0], "Unknown")
            confidence_text = f"{confidence:.1%}"
            full_text = f"{emotion_text} {confidence_text}"
            
            # Background cho text (semi-transparent effect với rectangle)
            text_size = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame1, 
                         (x, y - 35), 
                         (x + text_size[0] + 10, y - 5), 
                         (26, 26, 46), -1)  # Dark background
            cv2.rectangle(frame1, 
                         (x, y - 35), 
                         (x + text_size[0] + 10, y - 5), 
                         (233, 69, 96), 2)  # Border
            
            # Text với màu đẹp
            cv2.putText(frame1, full_text, 
                       (x + 5, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            break  # Chỉ xử lý khuôn mặt đầu tiên/lớn nhất
    else:
        # Vẫn vẽ rectangle ngay cả khi không predict
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame1, (x, y - 50), (x + w, y + h + 10), (233, 69, 96), 3)
            cv2.rectangle(frame1, (x + 2, y - 48), (x + w - 2, y + h + 8), (83, 52, 131), 2)
            if len(prediction_history) > 0:
                emotion_text = emotion_dict.get(show_text[0], "Unknown")
                text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame1, 
                             (x, y - 35), 
                             (x + text_size[0] + 10, y - 5), 
                             (26, 26, 46), -1)
                cv2.rectangle(frame1, 
                             (x, y - 35), 
                             (x + text_size[0] + 10, y - 5), 
                             (233, 69, 96), 2)
                cv2.putText(frame1, emotion_text, 
                           (x + 5, y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    pic = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_vid)


def show_vid2():
    emotion_index = show_text[0]
    emoji_path = emoji_dist.get(emotion_index)
    if emoji_path and emoji_path.exists():
        frame2 = cv2.imread(str(emoji_path))
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(frame2)
        
        # Resize emoji để vừa với container (250x250 pixels)
        emoji_size = (250, 250)
        # Sử dụng LANCZOS resampling cho chất lượng tốt nhất
        try:
            img2 = img2.resize(emoji_size, Image.Resampling.LANCZOS)
        except AttributeError:
            # Fallback cho phiên bản PIL cũ
            img2 = img2.resize(emoji_size, Image.LANCZOS)
        
        imgtk2 = ImageTk.PhotoImage(image=img2)
        lmain2.imgtk2 = imgtk2
        lmain2.configure(image=imgtk2)

    # Cập nhật emotion text với font đẹp hơn (đã được set trong main)
    emotion_name = emotion_dict.get(emotion_index, "Neutral")
    lmain3.configure(text=emotion_name)
    lmain2.after(200, show_vid2)


def on_close():
    if cap1.isOpened():
        cap1.release()
    cv2.destroyAllWindows()
    root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    
    # Màu sắc theme đẹp hơn
    BG_COLOR = "#1a1a2e"  # Dark blue-gray
    SECONDARY_BG = "#16213e"  # Slightly lighter
    ACCENT_COLOR = "#0f3460"  # Blue accent
    TEXT_COLOR = "#e94560"  # Pink/red for headings
    TEXT_SECONDARY = "#ffffff"  # White for main text
    TEXT_TERTIARY = "#a8a8a8"  # Light gray
    BORDER_COLOR = "#533483"  # Purple border
    BUTTON_BG = "#e94560"  # Pink/red button
    BUTTON_HOVER = "#c73650"  # Darker pink on hover
    
    root["bg"] = BG_COLOR
    root.title("🎭 Emoji Creator - AI Emotion Detection")
    root.geometry("1400x900+100+10")
    root.protocol("WM_DELETE_WINDOW", on_close)
    
    # Tạo frame chính với padding
    main_frame = tk.Frame(root, bg=BG_COLOR)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # Header section với gradient effect (simulated) - căn giữa
    header_frame = tk.Frame(main_frame, bg=SECONDARY_BG, relief=tk.RAISED, bd=2)
    header_frame.pack(fill=tk.X, pady=(0, 20))
    
    # Try to load logo, if not found, create beautiful text heading
    logo_path = SCRIPT_DIR / "logo.png"
    if logo_path.exists():
        img = ImageTk.PhotoImage(Image.open(str(logo_path)))
        heading = Label(header_frame, image=img, bg=SECONDARY_BG)
        heading.pack(pady=15)
    else:
        # Beautiful text heading với gradient effect (simulated) - căn giữa
        heading = Label(
            header_frame, 
            text="🎭 Emoji Creator", 
            font=("Segoe UI", 36, "bold"), 
            bg=SECONDARY_BG, 
            fg=TEXT_COLOR
        )
        heading.pack(pady=15)
    
    heading2 = Label(
        header_frame, 
        text="AI Emotion Detection System", 
        font=("Segoe UI", 18, "italic"), 
        bg=SECONDARY_BG, 
        fg=TEXT_TERTIARY
    )
    heading2.pack(pady=(0, 15))
    
    # Content area với grid layout cân đối
    content_frame = tk.Frame(main_frame, bg=BG_COLOR)
    content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Container cho left và right side - căn giữa
    main_content = tk.Frame(content_frame, bg=BG_COLOR)
    main_content.pack(expand=True, fill=tk.BOTH)
    
    # Left side - Video feed với border đẹp
    left_container = tk.Frame(main_content, bg=BG_COLOR)
    left_container.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 15))
    
    video_frame = tk.Frame(left_container, bg=BORDER_COLOR, relief=tk.RAISED, bd=3)
    video_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    
    video_label_frame = tk.Label(
        video_frame, 
        text="📹 Live Camera Feed", 
        font=("Segoe UI", 12, "bold"), 
        bg=BORDER_COLOR, 
        fg=TEXT_SECONDARY
    )
    video_label_frame.pack(pady=8)
    
    lmain = tk.Label(
        master=video_frame, 
        bg="#000000",
        relief=tk.SUNKEN,
        bd=2
    )
    lmain.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)
    
    # Right side - Emotion display
    right_container = tk.Frame(main_content, bg=BG_COLOR)
    right_container.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(15, 0))
    
    emotion_frame = tk.Frame(right_container, bg=BG_COLOR)
    emotion_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    
    # Emotion label với style đẹp - căn giữa
    emotion_label_frame = tk.Frame(emotion_frame, bg=SECONDARY_BG, relief=tk.RAISED, bd=2)
    emotion_label_frame.pack(fill=tk.X, pady=(0, 15))
    
    emotion_title = tk.Label(
        emotion_label_frame,
        text="😊 Detected Emotion",
        font=("Segoe UI", 14, "bold"),
        bg=SECONDARY_BG,
        fg=TEXT_COLOR
    )
    emotion_title.pack(pady=10)
    
    # Emotion name với style lớn và đẹp - căn giữa
    lmain3 = tk.Label(
        master=emotion_frame, 
        text="Neutral",
        font=("Segoe UI", 42, "bold"),
        bg=BG_COLOR,
        fg=TEXT_COLOR,
        relief=tk.RAISED,
        bd=3,
        padx=30,
        pady=15
    )
    lmain3.pack(pady=15)
    
    # Emoji display với border đẹp và kích thước cố định - căn giữa
    emoji_wrapper = tk.Frame(emotion_frame, bg=BG_COLOR)
    emoji_wrapper.pack(expand=True, fill=tk.BOTH, pady=10)
    
    emoji_container = tk.Frame(emoji_wrapper, bg=BORDER_COLOR, relief=tk.RAISED, bd=3)
    emoji_container.pack(expand=True)  # Căn giữa trong wrapper
    
    emoji_label = tk.Label(
        emoji_container,
        text="🎭 Emoji Preview",
        font=("Segoe UI", 12, "bold"),
        bg=BORDER_COLOR,
        fg=TEXT_SECONDARY
    )
    emoji_label.pack(pady=8)
    
    # Label với kích thước cố định để hiển thị emoji đầy đủ
    lmain2 = tk.Label(
        master=emoji_container,
        bg="#000000",
        relief=tk.SUNKEN,
        bd=2,
        width=250,
        height=250
    )
    lmain2.pack(padx=15, pady=(0, 15))
    
    # Status bar - căn giữa text
    status_frame = tk.Frame(main_frame, bg=SECONDARY_BG, relief=tk.SUNKEN, bd=1)
    status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 10))
    
    status_text = f"✓ Model loaded: {model_version} | Optimizations: Frame skipping, Smoothing, Histogram EQ"
    status_label = tk.Label(
        status_frame,
        text=status_text,
        font=("Segoe UI", 9),
        bg=SECONDARY_BG,
        fg=TEXT_TERTIARY
    )
    status_label.pack(padx=10, pady=5)
    
    # Footer với button đẹp - căn giữa
    footer_frame = tk.Frame(main_frame, bg=BG_COLOR)
    footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
    
    # Button với style đẹp hơn - căn giữa
    quit_button = tk.Button(
        footer_frame,
        text="❌ Quit Application",
        command=on_close,
        font=("Segoe UI", 16, "bold"),
        bg=BUTTON_BG,
        fg=TEXT_SECONDARY,
        activebackground=BUTTON_HOVER,
        activeforeground=TEXT_SECONDARY,
        relief=tk.RAISED,
        bd=3,
        padx=30,
        pady=10,
        cursor="hand2"
    )
    quit_button.pack(pady=10)
    
    show_vid()
    show_vid2()
    root.mainloop()
