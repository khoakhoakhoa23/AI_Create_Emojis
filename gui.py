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
# Thử nhiều đường dẫn có thể cho emojis
EMOJI_DIR = SCRIPT_DIR / "emojis" / "emojis"
if not EMOJI_DIR.exists():
    EMOJI_DIR = BASE_DIR / "emojis" / "emojis"
if not EMOJI_DIR.exists():
    EMOJI_DIR = SCRIPT_DIR / "emojis"
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
    print(f"[OK] Successfully loaded NEW model architecture from {weights_to_load}")
except (ValueError, Exception) as e:
    print(f"[FAILED] Failed to load with NEW architecture: {str(e)[:100]}...")
    print(f"Attempting to load with OLD architecture from {weights_to_load}...")
    try:
        emotion_model = create_old_model()
        emotion_model.load_weights(weights_to_load)
        model_version = "OLD"
        print(f"[OK] Successfully loaded OLD model architecture from {weights_to_load}")
        print("[WARNING] Using OLD model architecture. For better accuracy, please retrain with:")
        print("   python train.py")
    except Exception as e2:
        print(f"[ERROR] Failed to load model weights: {str(e2)}")
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

# Kiểm tra và in thông tin về emoji paths
print(f"\n{'='*60}")
print(f"Emoji directory: {EMOJI_DIR}")
print(f"Emoji directory exists: {EMOJI_DIR.exists()}")
if EMOJI_DIR.exists():
    print(f"Files in emoji directory:")
    try:
        for file in EMOJI_DIR.iterdir():
            print(f"  - {file.name}")
    except:
        pass
print(f"\nChecking emoji files:")
for idx, path in emoji_dist.items():
    exists = path.exists()
    status = "[OK]" if exists else "[MISSING]"
    print(f"  {emotion_dict[idx]:12s}: {path.name:20s} {status}")
    if not exists:
        print(f"    Full path: {path}")
print(f"{'='*60}\n")

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
        try:
            frame2 = cv2.imread(str(emoji_path))
            if frame2 is not None:
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                img2 = Image.fromarray(frame2)
                
                # Lấy kích thước emoji đã được tính toán từ lúc khởi tạo
                try:
                    # Ưu tiên dùng kích thước đã lưu trong label
                    if hasattr(lmain2, 'emoji_size'):
                        emoji_size_pixels = lmain2.emoji_size
                    else:
                        # Nếu không có, tính toán lại từ root window
                        root.update_idletasks()
                        root_height = root.winfo_height()
                        emoji_size_pixels = max(250, int(root_height * 0.35))
                    
                    # Đảm bảo kích thước tối thiểu và tối đa
                    emoji_size_pixels = max(250, min(400, emoji_size_pixels))
                    emoji_size = (emoji_size_pixels, emoji_size_pixels)
                except Exception as e:
                    # Fallback cuối cùng: kích thước cố định lớn
                    emoji_size = (300, 300)
                
                # Resize emoji với kích thước đã tính
                try:
                    img2 = img2.resize(emoji_size, Image.Resampling.LANCZOS)
                except AttributeError:
                    # Fallback cho phiên bản PIL cũ
                    img2 = img2.resize(emoji_size, Image.LANCZOS)
                
                # Tạo PhotoImage và hiển thị
                imgtk2 = ImageTk.PhotoImage(image=img2)
                lmain2.imgtk2 = imgtk2  # Giữ reference để tránh garbage collection
                lmain2.configure(image=imgtk2)  # Không set width/height, PhotoImage tự có kích thước
            else:
                print(f"Warning: Could not load emoji image from {emoji_path}")
        except Exception as e:
            print(f"Error loading emoji: {e}")
            import traceback
            traceback.print_exc()
    else:
        if emoji_path:
            print(f"Warning: Emoji file not found: {emoji_path}")

    # Cập nhật emotion text với font đẹp hơn (đã được set trong main)
    emotion_name = emotion_dict.get(emotion_index, "Neutral")
    lmain3.configure(text=emotion_name)
    lmain2.after(200, show_vid2)


def on_close():
    if cap1.isOpened():
        cap1.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass  # Ignore OpenCV window errors on Windows
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
    root.title("Emoji Creator - AI Emotion Detection")
    
    # Tính toán kích thước responsive dựa trên màn hình
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Sử dụng 90% màn hình, tối thiểu 800x600
    window_width = max(800, int(screen_width * 0.9))
    window_height = max(600, int(screen_height * 0.85))
    
    # Căn giữa cửa sổ
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    root.minsize(800, 600)  # Kích thước tối thiểu
    root.protocol("WM_DELETE_WINDOW", on_close)
    
    # Tạo frame chính với padding responsive
    main_frame = tk.Frame(root, bg=BG_COLOR)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Header section với gradient effect (simulated) - căn giữa
    header_frame = tk.Frame(main_frame, bg=SECONDARY_BG, relief=tk.RAISED, bd=2)
    header_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Tính toán font size dựa trên kích thước màn hình
    base_font_size = max(20, int(window_width / 40))
    sub_font_size = max(12, int(window_width / 80))
    
    # Try to load logo, if not found, create beautiful text heading
    logo_path = SCRIPT_DIR / "logo.png"
    if logo_path.exists():
        try:
            img = ImageTk.PhotoImage(Image.open(str(logo_path)))
            heading = Label(header_frame, image=img, bg=SECONDARY_BG)
            heading.pack(pady=8)
        except:
            heading = Label(
                header_frame, 
                text="Emoji Creator", 
                font=("Segoe UI", base_font_size, "bold"), 
                bg=SECONDARY_BG, 
                fg=TEXT_COLOR
            )
            heading.pack(pady=8)
    else:
        # Beautiful text heading với gradient effect (simulated) - căn giữa
        heading = Label(
            header_frame, 
            text="Emoji Creator", 
            font=("Segoe UI", base_font_size, "bold"), 
            bg=SECONDARY_BG, 
            fg=TEXT_COLOR
        )
        heading.pack(pady=8)
    
    heading2 = Label(
        header_frame, 
        text="AI Emotion Detection System", 
        font=("Segoe UI", sub_font_size, "italic"), 
        bg=SECONDARY_BG, 
        fg=TEXT_TERTIARY
    )
    heading2.pack(pady=(0, 8))
    
    # Content area với grid layout cân đối - responsive
    content_frame = tk.Frame(main_frame, bg=BG_COLOR)
    content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    # Sử dụng grid để chia đều không gian
    content_frame.columnconfigure(0, weight=1, uniform="equal")
    content_frame.columnconfigure(1, weight=1, uniform="equal")
    content_frame.rowconfigure(0, weight=1)
    
    # Left side - Video feed với border đẹp
    left_container = tk.Frame(content_frame, bg=BG_COLOR)
    left_container.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
    left_container.columnconfigure(0, weight=1)
    left_container.rowconfigure(1, weight=1)
    
    video_frame = tk.Frame(left_container, bg=BORDER_COLOR, relief=tk.RAISED, bd=2)
    video_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    video_frame.columnconfigure(0, weight=1)
    video_frame.rowconfigure(1, weight=1)
    
    video_label_frame = tk.Label(
        video_frame, 
        text="Live Camera Feed", 
        font=("Segoe UI", max(10, int(window_width / 120)), "bold"), 
        bg=BORDER_COLOR, 
        fg=TEXT_SECONDARY
    )
    video_label_frame.grid(row=0, column=0, pady=5)
    
    lmain = tk.Label(
        master=video_frame, 
        bg="#000000",
        relief=tk.SUNKEN,
        bd=2
    )
    lmain.grid(row=1, column=0, padx=5, pady=(0, 5), sticky="nsew")
    
    # Right side - Emotion display
    right_container = tk.Frame(content_frame, bg=BG_COLOR)
    right_container.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
    right_container.columnconfigure(0, weight=1)
    right_container.rowconfigure(1, weight=1)
    
    emotion_frame = tk.Frame(right_container, bg=BG_COLOR)
    emotion_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    emotion_frame.columnconfigure(0, weight=1)
    
    # Emotion label với style đẹp - căn giữa
    emotion_label_frame = tk.Frame(emotion_frame, bg=SECONDARY_BG, relief=tk.RAISED, bd=2)
    emotion_label_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    
    emotion_title_font_size = max(10, int(window_width / 100))
    emotion_title = tk.Label(
        emotion_label_frame,
        text="Detected Emotion",
        font=("Segoe UI", emotion_title_font_size, "bold"),
        bg=SECONDARY_BG,
        fg=TEXT_COLOR
    )
    emotion_title.pack(pady=5)
    
    # Emotion name với style lớn và đẹp - căn giữa, responsive
    emotion_name_font_size = max(24, int(window_width / 35))
    lmain3 = tk.Label(
        master=emotion_frame, 
        text="Neutral",
        font=("Segoe UI", emotion_name_font_size, "bold"),
        bg=BG_COLOR,
        fg=TEXT_COLOR,
        relief=tk.RAISED,
        bd=2,
        padx=15,
        pady=8
    )
    lmain3.grid(row=1, column=0, pady=10, sticky="ew")
    
    # Emoji display với border đẹp và responsive - căn giữa
    emoji_wrapper = tk.Frame(emotion_frame, bg=BG_COLOR)
    emoji_wrapper.grid(row=2, column=0, sticky="nsew", pady=10)
    emoji_wrapper.columnconfigure(0, weight=1)
    emoji_wrapper.rowconfigure(1, weight=1)
    
    emoji_container = tk.Frame(emoji_wrapper, bg=BORDER_COLOR, relief=tk.RAISED, bd=2)
    emoji_container.grid(row=1, column=0, sticky="")
    
    emoji_label_font_size = max(9, int(window_width / 120))
    emoji_label = tk.Label(
        emoji_container,
        text="Emoji Preview",
        font=("Segoe UI", emoji_label_font_size, "bold"),
        bg=BORDER_COLOR,
        fg=TEXT_SECONDARY
    )
    emoji_label.pack(pady=5)
    
    # Label với kích thước responsive để hiển thị emoji đầy đủ
    # Tính toán kích thước emoji dựa trên màn hình (khoảng 30-35% chiều cao, tối thiểu 250px)
    emoji_size_pixels = max(250, min(400, int(window_height * 0.35)))
    
    # Tạo label với kích thước cố định để đảm bảo emoji hiển thị đúng
    # Sử dụng pixel trực tiếp thông qua image, không dùng width/height của label
    lmain2 = tk.Label(
        master=emoji_container,
        bg="#000000",
        relief=tk.SUNKEN,
        bd=2,
        text="Loading emoji...",  # Placeholder text
        fg="white"
    )
    lmain2.pack(padx=15, pady=(0, 15))
    # Lưu kích thước emoji để dùng trong show_vid2
    lmain2.emoji_size = emoji_size_pixels
    
    # Load emoji ngay từ đầu để hiển thị
    try:
        initial_emoji_path = emoji_dist.get(4)  # Neutral emoji
        if initial_emoji_path and initial_emoji_path.exists():
            initial_frame = cv2.imread(str(initial_emoji_path))
            if initial_frame is not None:
                initial_frame = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB)
                initial_img = Image.fromarray(initial_frame)
                initial_img = initial_img.resize((emoji_size_pixels, emoji_size_pixels), Image.Resampling.LANCZOS)
                initial_imgtk = ImageTk.PhotoImage(image=initial_img)
                lmain2.imgtk2 = initial_imgtk
                lmain2.configure(image=initial_imgtk, text="")
    except Exception as e:
        print(f"Could not load initial emoji: {e}")
    
    # Footer với button đẹp - căn giữa
    footer_frame = tk.Frame(main_frame, bg=BG_COLOR)
    footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
    
    # Button với style đẹp hơn - căn giữa, responsive
    button_font_size = max(12, int(window_width / 90))
    quit_button = tk.Button(
        footer_frame,
        text="Quit Application",
        command=on_close,
        font=("Segoe UI", button_font_size, "bold"),
        bg=BUTTON_BG,
        fg=TEXT_SECONDARY,
        activebackground=BUTTON_HOVER,
        activeforeground=TEXT_SECONDARY,
        relief=tk.RAISED,
        bd=2,
        padx=20,
        pady=8,
        cursor="hand2"
    )
    quit_button.pack(pady=5)
    
    # Status bar - căn giữa text
    status_frame = tk.Frame(main_frame, bg=SECONDARY_BG, relief=tk.SUNKEN, bd=1)
    status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 5))
    
    status_text = f"Model loaded: {model_version} | Optimizations: Frame skipping, Smoothing, Histogram EQ"
    status_font_size = max(8, int(window_width / 150))
    status_label = tk.Label(
        status_frame,
        text=status_text,
        font=("Segoe UI", status_font_size),
        bg=SECONDARY_BG,
        fg=TEXT_TERTIARY
    )
    status_label.pack(padx=5, pady=3)
    
    show_vid()
    show_vid2()
    root.mainloop()
