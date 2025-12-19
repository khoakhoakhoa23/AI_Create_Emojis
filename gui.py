from pathlib import Path
import tkinter as tk
from tkinter import *

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
CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
EMOJI_DIR = BASE_DIR / "emojis" / "emojis"
IMG_SIZE = (48, 48)

# Model architecture phải giống hệt với train.py
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
        
        # Block 3
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
emotion_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-4, decay=1e-6),
    metrics=["accuracy"],
)

# Load weights if they exist, otherwise show error
if not WEIGHTS_PATH.exists():
    print(f"ERROR: Model weights not found at {WEIGHTS_PATH}")
    print("Please run train.py first to train the model.")
    exit(1)

emotion_model.load_weights(WEIGHTS_PATH)
print(f"Loaded model weights from {WEIGHTS_PATH}")

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

show_text = [0]


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
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y : y + h, x : x + w]
        cropped_img = cv2.resize(roi_gray_frame, IMG_SIZE)
        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0) / 255.0
        prediction = emotion_model.predict(cropped_img, verbose=0)
        maxindex = int(np.argmax(prediction))
        show_text[0] = maxindex

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
        imgtk2 = ImageTk.PhotoImage(image=img2)
        lmain2.imgtk2 = imgtk2
        lmain2.configure(image=imgtk2)

    lmain3.configure(text=emotion_dict.get(emotion_index, ""), font=("arial", 45, "bold"))
    lmain2.after(200, show_vid2)


def on_close():
    if cap1.isOpened():
        cap1.release()
    cv2.destroyAllWindows()
    root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    
    # Try to load logo, if not found, skip it
    logo_path = SCRIPT_DIR / "logo.png"
    if logo_path.exists():
        img = ImageTk.PhotoImage(Image.open(str(logo_path)))
        heading = Label(root, image=img, bg="black")
        heading.pack()
    else:
        # Create a simple text heading if logo not found
        heading = Label(root, text="Emoji Creator", pady=10, font=("arial", 30, "bold"), bg="black", fg="#CDCDCD")
        heading.pack()
    heading2 = Label(root, text="Photo to Emoji", pady=20, font=("arial", 45, "bold"), bg="black", fg="#CDCDCD")

    heading2.pack()
    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)

    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg="black")
    lmain.pack(side=LEFT)
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900, y=350)

    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root["bg"] = "black"
    root.protocol("WM_DELETE_WINDOW", on_close)
    Button(root, text="Quit", fg="red", command=on_close, font=("arial", 25, "bold")).pack(side=BOTTOM)
    show_vid()
    show_vid2()
    root.mainloop()
