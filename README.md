# Emoji Creator với Deep Learning

Dự án nhận diện cảm xúc từ khuôn mặt và hiển thị emoji tương ứng sử dụng Deep Learning.

## Mô tả

Ứng dụng sử dụng mô hình CNN (Convolutional Neural Network) để nhận diện 7 loại cảm xúc:
- Angry (Tức giận)
- Disgust (Ghê tởm)
- Fear (Sợ hãi)
- Happy (Vui vẻ)
- Neutral (Bình thường)
- Sad (Buồn)
- Surprise (Ngạc nhiên)

Ứng dụng sẽ:
1. Quay video từ webcam
2. Phát hiện khuôn mặt trong video
3. Nhận diện cảm xúc
4. Hiển thị emoji tương ứng

## Cấu trúc thư mục

```
emoji-creator-project-code/
├── train.py              # Script để train model
├── gui.py                # Giao diện ứng dụng
├── emotion_model.h5      # File weights của model (sẽ được tạo sau khi train)
├── requirements.txt      # Dependencies
├── README.md            # File này
├── archive/             # Thư mục chứa dữ liệu training
│   ├── train/          # Dữ liệu training (7 thư mục cảm xúc)
│   └── test/           # Dữ liệu test (7 thư mục cảm xúc)
└── emojis/             # Thư mục chứa emoji images
    └── emojis/
        ├── angry.png
        ├── disgusted.png
        ├── fearful.png
        ├── happy.png
        ├── neutral.png
        ├── sad.png
        └── surpriced.png
```

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Train model

Trước khi chạy ứng dụng, bạn cần train model:

```bash
cd emoji-creator-project-code
python train.py
```

Quá trình training sẽ mất một khoảng thời gian (tùy thuộc vào GPU/CPU). Model sẽ được lưu vào file `emotion_model.h5`.

**Lưu ý:** 
- Training với 50 epochs có thể mất vài giờ trên CPU
- Model đã được cải thiện với:
  - **Data Augmentation**: Xoay, dịch chuyển, zoom, lật ngang để tăng dữ liệu training
  - **BatchNormalization**: Giúp training ổn định và nhanh hơn
  - **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau để tối ưu training
  - **Architecture cải thiện**: Thêm layers và BatchNormalization để tăng độ chính xác
- Nếu muốn train nhanh hơn, có thể giảm số epochs trong file `train.py`
- Model tốt nhất sẽ được lưu vào `emotion_model_best.h5` tự động

### 3. Chạy ứng dụng

Sau khi train xong, chạy ứng dụng:

```bash
python gui.py
```

## Sử dụng

1. Mở ứng dụng bằng lệnh `python gui.py`
2. Cho phép ứng dụng truy cập webcam
3. Đưa khuôn mặt vào khung hình
4. Ứng dụng sẽ tự động nhận diện cảm xúc và hiển thị emoji tương ứng
5. Nhấn nút "Quit" để thoát

## Yêu cầu hệ thống

- Python 3.7 trở lên
- Webcam
- RAM: Tối thiểu 4GB (khuyến nghị 8GB)
- GPU: Không bắt buộc nhưng sẽ giúp training nhanh hơn

## Lưu ý

- File `logo.png` là tùy chọn. Nếu không có, ứng dụng vẫn chạy bình thường với text heading
- Đảm bảo có đủ ánh sáng khi sử dụng webcam để nhận diện tốt hơn
- Model đã được train với dữ liệu grayscale 48x48 pixels

## Tác giả

Dựa trên tutorial từ DataFlair: https://data-flair.training/blogs/create-emoji-with-deep-learning/

