# Quy trình huấn luyện MobileNetV2

Notebook chính thức: [notebooks/training/train_mobilenetv2_colab.ipynb](../notebooks/training/train_mobilenetv2_colab.ipynb).

## Dữ liệu

Notebook tải PlantVillage qua TensorFlow Datasets, không yêu cầu upload dataset thủ công. Tập dữ liệu được lọc thành đúng 15 lớp mà ứng dụng hỗ trợ. Tên lớp được chuẩn hóa về tên thư mục/class mapping mà ứng dụng sử dụng.

Nếu TFDS thay đổi nhãn hoặc không cung cấp một trong 15 lớp, notebook sẽ dừng thay vì âm thầm gán nhãn sai.

## Tiền xử lý và chia tập

- Ảnh được giải mã thành RGB và resize về 128 × 128.
- Pixel được chuẩn hóa về [0,1].
- Chia dữ liệu phân tầng, seed mặc định 42: 70% train, 15% validation, 15% test.
- Augmentation nằm trong model và chỉ hoạt động khi training.

## Mô hình

- MobileNetV2, ImageNet pretrained, include_top=False.
- Global Average Pooling.
- Dropout 0.3.
- Dense softmax với 15 đầu ra.
- Lớp Rescaling bên trong model đổi input [0,1] sang [-1,1] theo yêu cầu MobileNetV2.

## Hai giai đoạn

1. Đóng băng backbone và huấn luyện classification head với Adam learning rate 1e-3 (mặc định 12 epochs).
2. Mở 30 lớp cuối của MobileNetV2, giữ Batch Normalization đóng băng, fine-tune với Adam learning rate 1e-5 (mặc định 8 epochs).

Checkpoint tốt nhất được chọn theo validation accuracy; EarlyStopping theo validation loss. Test set chỉ được sử dụng sau khi chọn checkpoint.

## Sao lưu và xuất kết quả

Notebook lưu checkpoint theo epoch vào `MyDrive/PlantDiseaseTraining/checkpoints`, log vào `training_log.csv`, sau đó đóng gói model và báo cáo thành ZIP trong Google Drive.

Checkpoint hiện lưu model và optimizer state nhưng chưa tự động tiếp tục từ đúng epoch khi runtime bị ngắt. Không commit dataset hoặc model huấn luyện mới tự động lên GitHub. Kiểm tra model, class mapping và các báo cáo trước khi thay model đang chạy ứng dụng.

## Kiểm thử ứng dụng

Dùng [notebooks/testing/test_streamlit_colab.ipynb](../notebooks/testing/test_streamlit_colab.ipynb) để clone repository, kiểm tra model, chạy Streamlit và mở giao diện qua Cloudflare Quick Tunnel.

## Giới hạn

PlantVillage là ảnh trong điều kiện tương đối kiểm soát. Cần đánh giá độc lập trên ảnh thực địa tại địa phương; không diễn giải confidence thành xác suất chẩn đoán chắc chắn.

## Dataset đã sử dụng

Danh mục có thể kiểm tra được: [data/DATASETS.md](../data/DATASETS.md) và [data/dataset_catalog.json](../data/dataset_catalog.json).

Hiện chỉ xác nhận được PlantVillage qua TensorFlow Datasets (`plant_village`) trong pipeline hiện tại. Counts và split trong catalog là của lần chạy được lưu metadata, không phải cam kết rằng mọi lần tải dữ liệu sau sẽ có cùng counts. Cần ghi version TFDS, fingerprint, class counts và split IDs ở từng lần chạy mới.

## So sánh model trên ảnh thực địa

Dùng [notebooks/testing/compare_models_on_labeled_dataset.ipynb](../notebooks/testing/compare_models_on_labeled_dataset.ipynb) để chạy hai hoặc nhiều model trên cùng tập ảnh có nhãn thật. Không so sánh bằng ảnh train; không dùng confidence hoặc prediction agreement thay cho accuracy.
