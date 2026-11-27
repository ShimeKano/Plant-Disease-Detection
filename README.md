# 🌱 Plant Disease AI

Ứng dụng AI nhận diện bệnh và tình trạng lá cây trồng từ hình ảnh, xây dựng bằng **MobileNetV2** và **Streamlit**. Phiên bản hiện tại tập trung vào 15 lớp của cà chua, khoai tây và ớt chuông từ bộ dữ liệu PlantVillage.

> Đây là sản phẩm nghiên cứu/đồ án và công cụ tham khảo. Kết quả mô hình không thay thế chẩn đoán của cán bộ kỹ thuật nông nghiệp.

## Tính năng

- Tải ảnh lá cây lên giao diện web.
- Dự đoán 15 lớp bệnh/tình trạng.
- Hiển thị lớp dự đoán, confidence và Top-3.
- Hiển thị triệu chứng thường gặp và hướng xử lý tham khảo bằng tiếng Việt.
- Hiển thị lưu ý về thuốc bảo vệ thực vật, nhãn thuốc và thời gian cách ly.
- Chạy giao diện bằng Streamlit trên máy cá nhân hoặc Google Colab.

## Phạm vi nhận diện

| Cây trồng | Các lớp |
|---|---|
| Ớt chuông | Đốm vi khuẩn, khỏe mạnh |
| Khoai tây | Cháy lá sớm, mốc sương, khỏe mạnh |
| Cà chua | Đốm vi khuẩn, cháy lá sớm, mốc sương, mốc lá, đốm lá Septoria, nhện đỏ hai chấm, Target Spot, virus xoăn vàng lá, virus khảm, khỏe mạnh |

Danh sách nhãn kỹ thuật được lưu trong `model/class_names.json` và phải giữ đúng thứ tự đầu ra của model.

## Mô hình và huấn luyện

Pipeline huấn luyện chính thức được triển khai trong notebook Colab:

- Kiến trúc: MobileNetV2, pretrained trên ImageNet.
- Kích thước đầu vào: 128 × 128 × 3.
- Số lớp đầu ra: 15.
- Global Average Pooling, Dropout 0.3 và Dense softmax.
- Augmentation chỉ áp dụng trong giai đoạn training.
- Chia tập phân tầng: 70% train, 15% validation, 15% test.
- Hai giai đoạn: huấn luyện classification head, sau đó fine-tune 30 lớp cuối; Batch Normalization được giữ đóng băng.
- Seed mặc định: 42.
- Chọn model theo validation accuracy; đánh giá cuối cùng trên test set độc lập.

### Huấn luyện trên Google Colab

[**Mở notebook huấn luyện MobileNetV2**](https://colab.research.google.com/github/ShimeKano/Plant-Disease-Detection/blob/main/notebooks/training/train_mobilenetv2_colab.ipynb)

Notebook tự tải PlantVillage thông qua TensorFlow Datasets, lọc đúng 15 lớp, tạo dataset, huấn luyện, đánh giá và đóng gói model cùng báo cáo vào Google Drive. Không cần upload dataset thủ công.

Kết quả gồm `best_model.keras`, `class_names.json`, `training_metadata.json`, `training_log.csv`, báo cáo phân loại, confusion matrix, learning curves và workflow diagram.

Checkpoint theo epoch được lưu riêng lên Google Drive. Notebook hiện lưu checkpoint nhưng chưa tự động resume đúng epoch sau khi runtime bị ngắt. Model hiện có trong repository không bị notebook ghi đè và không được tự động đẩy lên GitHub.

**Chưa công bố chỉ số hiệu năng của lần huấn luyện mới cho đến khi notebook được chạy hoàn tất và kết quả được kiểm tra.** Các tệp trong `reports/archived-training-run/` là kết quả lịch sử được giữ lại để tham khảo, không mặc định là kết quả của lần huấn luyện mới nhất.

Chi tiết kỹ thuật: [docs/TRAINING.md](docs/TRAINING.md).

## Kiểm thử ứng dụng trên Google Colab

[**Mở notebook kiểm thử Streamlit**](https://colab.research.google.com/github/ShimeKano/Plant-Disease-Detection/blob/main/notebooks/testing/test_streamlit_colab.ipynb)

Notebook sẽ clone repository, cài dependencies, kiểm tra model, khởi chạy Streamlit ở cổng 8501, tạo Cloudflare Quick Tunnel để lấy URL mở ứng dụng, và cho phép tải ảnh lên Colab để kiểm thử dự đoán trực tiếp.

Cloudflare Quick Tunnel tạo URL công khai tạm thời; không dùng để triển khai production hoặc tải lên dữ liệu nhạy cảm.

## Chạy trên máy cá nhân

Yêu cầu Python tương thích với TensorFlow và môi trường có đủ RAM.

```bash
git clone https://github.com/ShimeKano/Plant-Disease-Detection.git
cd Plant-Disease-Detection
python -m venv .venv
```

Kích hoạt môi trường ảo, sau đó cài thư viện:

```bash
# Windows
.venv\\Scripts\\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Mở URL localhost mà Streamlit hiển thị trong terminal.

## Cấu trúc repository

```text
Plant-Disease-Detection/
├── app.py
├── requirements.txt
├── README.md
├── model/
│   ├── best_model.keras
│   ├── best_model.h5
│   └── class_names.json
├── src/
│   └── train.py
├── docs/
│   └── TRAINING.md
├── notebooks/
│   ├── training/
│   │   └── train_mobilenetv2_colab.ipynb
│   └── testing/
│       └── test_streamlit_colab.ipynb
└── reports/
    └── archived-training-run/
        └── previous metrics and plots
```

## Giới hạn và lưu ý khoa học

- Model chỉ phân loại trong 15 lớp đã huấn luyện; bệnh hoặc cây trồng ngoài phạm vi này có thể bị gán nhầm vào một lớp có sẵn.
- PlantVillage chủ yếu gồm ảnh lá trong điều kiện tương đối kiểm soát. Ảnh thực địa có nền phức tạp, ánh sáng khác nhau, nhiều lá hoặc nhiều triệu chứng có thể làm giảm độ chính xác.
- Confidence là điểm đầu ra của mô hình, không phải xác suất chẩn đoán chính xác đã được hiệu chuẩn.
- Ảnh không rõ, lá quá nhỏ, nhiều lá trong cùng khung hình hoặc cây không thuộc phạm vi nên được xem là trường hợp cần kiểm tra lại.
- Cần đánh giá bổ sung bằng ảnh thực tế tại địa phương trước khi ứng dụng rộng rãi.

## Hướng dẫn thuốc bảo vệ thực vật

Thông tin triệu chứng và hướng xử lý trong ứng dụng là nội dung tham khảo tĩnh. Không xem kết quả AI là đơn thuốc. Trước khi sử dụng sản phẩm cần xác nhận đúng cây trồng và đối tượng phòng trừ trên nhãn, kiểm tra sản phẩm còn được phép lưu hành/đăng ký tại Việt Nam, tuân thủ liều lượng, cách dùng, trang bị bảo hộ và thời gian cách ly (PHI). Không tự ý tăng liều hoặc phối trộn sản phẩm; hỏi cán bộ kỹ thuật/bảo vệ thực vật địa phương khi triệu chứng không rõ hoặc bệnh lan nhanh.

## Công nghệ

Python · TensorFlow/Keras · MobileNetV2 · TensorFlow Datasets · NumPy · Pillow · scikit-learn · Matplotlib · Streamlit.

## Nguồn gốc và ghi nhận

Repository này được phát triển từ fork của [aliialzein/Plant-Disease-Detection](https://github.com/aliialzein/Plant-Disease-Detection). Notebook huấn luyện ban đầu đã được thay bằng pipeline Colab của dự án; các thay đổi, nội dung và kết quả được duy trì trong repository này. Vui lòng giữ thông tin nguồn gốc khi tiếp tục phân phối hoặc phát triển dựa trên mã nguồn upstream.

## Disclaimer

Đây là prototype nghiên cứu/giáo dục, không phải hệ thống chẩn đoán nông nghiệp chuyên nghiệp. Kết quả chỉ có giá trị tham khảo và không nên là căn cứ duy nhất để quyết định xử lý cây hoặc sử dụng thuốc.
