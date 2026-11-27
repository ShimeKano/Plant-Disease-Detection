# Model artifacts

Ứng dụng tự động phát hiện các file `.keras` và `.h5` trong thư mục này. Để so sánh hai phiên bản độc lập, đặt cả hai model vào đây, dùng tên file khác nhau và đảm bảo cùng thứ tự lớp/tiền xử lý.

## Các file hiện có

- `best_model.keras`: model mặc định ưu tiên của ứng dụng.
- `best_model.h5`: model định dạng HDF5 cũ/legacy.

**Chưa xác minh hai file này là hai lần huấn luyện độc lập.** Có thể chúng chỉ là hai định dạng serialize của cùng một model. Không nên gọi đây là so sánh model cũ và mới cho đến khi xác định được nguồn gốc và training run tương ứng.

Model cần:
- Input RGB 128 × 128 × 3.
- Pixel đầu vào trong khoảng [0,1].
- 15 đầu ra theo đúng thứ tự trong `class_names.json`.
- Kiến trúc tương thích với TensorFlow/Keras hiện dùng.

Ghi lại training run ID, dataset/version, ngày huấn luyện, class mapping, preprocessing và test metrics cho từng model. Không ghi đè model hiện hành khi chưa kiểm thử model mới.
