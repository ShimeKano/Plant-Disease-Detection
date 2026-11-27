import json
from pathlib import Path

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Plant Disease AI", page_icon="🌱", layout="centered")

st.title("🌱 Plant Disease AI")
st.caption("Nhận diện 15 nhóm tình trạng bệnh trên cà chua, khoai tây và ớt chuông.")

IMG_SIZE = 128
MODEL_DIR = Path("model")
NEW_MODEL_PATH = MODEL_DIR / "best_model.keras"
LEGACY_MODEL_PATH = MODEL_DIR / "best_model.h5"
MODEL_PATH = str(NEW_MODEL_PATH if NEW_MODEL_PATH.exists() else LEGACY_MODEL_PATH)

# IMPORTANT: this order matches Keras flow_from_directory alphabetical class_indices
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato__Bacterial_spot",
    "Tomato__Early_blight",
    "Tomato__Late_blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__healthy",
]

DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "name": "🌶️ Ớt chuông — Bệnh đốm vi khuẩn",
        "symptoms": "Đốm nhỏ màu nâu/đen trên lá, đôi khi có quầng vàng.",
        "treatment": "Loại bỏ lá bị bệnh, giữ lá khô, tưới vào gốc và tăng thông thoáng. Một số thuốc gốc đồng có thể được dùng để quản lý bệnh nếu nhãn sản phẩm cho phép dùng trên ớt.",
        "active": "Hoạt chất thường gặp: hợp chất đồng (copper). Vi khuẩn có thể kháng đồng, nên không xem đây là thuốc chữa chắc chắn.",
    },
    "Pepper__bell___healthy": {
        "name": "🌶️ Ớt chuông — Khỏe mạnh",
        "symptoms": "Lá xanh, phát triển bình thường, không thấy triệu chứng bệnh rõ ràng.",
        "treatment": "Chưa thấy dấu hiệu bệnh thuộc các class của model. Tiếp tục theo dõi cây.",
        "active": "Không cần thuốc theo kết quả này.",
    },
    "Potato___Early_blight": {
        "name": "🥔 Khoai tây — Bệnh cháy lá sớm",
        "symptoms": "Đốm nâu tròn, thường có vòng đồng tâm; lá có thể vàng và khô dần.",
        "treatment": "Loại bỏ lá bệnh, giữ tán lá khô, tăng thông thoáng và vệ sinh tàn dư cây. Nếu cần thuốc, dùng thuốc trừ nấm được đăng ký cho khoai tây và đúng bệnh trên nhãn.",
        "active": "Hoạt chất thường được dùng trong chương trình quản lý bệnh có thể gồm chlorothalonil hoặc nhóm thuốc trừ nấm khác tùy nhãn và khu vực.",
    },
    "Potato___Late_blight": {
        "name": "🥔 Khoai tây — Bệnh mốc sương",
        "symptoms": "Vết nâu sẫm/đen lớn, lan nhanh; khi ẩm có thể xuất hiện lớp mốc trắng.",
        "treatment": "Cách ly và loại bỏ phần bệnh, giữ tán lá khô và theo dõi sát. Mốc sương phát triển rất nhanh; nếu dùng thuốc cần thuốc được đăng ký cho bệnh này và phải phun theo nhãn.",
        "active": "Thuốc chuyên quản lý Phytophthora thường được dùng theo chương trình luân phiên nhóm hoạt chất; không nên tự pha hoặc tăng liều.",
    },
    "Potato___healthy": {
        "name": "🥔 Khoai tây — Khỏe mạnh",
        "symptoms": "Lá xanh bình thường, không thấy dấu hiệu bệnh rõ ràng.",
        "treatment": "Chưa thấy dấu hiệu bệnh thuộc các class của model. Tiếp tục theo dõi cây.",
        "active": "Không cần thuốc theo kết quả này.",
    },
    "Tomato__Bacterial_spot": {
        "name": "🍅 Cà chua — Bệnh đốm vi khuẩn",
        "symptoms": "Đốm nhỏ sẫm màu, thường có quầng vàng; vết bệnh nặng có thể làm lá rụng.",
        "treatment": "Loại bỏ lá bệnh, tránh tưới phun lên lá, giữ khoảng cách thông thoáng và vệ sinh dụng cụ. Một số thuốc gốc đồng có thể được dùng để quản lý bệnh nếu nhãn cho phép.",
        "active": "Hoạt chất thường gặp: hợp chất đồng (copper). Kháng đồng đã được ghi nhận nên hiệu quả có thể hạn chế.",
    },
    "Tomato__Early_blight": {
        "name": "🍅 Cà chua — Bệnh cháy lá sớm",
        "symptoms": "Đốm nâu với các vòng đồng tâm, thường xuất hiện trước ở lá già.",
        "treatment": "Tỉa lá bệnh ở mức vừa phải, giữ lá khô, tưới vào gốc, tăng thông thoáng và vệ sinh tàn dư. Nếu cần thuốc, chọn thuốc trừ nấm được đăng ký cho cà chua và bệnh này.",
        "active": "Chlorothalonil là một hoạt chất thường xuất hiện trong chương trình quản lý cháy lá sớm; luôn kiểm tra nhãn sản phẩm tại Việt Nam.",
    },
    "Tomato__Late_blight": {
        "name": "🍅 Cà chua — Bệnh mốc sương",
        "symptoms": "Vết nâu lớn lan nhanh; trong điều kiện ẩm có thể xuất hiện lớp mốc trắng.",
        "treatment": "Loại bỏ phần bệnh, giảm ẩm trên tán lá, tăng thông thoáng và theo dõi sát. Nếu bệnh lan nhanh, dùng thuốc được đăng ký cho late blight/Phytophthora trên cà chua theo đúng nhãn.",
        "active": "Cần thuốc chuyên quản lý Phytophthora; hoạt chất và lịch phun phải theo nhãn sản phẩm và tình hình kháng thuốc.",
    },
    "Tomato__Leaf_Mold": {
        "name": "🍅 Cà chua — Bệnh mốc lá",
        "symptoms": "Mặt trên lá có vùng vàng; mặt dưới có thể có lớp mốc xanh vàng đến nâu ôliu.",
        "treatment": "Giảm độ ẩm, tăng thông gió, tránh để lá ướt kéo dài và loại bỏ tàn dư bệnh. Một số sản phẩm gốc đồng có thể được dùng nếu được đăng ký cho bệnh này.",
        "active": "Copper hydroxide có dữ liệu sử dụng trong quản lý mốc lá; phải kiểm tra nhãn sản phẩm.",
    },
    "Tomato__Septoria_leaf_spot": {
        "name": "🍅 Cà chua — Bệnh đốm lá Septoria",
        "symptoms": "Nhiều đốm nhỏ tròn, tâm xám/nâu nhạt, viền sẫm; có thể có chấm đen nhỏ ở giữa.",
        "treatment": "Tỉa lá bệnh, giữ tán lá khô, tưới vào gốc và vệ sinh tàn dư. Nếu cần thuốc, dùng thuốc trừ nấm được đăng ký cho Septoria trên cà chua.",
        "active": "Chlorothalonil là một hoạt chất thường được dùng trong chương trình quản lý bệnh đốm lá; kiểm tra nhãn và thời gian cách ly.",
    },
    "Tomato__Spider_mites_Two_spotted_spider_mite": {
        "name": "🍅 Cà chua — Nhện đỏ hai chấm",
        "symptoms": "Lá có nhiều chấm vàng li ti, bạc màu; nặng có thể xuất hiện tơ nhện mảnh.",
        "treatment": "Kiểm tra mặt dưới lá, rửa/tưới nhẹ để giảm mật số và bảo vệ thiên địch. Xà phòng diệt côn trùng hoặc dầu làm vườn có thể hiệu quả khi tiếp xúc trực tiếp với nhện.",
        "active": "Xà phòng diệt côn trùng (potassium salts of fatty acids) hoặc dầu làm vườn/neem; cần phun trúng mặt dưới lá và lặp lại theo nhãn.",
    },
    "Tomato__Target_Spot": {
        "name": "🍅 Cà chua — Bệnh đốm vòng (Target Spot)",
        "symptoms": "Đốm nâu có các vòng đồng tâm giống hình bia; bệnh nặng có thể làm lá vàng và rụng.",
        "treatment": "Giữ tán lá khô, tăng thông thoáng, vệ sinh lá bệnh và tàn dư. Nếu cần thuốc, dùng thuốc trừ nấm được đăng ký cho target spot trên cà chua.",
        "active": "Hoạt chất phụ thuộc sản phẩm và khu vực; nên chọn theo nhãn đăng ký thay vì tự dùng một thuốc cố định.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "🍅 Cà chua — Virus xoăn vàng lá",
        "symptoms": "Lá vàng, nhỏ lại và cuốn/xoăn; cây có thể sinh trưởng kém.",
        "treatment": "Không có thuốc diệt virus để chữa cây đã nhiễm. Loại bỏ cây bệnh nặng, kiểm soát côn trùng môi giới và dùng cây giống sạch bệnh.",
        "active": "Không có thuốc chữa virus trực tiếp; quản lý vector và vệ sinh cây trồng là chính.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "name": "🍅 Cà chua — Virus khảm",
        "symptoms": "Lá có các mảng xanh đậm/xanh nhạt xen kẽ dạng khảm, đôi khi bị biến dạng.",
        "treatment": "Không có thuốc chữa virus trực tiếp. Loại bỏ cây nhiễm nặng, vệ sinh tay/dụng cụ và dùng hạt/cây giống sạch bệnh.",
        "active": "Không có thuốc diệt virus trực tiếp.",
    },
    "Tomato__healthy": {
        "name": "🍅 Cà chua — Khỏe mạnh",
        "symptoms": "Lá xanh tương đối đồng đều, không có triệu chứng bệnh rõ ràng.",
        "treatment": "Chưa thấy dấu hiệu bệnh thuộc các class của model. Tiếp tục theo dõi cây.",
        "active": "Không cần thuốc theo kết quả này.",
    },
}

@st.cache_resource
def load_model(model_path, modified_time):
    # modified_time invalidates Streamlit's cache when a model file is replaced.
    return tf.keras.models.load_model(model_path, compile=False)

CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
if CLASS_NAMES_PATH.exists():
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        CLASS_NAMES = json.load(file)

missing_info = [name for name in CLASS_NAMES if name not in DISEASE_INFO]
if missing_info:
    st.error(f"Thiếu thông tin hiển thị cho các lớp: {missing_info}")
    st.stop()

# Automatically discover model artifacts placed in model/.
MODEL_FILES = sorted(
    [p for p in MODEL_DIR.glob("*.keras") if p.is_file()] +
    [p for p in MODEL_DIR.glob("*.h5") if p.is_file()],
    key=lambda p: (p.suffix != ".keras", p.name.lower())
)
if not MODEL_FILES:
    st.error("Không tìm thấy model .keras hoặc .h5 trong thư mục model/.")
    st.stop()

MODEL_OPTIONS = {p.name: p for p in MODEL_FILES}
st.subheader("⚙️ Chọn model để so sánh")
st.caption("Ứng dụng tự tìm các file .keras và .h5 trong thư mục model/. Hãy đặt hai model cần đánh giá vào thư mục này.")

if len(MODEL_FILES) >= 2:
    left, right = st.columns(2)
    names = list(MODEL_OPTIONS.keys())
    default_a = names[0]
    default_b = names[1]
    with left:
        model_a_name = st.selectbox("Model A", names, index=0, key="model_a")
    other_names = [name for name in names if name != model_a_name]
    with right:
        model_b_name = st.selectbox("Model B", other_names, index=0, key="model_b")
    selected_models = [MODEL_OPTIONS[model_a_name], MODEL_OPTIONS[model_b_name]]
    st.info("Hai lựa chọn hiện tại là hai tệp model riêng. Nếu chúng chỉ là hai định dạng lưu của cùng một lần huấn luyện thì kết quả không thể được xem là so sánh hai phiên bản AI độc lập.")
else:
    selected_models = [MODEL_FILES[0]]
    st.warning("Hiện chỉ tìm thấy một model. Thêm model thứ hai vào thư mục model/ để bật so sánh A/B.")

uploaded_file = st.file_uploader("📸 Chọn ảnh lá cây để kiểm tra", type=["jpg", "jpeg", "png"])

def prepare_input(image):
    resized = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
    input_tensor = prepare_input(image)
    results = []

    for model_path in selected_models:
        try:
            model = load_model(str(model_path), model_path.stat().st_mtime)
            output_classes = int(model.output_shape[-1])
            if output_classes != len(CLASS_NAMES):
                st.error(f"{model_path.name}: model có {output_classes} đầu ra, nhưng class_names.json có {len(CLASS_NAMES)} lớp.")
                continue
            scores = model.predict(input_tensor, verbose=0)[0]
            pred_index = int(np.argmax(scores))
            results.append({
                "name": model_path.name,
                "scores": scores,
                "index": pred_index,
                "class": CLASS_NAMES[pred_index],
                "confidence": float(scores[pred_index]) * 100,
            })
        except Exception as exc:
            st.error(f"Không thể tải/chạy {model_path.name}: {exc}")

    if results:
        st.divider()
        st.subheader("🔬 So sánh dự đoán")
        columns = st.columns(len(results))
        for column, result in zip(columns, results):
            with column:
                st.markdown(f"**{result['name']}**")
                st.success(DISEASE_INFO[result["class"]]["name"])
                st.metric("Confidence", f"{result['confidence']:.2f}%")
                st.caption("Top 3")
                for idx in np.argsort(result["scores"])[-3:][::-1]:
                    cls = CLASS_NAMES[int(idx)]
                    st.write(f"{DISEASE_INFO[cls]['name']} — {float(result['scores'][idx])*100:.2f}%")

        if len(results) == 2:
            agree = results[0]["class"] == results[1]["class"]
            if agree:
                st.success("Hai model đưa ra cùng một nhãn dự đoán.")
            else:
                st.warning("Hai model không thống nhất. Không nên chọn thuốc hoặc xử lý cây chỉ dựa trên một trong hai kết quả.")
                comparison = {
                    "Model A": DISEASE_INFO[results[0]["class"]]["name"],
                    "Model B": DISEASE_INFO[results[1]["class"]]["name"],
                }
                st.table(comparison)

        # Treatment guidance is shown only when the selected models agree.
        if len(results) == 1 or (len(results) == 2 and results[0]["class"] == results[1]["class"]):
            info = DISEASE_INFO[results[0]["class"]]
            st.divider()
            st.subheader("🌿 Triệu chứng và hướng xử lý tham khảo")
            st.info(f"**Triệu chứng thường gặp:** {info['symptoms']}")
            st.write(info["treatment"])
            st.caption(info["active"])
        else:
            st.info("Do kết quả đang mâu thuẫn, ứng dụng tạm ẩn hướng xử lý/thuốc để tránh đưa ra khuyến nghị dựa trên kết luận chưa chắc chắn.")

        st.warning(
            "⚠️ Confidence là điểm đầu ra của mô hình, không đồng nghĩa xác suất chẩn đoán đúng. "
            "Đây là gợi ý tham khảo, không phải chẩn đoán chuyên gia. Thuốc bảo vệ thực vật phải đúng cây + đúng bệnh trên nhãn, đúng liều và thời gian cách ly."
        )

st.divider()
st.caption(
    "Model được huấn luyện trên PlantVillage với 15 class. "
    "Ảnh thực địa có thể khác dữ liệu huấn luyện. Muốn kết luận model nào tốt hơn, hãy đánh giá cả hai trên cùng một tập ảnh thực tế có nhãn đúng bằng notebook so sánh."
)
