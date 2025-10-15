# ==========================================
# 🩺 Breast Ultrasound AI Diagnostic App (Final)
# ==========================================

import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from io import BytesIO

# ==============================
# 🔹 Download pretrained models
# ==============================
# ✅ Model phân đoạn (được lưu trong thư mục trên Drive)
seg_model_dir = "Seg_model_saved"
seg_model_gdrive = "https://drive.google.com/drive/folders/1tMGSiSCPbzvxOUEX9qnMOtQEazzDWMKe?usp=sharing"

# ✅ Model phân loại (.keras)
clf_model_path = "Classifier_model.keras"
clf_model_id = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"  # link file .keras

# ✅ Gdown không tải thư mục được, nên ta cần gợi ý người dùng tải sẵn
if not os.path.exists(seg_model_dir):
    st.warning("⚠️ Model phân đoạn chưa có sẵn. Hãy tải thủ công thư mục từ link Drive và nén lại thành .zip để upload vào workspace của bạn.")
    st.markdown(f"[📁 Tải model phân đoạn tại đây]({seg_model_gdrive})")

# ✅ Tải model phân loại nếu chưa có
if not os.path.exists(clf_model_path):
    gdown.download(f"https://drive.google.com/uc?id={clf_model_id}", clf_model_path, quiet=False)

# ==============================
# 🔹 Load models safely
# ==============================
@st.cache_resource
def load_models():
    from tensorflow import keras
    if hasattr(keras.config, "enable_unsafe_deserialization"):
        keras.config.enable_unsafe_deserialization()

    # ⚙️ Load models (compile=False để tránh lỗi marshal / optimizer)
    classifier = tf.keras.models.load_model(clf_model_path, compile=False)
    segmentor = tf.keras.models.load_model(seg_model_dir, compile=False)

    return classifier, segmentor

# ==============================
# 🔹 Image preprocessing
# ==============================
def classify_preprop(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def segment_preprop(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def segment_postprop(image, mask):
    # chọn lớp có xác suất cao nhất
    mask = np.argmax(mask, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    # overlay: chỉ hiển thị vùng có khối u
    return np.squeeze(image) * (mask > 0)

# ==============================
# 🔹 Pipeline dự đoán
# ==============================
def predict_pipeline(file, classifier, segmentor):
    image_bytes = file.read()
    image_to_classify = classify_preprop(image_bytes)
    image_to_segment = segment_preprop(image_bytes)

    with tf.device('/CPU:0'):
        clf_out = classifier.predict(image_to_classify, verbose=0)
        seg_out = segmentor.predict(image_to_segment, verbose=0)[0]

    seg_image = segment_postprop(image_to_segment, seg_out)
    return clf_out, seg_image, image_bytes

# ==============================
# 🔹 Streamlit UI
# ==============================
st.set_page_config(page_title="Breast Ultrasound AI", layout="wide", page_icon="🩺")
st.sidebar.title("📘 Navigation")

app_mode = st.sidebar.selectbox('Chọn trang', [
    'Ứng dụng chẩn đoán',
    'Thông tin chung',
    'Thống kê về dữ liệu huấn luyện'
])

# -----------------------------
# Trang 1: Thông tin
# -----------------------------
if app_mode == 'Thông tin chung':
    st.title('👨‍🎓 Giới thiệu về thành viên')
    st.markdown('<h4>Lê Vũ Anh Tin - 11TH</h4>', unsafe_allow_html=True)
    try:
        st.image('Tin.jpg', caption='Lê Vũ Anh Tin', width=250)
        st.image('school.jpg', caption='Trường THPT Chuyên Nguyễn Du', width=250)
    except:
        st.info("🖼️ Ảnh giới thiệu chưa được tải lên.")

# -----------------------------
# Trang 2: Thống kê dữ liệu
# -----------------------------
elif app_mode == 'Thống kê về dữ liệu huấn luyện':
    st.title('📊 Thống kê tổng quan về tập dữ liệu')
    st.caption("""
    Tập dữ liệu **Breast Ultrasound Images (BUI)** được kết hợp từ hai nguồn:
    - BUSI (Arya Shah, Kaggle)
    - BUS-UCLM (Orvile, Kaggle)
    
    Tổng cộng **1578 ảnh siêu âm vú** có mask phân đoạn tương ứng.
    """)
    st.markdown("[🔗 Link dataset gốc](https://drive.google.com/drive/folders/1eSAA5pMuEz1GgATBmvXbjjaihO1yBo1l?usp=drive_link)")

# -----------------------------
# Trang 3: Ứng dụng chẩn đoán
# -----------------------------
elif app_mode == 'Ứng dụng chẩn đoán':
    st.title('🩺 Ứng dụng chẩn đoán bệnh ung thư vú từ ảnh siêu âm')

    if not os.path.exists(seg_model_dir):
        st.error("❌ Model phân đoạn chưa sẵn sàng. Hãy tải thư mục từ link Drive trước khi chạy.")
        st.stop()

    classifier, segmentor = load_models()

    file = st.file_uploader("📤 Tải ảnh siêu âm (JPG hoặc PNG)", type=["jpg", "png"])
    if file is None:
        st.info("👆 Vui lòng tải ảnh lên để bắt đầu chẩn đoán.")
    else:
        slot = st.empty()
        slot.text("⏳ Đang phân tích ảnh...")

        clf_out, seg_image, image_bytes = predict_pipeline(file, classifier, segmentor)
        input_image = Image.open(BytesIO(image_bytes))

        col1, col2 = st.columns(2)
        with col1:
            st.image(input_image, caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(seg_image, caption="Kết quả phân đoạn", use_container_width=True)

        class_names = ['benign', 'malignant', 'normal']
        result = class_names[np.argmax(clf_out)]

        # Hiển thị kết quả chẩn đoán
        if result == 'benign':
            st.success("🟢 Kết luận: Khối u lành tính.")
        elif result == 'malignant':
            st.error("🔴 Kết luận: Ung thư vú ác tính.")
        else:
            st.info("⚪ Kết luận: Không phát hiện khối u.")

        slot.success("✅ Hoàn tất chẩn đoán!")

        # Biểu đồ xác suất
        chart_df = pd.DataFrame({
            'Loại chẩn đoán': ["Lành tính", "Ác tính", "Bình thường"],
            'Xác suất (%)': [clf_out[0,0]*100, clf_out[0,1]*100, clf_out[0,2]*100]
        })
        chart = alt.Chart(chart_df).mark_bar().encode(
            x='Loại chẩn đoán',
            y='Xác suất (%)',
            color='Loại chẩn đoán'
        )
        st.altair_chart(chart, use_container_width=True)

        st.write(f"- **Khối u lành tính:** {clf_out[0,0]*100:.1f}%")
        st.write(f"- **Ung thư vú:** {clf_out[0,1]*100:.1f}%")
        st.write(f"- **Bình thường:** {clf_out[0,2]*100:.1f}%")
