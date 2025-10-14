# ==========================================
# 🩺 Breast Ultrasound AI Diagnostic App
# ==========================================

import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO

# ==============================
# 🔹 Download pretrained models
# ==============================
seg_model_path = "Seg_model.keras"
seg_model_id = "1JOgis3Yn8YuwZGxsYAj5l-mTvKy7vG2C"   # 🆕 Segmentation model
clf_model_path = "Classifier_model_2.keras"
clf_model_id = "1wgAMMN4qV1AHZNKe09f4xj9idO1rL7C3"  # Classification model

if not os.path.exists(seg_model_path):
    gdown.download(f"https://drive.google.com/uc?id={seg_model_id}", seg_model_path, quiet=False)

if not os.path.exists(clf_model_path):
    gdown.download(f"https://drive.google.com/uc?id={clf_model_id}", clf_model_path, quiet=False)

# ==============================
# 🔹 Load models safely
# ==============================
def load_model():
    import tensorflow as tf
    from tensorflow import keras   # ✅ Dùng keras tích hợp trong TensorFlow (ổn định hơn)

    def dice_loss(y_true, y_pred):
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        return 1 - 2 * intersection / union

    # ⚙️ Dòng này chỉ có trong Keras độc lập, nhưng không trong TensorFlow
    # → vì vậy ta chỉ dùng safe_mode=False để bỏ giới hạn an toàn
    classifier = tf.keras.models.load_model("Classifier_model_2.h5")
    segmentor = tf.keras.models.load_model(
        "Seg_model.keras",
        custom_objects={"dice_loss": dice_loss},
        safe_mode=False,          # ⚠️ Bỏ kiểm tra Lambda layer
        compile=False
    )

    return classifier, segmentor

# ==============================
# 🔹 Image preprocessing
# ==============================
def classify_preprop(image_file): 
    image = Image.open(BytesIO(image_file)).convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def segment_preprop(image_file):
    image = Image.open(BytesIO(image_file)).convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def segment_postprop(image, mask):   
    image = np.squeeze(image)
    mask = np.squeeze(mask)
    mask = np.expand_dims(mask, axis=2)
    return image * mask

def preprocessing_uploader(file, classifier, segmentor):
    image_file = file.read()
    image_to_classify = classify_preprop(image_file)
    image_to_segment = segment_preprop(image_file)

    classify_output = classifier.predict(image_to_classify)
    segment_output = segmentor.predict(image_to_segment)[0]
    segment_output = segment_postprop(image_to_segment, segment_output)
    return classify_output, segment_output

# ==============================
# 🔹 Streamlit App UI
# ==============================
st.sidebar.title("📘 Navigation")
app_mode = st.sidebar.selectbox('Chọn trang', [
    'Ứng dụng chẩn đoán', 
    'Thông tin chung',
    'Thống kê về dữ liệu huấn luyện'
])

# -----------------------------
# Trang thông tin
# -----------------------------
if app_mode == 'Thông tin chung':
    st.title('👨‍🎓 Giới thiệu về thành viên')
    st.markdown('<h4>Lê Vũ Anh Tin - 11TH</h4>', unsafe_allow_html=True)
    tin_ava = Image.open('Tin.jpg')
    st.image(tin_ava, caption='Lê Vũ Anh Tin')
    st.markdown('<h5>Trường THPT Chuyên Nguyễn Du</h5>', unsafe_allow_html=True)
    school_ava = Image.open('school.jpg')
    st.image(school_ava, caption='Trường THPT Chuyên Nguyễn Du')

# -----------------------------
# Trang thống kê dữ liệu
# -----------------------------
elif app_mode == 'Thống kê về dữ liệu huấn luyện': 
    st.title('📊 Thống kê tổng quan về tập dữ liệu')
    st.caption("""
    Trong nghiên cứu này, tập dữ liệu **Breast Ultrasound Images (BUI)** được kết hợp từ hai nguồn công khai:
    - BUSI (Arya Shah, Kaggle)
    - BUS-UCLM (Orvile, Kaggle)
    Tổng cộng gồm **1578 ảnh siêu âm vú** với mặt nạ phân đoạn tương ứng.
    """)
    st.caption('Chi tiết dataset: https://drive.google.com/drive/folders/1eSAA5pMuEz1GgATBmvXbjjaihO1yBo1l?usp=drive_link')

# -----------------------------
# Trang ứng dụng chẩn đoán
# -----------------------------
elif app_mode == 'Ứng dụng chẩn đoán':
    st.title('🩺 Ứng dụng chẩn đoán bệnh ung thư vú từ ảnh siêu âm')

    # Load models (chỉ chạy 1 lần)
    classifier, segmentor = load_model()

    file = st.file_uploader("📤 Tải ảnh siêu âm vú (jpg/png)", type=["jpg", "png"])

    if file is None:
        st.info('👆 Vui lòng tải ảnh siêu âm lên để bắt đầu chẩn đoán.')
    else:
        slot = st.empty()
        slot.text('⏳ Đang phân tích ảnh...')

        classify_output, segment_output = preprocessing_uploader(file, classifier, segmentor)
        test_image = Image.open(file)
        st.image(test_image, caption="Ảnh đầu vào", width=400)

        class_names = ['benign', 'malignant', 'normal']
        result_name = class_names[np.argmax(classify_output)]

        st.image(segment_output, caption="Ảnh phân đoạn khối u", width=400)

        # Hiển thị kết quả
        if result_name == 'benign':
            st.error('🟢 Chẩn đoán: Bệnh nhân có khối u lành tính.')
        elif result_name == 'malignant':
            st.warning('🔴 Chẩn đoán: Bệnh nhân mắc ung thư vú.')
        else:
            st.success('⚪ Chẩn đoán: Không phát hiện dấu hiệu khối u.')

        slot.success('✅ Hoàn tất chẩn đoán!')

        # Biểu đồ xác suất
        bar_frame = pd.DataFrame({
            'Loại chẩn đoán': ["Lành tính", "Ác tính", "Bình thường"],
            'Xác suất dự đoán (%)': [
                classify_output[0,0]*100, classify_output[0,1]*100, classify_output[0,2]*100
            ]
        })
        bar_chart = alt.Chart(bar_frame).mark_bar().encode(
            x='Loại chẩn đoán',
            y='Xác suất dự đoán (%)',
            color='Loại chẩn đoán'
        )
        st.altair_chart(bar_chart, use_container_width=True)

        st.write(f"- **Khối u lành tính:** {classify_output[0,0]*100:.2f}%")
        st.write(f"- **Ung thư vú:** {classify_output[0,1]*100:.2f}%")
        st.write(f"- **Bình thường:** {classify_output[0,2]*100:.2f}%")
