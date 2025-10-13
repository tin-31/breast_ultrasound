import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ===========================
# 🧩 TẢI MODEL TỪ GOOGLE DRIVE
# ===========================
SEG_MODEL_PATH = "Seg_model.h5"
SEG_MODEL_ID = "1PC4ZNJJB5n-JKSc1mmyOLeQ5tClx4hcP"
if not os.path.exists(SEG_MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={SEG_MODEL_ID}", SEG_MODEL_PATH, quiet=False)

CLF_MODEL_PATH = "Classifier_model_2.h5"
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"
if not os.path.exists(CLF_MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={CLF_MODEL_ID}", CLF_MODEL_PATH, quiet=False)


# ===========================
# ⚙️ HÀM LOAD MODEL
# ===========================
@st.cache_resource
def load_model():
    def dice_loss(y_true, y_pred):
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        return 1 - 2 * intersection / union

    try:
        classifier = tf.keras.models.load_model(CLF_MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Không thể tải model phân loại: {e}")
        st.stop()

    try:
        # ⚠️ thêm safe_mode=False để tránh lỗi lambda layer
        segmentor = tf.keras.models.load_model(
            SEG_MODEL_PATH,
            custom_objects={'dice_loss': dice_loss},
            safe_mode=False
        )
    except Exception as e:
        st.error(f"❌ Không thể tải model phân đoạn: {e}")
        st.stop()

    return classifier, segmentor


# ===========================
# 📸 HÀM XỬ LÝ ẢNH
# ===========================
def classify_preprop(image_file):
    """Tiền xử lý ảnh cho mô hình phân loại"""
    classifyInputShape = (224, 224)
    image = Image.open(BytesIO(image_file)).convert("RGB")
    image = image.resize(classifyInputShape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def segment_preprop(image_file):
    """Tiền xử lý ảnh cho mô hình phân đoạn"""
    segmentInputShape = (256, 256)
    image = Image.open(BytesIO(image_file)).convert('RGB')
    image = image.resize(segmentInputShape)
    image = np.array(image) / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def segment_postprop(image, mask):
    """Hậu xử lý: áp mặt nạ khối u lên ảnh gốc"""
    image = np.squeeze(image)
    mask = np.squeeze(mask)
    mask = np.expand_dims(mask, axis=2)
    return image * mask


def preprocessing_uploader(file, classifier, segmentor):
    image_file = file.read()
    # Phân loại
    image_to_classify = classify_preprop(image_file)
    classify_output = classifier.predict(image_to_classify)
    # Phân đoạn
    image_to_segment = segment_preprop(image_file)
    segment_output = segmentor.predict(image_to_segment)[0]
    segment_output = segment_postprop(image_to_segment, segment_output)
    return classify_output, segment_output


# ===========================
# 🧭 GIAO DIỆN STREAMLIT
# ===========================
app_mode = st.sidebar.selectbox(
    'Chọn trang',
    ['Ứng dụng chẩn đoán', 'Thông tin chung', 'Thống kê về dữ liệu huấn luyện']
)

# ---------------------------
# 1️⃣ Trang thông tin chung
# ---------------------------
if app_mode == 'Thông tin chung':
    st.title('👩‍⚕️ Giới thiệu về nhóm thực hiện')
    st.markdown("""
    <style>
        .big-font { font-size:35px !important; }
        .name { font-size:25px !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font"> Học sinh thực hiện </p>', unsafe_allow_html=True)
    st.markdown('<p class="name"> Lê Vũ Anh Tin - 11TH </p>', unsafe_allow_html=True)
    if os.path.exists('Tin.jpg'):
        st.image('Tin.jpg')

    st.markdown('<p class="big-font"> Trường học tham gia cuộc thi </p>', unsafe_allow_html=True)
    st.markdown('<p class="name"> Trường THPT chuyên Nguyễn Du </p>', unsafe_allow_html=True)
    if os.path.exists('school.jpg'):
        st.image('school.jpg')


# ---------------------------
# 2️⃣ Trang thống kê dữ liệu
# ---------------------------
elif app_mode == 'Thống kê về dữ liệu huấn luyện':
    st.title('📊 Thống kê tổng quan về tập dữ liệu')
    st.caption("""
    Tập dữ liệu **Breast Ultrasound Images (BUI)** được tổng hợp từ hai nguồn công khai:
    - *BUSI Dataset* của Arya Shah (Kaggle)
    - *BUS-UCLM Dataset* của Orvile
    
    Tổng cộng **1578 ảnh siêu âm vú** có mặt nạ phân đoạn tương ứng.
    Dữ liệu được tiền xử lý và resize về **256x256** pixel.
    """)
    st.markdown('[📂 Xem dữ liệu tại đây](https://drive.google.com/drive/folders/1eSAA5pMuEz1GgATBmvXbjjaihO1yBo1l?usp=drive_link)')


# ---------------------------
# 3️⃣ Ứng dụng chẩn đoán
# ---------------------------
elif app_mode == 'Ứng dụng chẩn đoán':
    st.title('🩺 Ứng dụng chẩn đoán ung thư vú từ ảnh siêu âm')

    classifier, segmentor = load_model()

    file = st.file_uploader("Tải ảnh siêu âm vú (JPG hoặc PNG):", type=["jpg", "png"])

    if file is None:
        st.info('⬆️ Vui lòng tải ảnh siêu âm để chẩn đoán.')
    else:
        slot = st.empty()
        slot.text('🔄 Đang xử lý ảnh...')

        classify_output, segment_output = preprocessing_uploader(file, classifier, segmentor)

        test_image = Image.open(file)
        st.image(test_image, caption="Ảnh đầu vào", width=400)

        class_names = ['benign', 'malignant', 'normal']
        result_name = class_names[np.argmax(classify_output)]
        st.image(segment_output, caption="Kết quả phân đoạn khối u", width=400)

        # 🩻 Hiển thị kết quả
        if result_name == 'benign':
            st.success('✅ Kết luận: **Khối u lành tính.**')
        elif result_name == 'malignant':
            st.error('⚠️ Kết luận: **Khối u ác tính (ung thư).**')
        else:
            st.info('🩶 Kết luận: **Không phát hiện khối u.**')

        slot.success('✅ Hoàn tất chẩn đoán!')

        # 📈 Biểu đồ xác suất
        bar_frame = pd.DataFrame({
            'Loại chẩn đoán': ["Lành tính", "Ác tính", "Bình thường"],
            'Xác suất (%)': [classify_output[0,0]*100, classify_output[0,1]*100, classify_output[0,2]*100]
        })
        bar_chart = alt.Chart(bar_frame).mark_bar().encode(
            x='Loại chẩn đoán', y='Xác suất (%)'
        )
        st.altair_chart(bar_chart, use_container_width=True)

        # 🧾 Ghi chú chi tiết
        st.write(f"- **Lành tính:** {classify_output[0,0]*100:.2f}%")
        st.write(f"- **Ác tính:** {classify_output[0,1]*100:.2f}%")
        st.write(f"- **Bình thường:** {classify_output[0,2]*100:.2f}%")
