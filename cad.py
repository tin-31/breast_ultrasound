import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import uuid
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# ====== GOOGLE SETUP ======
def connect_google_services():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive.appdata"
    ]
    import json
    creds = Credentials.from_service_account_info(json.loads(st.secrets["google"]["service_account_info"]), scopes=scope)

    gc = gspread.authorize(creds)
    drive_service = build('drive', 'v3', credentials=creds)
    return gc, drive_service

def upload_to_drive(image_bytes, drive_service):
    file_id = str(uuid.uuid4())
    file_metadata = {
        'name': f"{file_id}.png",
        'parents': ['1MNgXaIZsWuxLb6JfE8eiANbXn5Rwxu8G'],  # Google Drive folder ID
    }
    media = MediaIoBaseUpload(BytesIO(image_bytes), mimetype='image/png')
    uploaded_file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return uploaded_file.get('id')

def write_to_sheet(gc, prediction, image_drive_id):
    sheet_id = "1yfnttAYT93SipMKfHW6tGoRBMQYTz7wwZ970cQp7HiE"  # Google Sheet ID
    worksheet = gc.open_by_key(sheet_id).sheet1
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_name = ['Lành tính', 'Ác tính', 'Bình thường'][np.argmax(prediction)]
    row = [
        now,
        result_name,
        round(prediction[0][0] * 100, 2),
        round(prediction[0][1] * 100, 2),
        round(prediction[0][2] * 100, 2),
        f"https://drive.google.com/file/d/{image_drive_id}/view"
    ]
    worksheet.append_row(row)

# ====== MODEL LOADING ======
def load_model():
    def dice_loss(y_true, y_pred):
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        return 1 - 2 * intersection / union

    classifier = tf.keras.models.load_model('Classifier_model_2.h5')
    segmentor = tf.keras.models.load_model('Seg_model.h5', custom_objects={'dice_loss': dice_loss})
    return classifier, segmentor

# ====== IMAGE PREPROCESSING ======
def classify_preprop(image_bytes): 
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)

def segment_preprop(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = img_to_array(image)
    return np.expand_dims(image, axis=0)

def segment_postprop(image, mask):   
    image = np.squeeze(image)
    mask = np.squeeze(mask)
    mask = np.expand_dims(mask, axis=2)
    return image * mask

def predict_models(image_bytes, classifier, segmentor):
    image_for_classify = classify_preprop(image_bytes)
    image_for_segment = segment_preprop(image_bytes)

    classify_output = classifier.predict(image_for_classify)
    segment_output = segmentor.predict(image_for_segment)[0]
    segment_output = segment_postprop(image_for_segment, segment_output)
    return classify_output, segment_output

# ====== STREAMLIT UI ======
st.set_page_config(page_title="Ứng dụng chẩn đoán ung thư vú", layout="wide")
app_mode = st.sidebar.selectbox('Chọn trang', ['Thông tin chung', 'Thống kê về dữ liệu huấn luyện', 'Ứng dụng chẩn đoán'])

if app_mode == 'Thông tin chung':
    st.title('Giới thiệu về thành viên')
    st.markdown('<p style="font-size:25px">Học sinh thực hiện: <strong>Lê Vũ Anh Tin - 10TH</strong></p>', unsafe_allow_html=True)
    st.image('Tin.jpg')
    st.markdown('<p style="font-size:25px">Trường THPT chuyên Nguyễn Du</p>', unsafe_allow_html=True)
    st.image('school.jpg')

elif app_mode == 'Thống kê về dữ liệu huấn luyện':
    st.title('Thống kê tập dữ liệu siêu âm vú')
    st.markdown("""
    - Dữ liệu từ bệnh viện Baheya, Cairo, Ai Cập.
    - 780 ảnh siêu âm vú từ 600 bệnh nhân nữ.
    - Chia 3 loại: bình thường, lành tính, ác tính.
    - Nguồn: [Kaggle Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data)
    """)

elif app_mode == 'Ứng dụng chẩn đoán':
    st.title('Ứng dụng chẩn đoán ung thư vú dựa trên ảnh siêu âm')

    classifier, segmentor = load_model()
    file = st.file_uploader("Tải ảnh siêu âm vú (jpg hoặc png):", type=["jpg", "png"])

    if file:
        image_bytes = file.read()
        st.image(image_bytes, caption="Ảnh đầu vào", width=400)

        with st.spinner("Đang chẩn đoán..."):
            gc, drive_service = connect_google_services()
            image_drive_id = upload_to_drive(image_bytes, drive_service)
            classify_output, segment_output = predict_models(image_bytes, classifier, segmentor)
            write_to_sheet(gc, classify_output, image_drive_id)

        class_names = ['benign', 'malignant', 'normal']
        result_label = class_names[np.argmax(classify_output)]

        st.image(segment_output, caption="Ảnh phân vùng khối u", width=400)

        if result_label == 'benign':
            st.error('Kết quả: **Khối u lành tính.**')
        elif result_label == 'malignant':
            st.warning('Kết quả: **Bệnh nhân mắc ung thư vú.**')
        else:
            st.success('Kết quả: **Không phát hiện khối u.**')

        # Biểu đồ xác suất
        bar_data = pd.DataFrame({
            'Loại chẩn đoán': ["Lành tính", "Ác tính", "Bình thường"],
            'Xác suất dự đoán (%)': [classify_output[0,0]*100, classify_output[0,1]*100, classify_output[0,2]*100]
        })
        bar_chart = alt.Chart(bar_data).mark_bar().encode(x='Loại chẩn đoán', y='Xác suất dự đoán (%)')
        st.altair_chart(bar_chart, use_container_width=True)

        st.write(f'- **Lành tính**: *{round(classify_output[0,0]*100, 2)}%*')
        st.write(f'- **Ác tính**: *{round(classify_output[0,1]*100, 2)}%*')
        st.write(f'- **Bình thường**: *{round(classify_output[0,2]*100, 2)}%*')

    else:
        st.info("Vui lòng tải ảnh để bắt đầu.")

