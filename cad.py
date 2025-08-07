import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image, ImageOps
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from datetime import datetime
import uuid

# === GOOGLE API SETUP ===
def connect_google_services():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive.appdata"
    ]
    creds = Credentials.from_service_account_file("your_service_account.json", scopes=scope)
    gc = gspread.authorize(creds)
    drive_service = build('drive', 'v3', credentials=creds)
    return gc, drive_service

def upload_to_drive(file, drive_service):
    file_id = str(uuid.uuid4())
    file_metadata = {
        'name': f"{file_id}.png",
        'parents': ['1MNgXaIZsWuxLb6JfE8eiANbXn5Rwxu8G'],  # Folder ID trên Drive
    }
    media = MediaIoBaseUpload(file, mimetype='image/png')
    uploaded_file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return uploaded_file.get('id')

def write_to_sheet(gc, prediction, image_drive_id):
    sh = gc.open_by_key("1yfnttAYT93SipMKfHW6tGoRBMQYTz7wwZ970cQp7HiE")
    worksheet = sh.sheet1
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

# === MODEL LOADING ===
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

# === PREPROCESSING ===
def classify_preprop(image_file): 
    image = Image.open(BytesIO(image_file)).convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)

def segment_preprop(image_file):
    image = Image.open(BytesIO(image_file)).convert('RGB')
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = img_to_array(image)
    return np.expand_dims(image, axis=0)

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

# === UI ===
app_mode = st.sidebar.selectbox('Chọn trang', ['Thông tin chung', 'Thống kê về dữ liệu huấn luyện', 'Ứng dụng chẩn đoán'])

if app_mode == 'Thông tin chung':
    st.title('Giới thiệu về thành viên')
    st.markdown('<p class="big-font"> Học sinh thực hiện </p>', unsafe_allow_html=True)
    st.markdown('<p class="name"> Lê Vũ Anh Tin - 10TH </p>', unsafe_allow_html=True)
    st.image(Image.open('Tin.jpg'))
    st.markdown('<p class="big-font"> Trường học tham gia cuộc thi </p>', unsafe_allow_html=True)
    st.markdown('<p class="name"> Trường THPT chuyên Nguyễn Du </p>', unsafe_allow_html=True)
    st.image(Image.open('school.jpg'))

elif app_mode == 'Thống kê về dữ liệu huấn luyện':
    st.title('Thống kê tổng quan về tập dữ liệu')
    st.caption('Tập dữ liệu ảnh siêu âm vú từ bệnh viện Baheya, Cairo, Ai Cập...')
    st.caption('Nguồn dữ liệu: [Kaggle Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data)')

elif app_mode == 'Ứng dụng chẩn đoán':
    st.title('Ứng dụng chẩn đoán bệnh ung thư vú dựa trên ảnh siêu âm vú')

    classifier, segmentor = load_model()
    file = st.file_uploader("Vui lòng tải ảnh siêu âm vú (jpg, png):", type=["jpg", "png"])

    if file is not None:
        slot = st.empty()
        slot.text('Hệ thống đang xử lý ảnh...')

        # Kết nối Google Drive và Sheets
        gc, drive_service = connect_google_services()

        # Lưu ảnh lên Drive
        file.seek(0)
        image_drive_id = upload_to_drive(file, drive_service)

        # Reset và chạy mô hình
        file.seek(0)
        classify_output, segment_output = preprocessing_uploader(file, classifier, segmentor)

        # Ghi dữ liệu vào Sheets
        write_to_sheet(gc, classify_output, image_drive_id)

        # Hiển thị kết quả
        test_image = Image.open(file)
        st.image(test_image, caption="Ảnh đầu vào", width=400)
        st.image(segment_output, caption="Ảnh vùng khối u", width=400)

        class_names = ['benign', 'malignant', 'normal']
        result_name = class_names[np.argmax(classify_output)]

        if result_name == 'benign':
            st.error('Chẩn đoán: **Khối u lành tính.**')
        elif result_name == 'malignant':
            st.warning('Chẩn đoán: **Bệnh nhân mắc ung thư vú.**')
        else:
            st.success('Chẩn đoán: **Không phát hiện khối u.**')

        slot.success('Chẩn đoán hoàn tất!')

        # Biểu đồ xác suất
        bar_frame = pd.DataFrame({
            'Xác suất dự đoán': [classify_output[0,0]*100, classify_output[0,1]*100, classify_output[0,2]*100],
            'Loại chẩn đoán': ["Lành tính", "Ác tính", "Bình thường"]
        })
        bar_chart = alt.Chart(bar_frame).mark_bar().encode(y='Xác suất dự đoán', x='Loại chẩn đoán')
        st.altair_chart(bar_chart, use_container_width=True)

        # Hiển thị xác suất
        st.write('- **Lành tính**: *{}%*'.format(round(classify_output[0,0]*100, 2)))
        st.write('- **Ác tính**: *{}%*'.format(round(classify_output[0,1]*100, 2)))
        st.write('- **Bình thường**: *{}%*'.format(round(classify_output[0,2]*100, 2)))
    else:
        st.text('Vui lòng tải lên ảnh để chẩn đoán.')
