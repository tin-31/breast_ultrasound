import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gspread
from google.oauth2 import service_account
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload

# Load model
model = tf.keras.models.load_model("ultrasound_model.keras")

# Hàm tiền xử lý ảnh
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Giao diện upload ảnh
st.title("Breast Ultrasound Classification")
file = st.file_uploader("Upload ảnh siêu âm", type=["jpg", "jpeg", "png"])

if file is not None:
    # Hiển thị ảnh
    image = Image.open(file)
    st.image(image, caption="Ảnh đã upload", use_column_width=True)

    # Tiền xử lý và phân loại
    preprocessed_image = preprocess_image(image)
    classify_output = model.predict(preprocessed_image)

    # Lấy tên class
    class_names = ['Benign', 'Malignant', 'Normal']
    result_index = np.argmax(classify_output)
    result_name = class_names[result_index]
    st.subheader("Kết quả phân loại:")
    st.success(f"{result_name}")

    # === Lưu kết quả lên Google Drive + Sheets ===

    # Load credentials
    creds = service_account.Credentials.from_service_account_file(
        "breast_ultrasound_service_account.json",
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )

    # === Google Sheet ===
    gc = gspread.authorize(creds)
    sh = gc.open_by_key("1yfnttAYT93SipMKfHW6tGoRBMQYTz7wwZ970cQp7HiE")  # <-- Sheet ID
    worksheet = sh.sheet1

    # === Google Drive ===
    drive_service = build('drive', 'v3', credentials=creds)
    file_bytes = file.getvalue()
    file_metadata = {
        'name': f"ultrasound_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        'parents': ['1MNgXaIZsWuxLb6JfE8eiANbXn5Rwxu8G']  # <-- Folder ID
    }
    media = MediaInMemoryUpload(file_bytes, mimetype='image/png')

    uploaded_file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    file_id = uploaded_file.get('id')
    file_url = f"https://drive.google.com/uc?id={file_id}"

    # === Ghi log vào Sheet ===
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    worksheet.append_row([
        timestamp,
        result_name,
        round(classify_output[0, 0]*100, 2),
        round(classify_output[0, 1]*100, 2),
        round(classify_output[0, 2]*100, 2),
        file_url
    ])

    st.info("Đã lưu kết quả lên Google Drive & Google Sheets.")
