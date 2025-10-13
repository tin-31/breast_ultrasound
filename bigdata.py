import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import altair as alt

# ==========================================
# 1️⃣ Tải mô hình phân đoạn (Seg_model.h5)
# ==========================================
seg_model_path = "Seg_model.h5"
seg_model_id = "1PC4ZNJJB5n-JKSc1mmyOLeQ5tClx4hcP"  # Google Drive ID

if not os.path.exists(seg_model_path):
    st.write("⏬ Đang tải mô hình phân đoạn từ Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={seg_model_id}", seg_model_path, quiet=False)

# ==========================================
# 2️⃣ Load model
# ==========================================
def dice_loss(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    return 1 - (2 * intersection + 1e-6) / (union + 1e-6)

@st.cache_resource
def load_seg_model():
    return tf.keras.models.load_model(seg_model_path, custom_objects={'dice_loss': dice_loss})

segmentor = load_seg_model()
st.success("✅ Mô hình phân đoạn đã sẵn sàng!")

# ==========================================
# 3️⃣ Hàm tiền xử lý và hậu xử lý
# ==========================================
def preprocess_image(image_file):
    segmentInputShape = (256, 256)
    image = Image.open(BytesIO(image_file)).convert('RGB')
    image = image.resize(segmentInputShape)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_mask(mask, threshold=0.5):
    mask = np.squeeze(mask)
    mask = (mask > threshold).astype(np.uint8)  # nhị phân hoá
    return mask

def overlay_mask(image, mask):
    """Chồng mask màu đỏ lên ảnh gốc"""
    image = np.array(image).astype(np.uint8)
    mask_rgb = np.zeros_like(image)
    mask_rgb[..., 0] = mask * 255  # red
    overlay = cv2.addWeighted(image, 0.7, mask_rgb, 0.5, 0)
    return overlay

# ==========================================
# 4️⃣ Giao diện chính
# ==========================================
st.title("🩻 Ứng dụng phân đoạn khối u vú từ ảnh siêu âm")
st.markdown("""
Ứng dụng này sử dụng mô hình học sâu U-Net được huấn luyện trên tập dữ liệu siêu âm vú (BUSI, BUS-UCLM, BrEaST).  
Mô hình giúp **phát hiện và tô vùng nghi ngờ tổn thương** trong ảnh siêu âm.
""")

file = st.file_uploader("📤 Vui lòng tải lên ảnh siêu âm (định dạng JPG/PNG):", type=["jpg", "png"])

if file:
    image_bytes = file.read()
    st.image(Image.open(BytesIO(image_bytes)), caption="Ảnh gốc", width=400)

    with st.spinner("🧠 Mô hình đang phân tích ảnh..."):
        img_input = preprocess_image(image_bytes)
        mask_pred = segmentor.predict(img_input)[0]
        mask_post = postprocess_mask(mask_pred)
        
        # Overlay mask lên ảnh gốc
        image_orig = Image.open(BytesIO(image_bytes)).convert("RGB").resize((256, 256))
        overlay = np.array(image_orig).copy()
        overlay[mask_post == 1] = [255, 0, 0]  # vùng tổn thương màu đỏ

    st.subheader("🩸 Kết quả phân đoạn:")
    st.image(mask_post * 255, caption="Mask (vùng tổn thương)", width=400)
    st.image(overlay, caption="Ảnh với vùng tổn thương được tô đỏ", width=400)

else:
    st.info("Vui lòng tải lên ảnh siêu âm để bắt đầu phân tích.")
