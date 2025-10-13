# ==========================================
# 🩻 ỨNG DỤNG PHÂN ĐOẠN KHỐI U VÚ TỪ ẢNH SIÊU ÂM
# ==========================================
import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import keras

# ==========================================
# 1️⃣ Cấu hình chung
# ==========================================
st.set_page_config(page_title="Phân đoạn khối u vú", layout="centered")
st.title("🩺 ỨNG DỤNG PHÂN ĐOẠN KHỐI U VÚ TỪ ẢNH SIÊU ÂM")
st.markdown("""
Ứng dụng này sử dụng mô hình **U-Net** được huấn luyện trên tập dữ liệu siêu âm vú (BUSI, BUS-UCLM, BrEaST).  
Mô hình giúp **phát hiện và tô vùng nghi ngờ tổn thương** trong ảnh siêu âm vú.
""")

# ==========================================
# 2️⃣ Tải mô hình phân đoạn
# ==========================================
seg_model_path = "Seg_model.h5"
seg_model_id = "1PC4ZNJJB5n-JKSc1mmyOLeQ5tClx4hcP"  # Google Drive ID

if not os.path.exists(seg_model_path):
    st.write("⏬ Đang tải mô hình phân đoạn từ Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={seg_model_id}", seg_model_path, quiet=False)

# ==========================================
# 3️⃣ Định nghĩa hàm dice_loss và load model
# ==========================================
def dice_loss(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    return 1 - (2 * intersection + 1e-6) / (union + 1e-6)

# ⚙️ Cho phép load Lambda layer trong model .h5
keras.config.enable_unsafe_deserialization()

@st.cache_resource
def load_seg_model():
    model = tf.keras.models.load_model(
        seg_model_path,
        custom_objects={'dice_loss': dice_loss},
        safe_mode=False
    )
    return model

segmentor = load_seg_model()
st.success("✅ Mô hình phân đoạn đã sẵn sàng!")

# ==========================================
# 4️⃣ Hàm tiền xử lý & hậu xử lý
# ==========================================
def preprocess_image(image_file):
    """Chuyển ảnh đầu vào về dạng chuẩn cho mô hình"""
    segmentInputShape = (256, 256)
    image = Image.open(BytesIO(image_file)).convert('RGB')
    image = image.resize(segmentInputShape)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_mask(mask, threshold=0.5):
    """Nhị phân hóa mask đầu ra"""
    mask = np.squeeze(mask)
    mask = (mask > threshold).astype(np.uint8)
    return mask

def overlay_mask(image, mask):
    """Chồng mask đỏ lên ảnh gốc"""
    image = np.array(image).astype(np.uint8)
    mask_rgb = np.zeros_like(image)
    mask_rgb[..., 0] = mask * 255  # tô màu đỏ cho vùng tổn thương
    overlay = cv2.addWeighted(image, 0.7, mask_rgb, 0.5, 0)
    return overlay

# ==========================================
# 5️⃣ Giao diện chính
# ==========================================
file = st.file_uploader("📤 Vui lòng tải lên ảnh siêu âm (JPG/PNG):", type=["jpg", "png"])

if file:
    # Hiển thị ảnh gốc
    image_bytes = file.read()
    image_orig = Image.open(BytesIO(image_bytes)).convert("RGB")
    st.image(image_orig, caption="Ảnh siêu âm gốc", width=400)

    with st.spinner("🧠 Mô hình đang phân tích ảnh..."):
        # Tiền xử lý và dự đoán
        img_input = preprocess_image(image_bytes)
        mask_pred = segmentor.predict(img_input)[0]
        mask_post = postprocess_mask(mask_pred)

        # Overlay vùng tổn thương
        image_resized = image_orig.resize((256, 256))
        overlay = overlay_mask(image_resized, mask_post)

    # Hiển thị kết quả
    st.subheader("📊 Kết quả phân đoạn:")
    st.image(mask_post * 255, caption="🩸 Mask vùng tổn thương", width=400)
    st.image(overlay, caption="📍 Ảnh có vùng tổn thương được tô đỏ", width=400)

    # Thống kê vùng tổn thương
    lesion_pixels = np.sum(mask_post)
    total_pixels = mask_post.size
    lesion_ratio = (lesion_pixels / total_pixels) * 100

    st.write(f"📏 **Số điểm ảnh tổn thương:** {lesion_pixels:,}")
    st.write(f"📐 **Tỷ lệ vùng tổn thương so với toàn ảnh:** {lesion_ratio:.2f}%")

else:
    st.info("⬆️ Hãy tải lên ảnh siêu âm để mô hình bắt đầu phân tích.")
