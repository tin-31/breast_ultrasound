# app.py
import os
import io
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from sklearn.utils import compute_class_weight

# ===== tải utils
from download_utils import safe_gdown

# ---------------- Config IDs (THAY BẰNG ID THẬT CỦA BẠN) ----------------
SEG_ID = "1uwD98MxgE0OW3AeP6wzpmupXRJrh5s4G"        # Seg_model_new.h5
CLF_ID = "19obBuZvcg5YSXbaTE4dRhPMHmZhkq3Ss"       # Classifier_model_2_new.h5
SEG_PATH = "Seg_model_new.h5"
CLF_PATH = "Classifier_model_2_new.h5"

# ---------------- Download models (ổn định) ----------------
try:
    safe_gdown(SEG_ID, SEG_PATH)
    safe_gdown(CLF_ID, CLF_PATH)
except Exception as e:
    st.error(f"Tải mô hình từ Google Drive thất bại: {e}")
    st.stop()

# ---------------- Custom metrics/loss (nếu cần cho seg) ----------------
def dice_loss(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersect = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return 1.0 - (2.0 * intersect + smooth) / (denom + smooth)

# ---------------- Cache models ----------------
@st.cache_resource(show_spinner=True)
def load_models():
    clf = tf.keras.models.load_model(CLF_PATH, compile=False)
    seg = tf.keras.models.load_model(SEG_PATH, compile=False,
                                     custom_objects={"dice_loss": dice_loss})
    return clf, seg

classifier, segmentor = load_models()

# ---------------- Preprocess helpers ----------------
def classify_preprop(image_bytes, target_hw=(224, 224)):
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    im = im.resize(target_hw)
    arr = img_to_array(im)            # (H,W,3) float32 0..255
    arr = np.expand_dims(arr, axis=0) # (1,H,W,3)
    arr = eff_preprocess(arr)         # đúng chuẩn EfficientNet ([-1..1] hoặc chuẩn riêng)
    return arr

def segment_preprop(image_bytes, target_hw=(256, 256)):
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    im = im.resize(target_hw)
    arr = np.array(im).astype("float32") / 255.0  # 0..1
    arr4d = np.expand_dims(arr, axis=0)           # (1,H,W,3)
    return arr4d, arr                             # trả thêm bản 2D/3D để hiển thị

def segment_postprop(image_2d_01, mask_prob_01):
    """
    image_2d_01: (H,W,3) 0..1
    mask_prob_01: (H,W) 0..1 (hoặc (H,W,1))
    Trả về ảnh đã nhân mask (H,W,3) 0..1
    """
    if mask_prob_01.ndim == 3 and mask_prob_01.shape[-1] == 1:
        mask_2d = mask_prob_01[..., 0]
    else:
        mask_2d = mask_prob_01
    mask_3 = np.repeat(mask_2d[..., None], 3, axis=-1)  # (H,W,3)
    return image_2d_01 * mask_3

def make_overlay(image_2d_01, mask_prob_01, threshold=0.5, alpha=0.4):
    """
    Overlay mask threshold lên ảnh gốc. Trả về uint8 (H,W,3).
    """
    if mask_prob_01.ndim == 3 and mask_prob_01.shape[-1] == 1:
        prob = mask_prob_01[..., 0]
    else:
        prob = mask_prob_01
    h, w = prob.shape
    base = (image_2d_01 * 255).astype(np.uint8)
    mask_bin = (prob >= threshold).astype(np.uint8)

    overlay = base.copy()
    color = np.zeros_like(base)
    color[..., 0] = 255  # tô đỏ vùng mask
    overlay = (1 - alpha) * overlay + alpha * (color * mask_bin[..., None])
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

def preprocessing_uploader(file, classifier, segmentor):
    # Đọc bytes
    image_bytes = file.read()

    # Chuẩn bị input cho classifier
    x_clf = classify_preprop(image_bytes, target_hw=(224, 224))
    # Chuẩn bị input cho segmentor và bản hiển thị
    x_seg4d, img256_01 = segment_preprop(image_bytes, target_hw=(256, 256))

    # Suy luận
    probs = classifier.predict(x_clf, verbose=0)[0]   # (3,)
    seg_prob = segmentor.predict(x_seg4d, verbose=0)[0]  # (H,W) hoặc (H,W,1) 0..1

    # Ảnh hiển thị 256×256 dạng 0..1
    return probs, seg_prob, img256_01

# ---------------- Sidebar / UI ----------------
st.sidebar.title("Tùy chọn hiển thị")
th = st.sidebar.slider("Ngưỡng mask (threshold)", 0.00, 1.00, 0.50, 0.01)
alpha = st.sidebar.slider("Độ đậm overlay (alpha)", 0.00, 1.00, 0.40, 0.01)
low_conf_bar = st.sidebar.slider("Cảnh báo nếu max-prob <", 0.00, 1.00, 0.60, 0.01)

app_mode = st.sidebar.selectbox('Chọn trang', [
    'Ứng dụng chẩn đoán', 'Thông tin chung', 'Thống kê về dữ liệu huấn luyện'
])

# ---------------- Pages ----------------
if app_mode == 'Thông tin chung':
    st.title('Giới thiệu')
    st.write("Mô tả nhóm, trường, nguồn dữ liệu …")
    st.caption('Nguồn dữ liệu: BUSI & BUS-UCLM (Kaggle).')

elif app_mode == 'Thống kê về dữ liệu huấn luyện':
    st.title('Thống kê dữ liệu (demo)')
    st.caption('Bạn có thể thay bằng thống kê thật của dataset…')

else:
    st.title('Ứng dụng chẩn đoán ung thư vú (siêu âm)')

    files = st.file_uploader("Tải ảnh siêu âm (JPG/PNG)", type=["jpg", "png"],
                             accept_multiple_files=True)

    if not files:
        st.info('Đang chờ tải lên…')
    else:
        class_names = ['benign', 'malignant', 'normal']
        rows = []

        for file in files:
            slot = st.empty()
            slot.text(f'Đang xử lý: {file.name} ...')

            probs, seg_prob, orig_256 = preprocessing_uploader(file, classifier, segmentor)
            pred_idx = int(np.argmax(probs))
            pred_name = class_names[pred_idx]
            confidence = float(np.max(probs))

            overlay = make_overlay(orig_256, seg_prob, threshold=th, alpha=alpha)

            col1, col2 = st.columns(2)
            with col1:
                st.image((orig_256*255).astype(np.uint8), caption=f"Ảnh 256×256: {file.name}", width=400)
                st.image(overlay, caption=f"Overlay mask (th={th:.2f}, α={alpha:.2f})", width=400)
            with col2:
                st.markdown(f"**Kết luận:** `{pred_name}`")
                st.markdown(f"**Độ tự tin:** `{confidence:.3f}`")
                if confidence < low_conf_bar:
                    st.warning(f"Độ tự tin thấp (< {low_conf_bar:.2f}). Cần thận trọng.")
                else:
                    st.success("Độ tự tin đạt yêu cầu.")

                # Biểu đồ xác suất
                frame = pd.DataFrame({
                    'Loại': ["Lành tính", "Ác tính", "Bình thường"],
                    'Xác suất (%)': [probs[0]*100, probs[1]*100, probs[2]*100]
                })
                chart = alt.Chart(frame).mark_bar().encode(
                    x='Loại', y='Xác suất (%)'
                )
                st.altair_chart(chart, use_container_width=True)

                st.write(f'- **benign**: {probs[0]*100:.2f}%')
                st.write(f'- **malignant**: {probs[1]*100:.2f}%')
                st.write(f'- **normal**: {probs[2]*100:.2f}%')

            slot.success('Xong!')

            rows.append({
                "filename": file.name,
                "pred_class": pred_name,
                "prob_benign": probs[0],
                "prob_malignant": probs[1],
                "prob_normal": probs[2],
                "uncertainty(1-maxprob)": 1.0 - confidence,
                "confidence_maxprob": confidence
            })

        df = pd.DataFrame(rows)
        st.subheader("Kết quả")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("Tải CSV", csv, file_name="predictions.csv", mime="text/csv")
