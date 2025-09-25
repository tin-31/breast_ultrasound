import os
import gdown

# Model phân đoạn
seg_model_path = "Seg_model_new.h5"
seg_model_id = "1uwD98MxgE0OW3AeP6wzpmupXRJrh5s4G"
if not os.path.exists(seg_model_path):
    gdown.download(f"https://drive.google.com/uc?id={seg_model_id}", seg_model_path, quiet=False)

# Model phân loại
clf_model_path = "Classifier_model_2_new.h5"
clf_model_id = "19obBuZvcg5YSXbaTE4dRhPMHmZhkq3Ss"
if not os.path.exists(clf_model_path):
    gdown.download(f"https://drive.google.com/uc?id={clf_model_id}", clf_model_path, quiet=False)

# ... các import khác và code Streamlit của bạn ...
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
#import keras
#import cv2 
from PIL import Image, ImageOps
#bug reason - the preproess function used in inference is not same with training preprocess function
# from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import load_img
from io import BytesIO
your_path = r""
# st.set_option('deprecation.showfileUploaderEncoding', False)
# @st.cache(allow_output_mutation=True)
def load_model():
    def dice_loss(y_true, y_pred):
        # Flatten the predictions and ground truth
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        # Compute the intersection and union
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)

        # Compute the Dice loss
        dice_loss = 1 - 2 * intersection / union

        return dice_loss
    
    classifier = tf.keras.models.load_model('Classifier_model_2_new.h5')
    segmentor = tf.keras.models.load_model('Seg_model_new.h5', custom_objects={'dice_loss': dice_loss})
    return classifier, segmentor


def predict_class(image, model):
# 	image = tf.cast(image, tf.float32)
	image = np.resize(image, (224,224))
# 	image_1 = image
	image = np.dstack((image,image,image))
# 	image_2 = image
# 	cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	image = np.expand_dims(image, axis = 0)
# 	image_3 = image   


	prediction = model.predict(image)

	return prediction

def classify_preprop(image_file): 
    classifyInputShape = (224, 224)
    image = Image.open(BytesIO(image_file))
    image = image.convert("RGB")
    image = image.resize(classifyInputShape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def segment_preprop(image_file):
    segmentInputShape = (256, 256)
    image = Image.open(BytesIO(image_file)).convert('RGB')
    image = image.resize(segmentInputShape)
    image = np.array(image)
    # Normalize 
    image = image / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def segment_postprop(image, mask):   
    #Apply mask to image then return the masked image
    image = np.squeeze(image)
    mask = np.squeeze(mask)
    #Add channel dimension
    mask = np.expand_dims(mask, axis=2)
    
    image = image * mask
    return image

def preprocessing_uploader(file, classifier, segmentor):
    image_file = file.read()

    #Two type of image for 2 model
    #Image for classifier
    image_to_classify = classify_preprop(image_file)
    
    #Image to segment
    image_to_segment = segment_preprop(image_file)
    #Infer
    classify_output = classifier.predict(image_to_classify)
    segment_output = segmentor.predict(image_to_segment)[0]
    segment_output = segment_postprop(image_to_segment, segment_output)
    return classify_output,segment_output
app_mode = st.sidebar.selectbox('Chọn trang',['Ứng dụng chẩn đoán', 'Thông tin chung','Thống kê về dữ liệu huấn luyện']) #two pages
if app_mode=='Thông tin chung':
    st.title('Giới thiệu về thành viên')
    st.markdown("""
    <style>
    .big-font {
    font-size:35px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .name {
    font-size:25px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font"> Học sinh thực hiện </p>', unsafe_allow_html=True)
    st.markdown('<p class="name"> Lê Vũ Anh Tin - 11TH </p>', unsafe_allow_html=True)
    tin_ava = Image.open('Tin.jpg')
    st.image(tin_ava)
    st.markdown('<p class="big-font"> Trường học tham gia cuộc thi </p>', unsafe_allow_html=True)
    st.markdown('<p class="name"> Trường THPT chuyên Nguyễn Du </p>', unsafe_allow_html=True)
    school_ava = Image.open('school.jpg')
    st.image(school_ava)
    
elif app_mode=='Thống kê về dữ liệu huấn luyện': 
    st.title('Thống kê tổng quan về tập dữ liệu')
    
    st.markdown("""
    <style>
    .big-font {
    font-size:30px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.caption('Trong nghiên cứu này, tập dữ liệu Breast Ultrasound Images (BUI) được xây dựng bằng cách kết hợp từ hai nguồn dữ liệu công khai trên nền tảng Kaggle, bao gồm: Breast Ultrasound Images Dataset (BUSI) do Arya Shah cung cấp , và BUS-UCLM Breast Ultrasound Dataset do Orvile cung cấp . Tập dữ liệu tổng hợp này bao gồm 1578 ảnh siêu âm vú đã được đánh giá và chú thích, với các mặt nạ phân đoạn tương ứng.')
    st.caption('Nội dung nghiên cứu khoa học và ứng dụng của nhóm được thiết kế dựa trên việc huấn luyện nhóm dữ liệu Breast Ultrasound Images Dataset. Dữ liệu đã được tiền xử lý và thay đổi kích thước về 256 x 256. Thông tin chi tiết của tập dữ liệu có thể tìm được ở dưới đây: ')
    st.caption('*"https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data"*')
    st.caption('*"https://www.kaggle.com/datasets/orvile/bus-uclm-breast-ultrasound-dataset"*')
    
elif app_mode=='Ứng dụng chẩn đoán':
    classifier, segmentor = load_model()
    st.title('Ứng dụng chẩn đoán bệnh ung thư vú dựa trên ảnh siêu âm vú')

    files = st.file_uploader("Bạn vui lòng nhập ảnh siêu âm vú", type=["jpg", "png"],
                             accept_multiple_files=True)

    if not files:
        st.info('Đang chờ tải lên...')
    else:
        class_names = ['benign', 'malignant', 'normal']
        rows = []  # để kết xuất CSV

        for file in files:
            slot = st.empty()
            slot.text(f'Đang chẩn đoán: {file.name} ...')

            # Suy luận
            classify_output, segment_prob, orig_256 = preprocessing_uploader(file, classifier, segmentor)
            probs = classify_output[0]  # (3,)
            pred_idx = int(np.argmax(probs))
            pred_name = class_names[pred_idx]
            confidence = float(np.max(probs))  # độ tự tin = max softmax

            # Overlay mask
            overlay = make_overlay(orig_256, segment_prob, threshold=th, alpha=alpha)

            # Hiển thị
            col1, col2 = st.columns(2)
            with col1:
                st.image((orig_256*255).astype(np.uint8), caption=f"Ảnh chuẩn hóa 256×256: {file.name}", width=400)
                st.image(overlay, caption=f"Overlay mask (th={th:.2f}, α={alpha:.2f})", width=400)
            with col2:
                st.markdown(f"**Kết luận phân loại:** `{pred_name}`")
                st.markdown(f"**Độ tự tin (max softmax):** `{confidence:.3f}`")
                if confidence < low_conf_bar:
                    st.warning(f"Độ tự tin thấp (< {low_conf_bar:.2f}). Vui lòng thận trọng khi diễn giải.")
                else:
                    st.success("Độ tự tin đạt yêu cầu.")

                # Biểu đồ cột xác suất
                bar_frame = pd.DataFrame({
                    'Loại chẩn đoán': ["Lành tính", "Ác tính", "Bình thường"],
                    'Xác suất dự đoán': [probs[0]*100, probs[1]*100, probs[2]*100]
                })
                bar_chart = alt.Chart(bar_frame).mark_bar().encode(
                    y='Xác suất dự đoán', x='Loại chẩn đoán'
                )
                st.altair_chart(bar_chart, use_container_width=True)

                st.write('- **Lành tính**: *{}%*'.format(round(probs[0]*100,2)))
                st.write('- **Ác tính**: *{}%*'.format(round(probs[1]*100,2)))
                st.write('- **Bình thường**: *{}%*'.format(round(probs[2]*100,2)))

            slot.success('Hoàn tất!')

            # Lưu dòng kết quả cho CSV
            rows.append({
                "filename": file.name,
                "cls_pred": pred_name,
                "prob_benign": probs[0],
                "prob_malignant": probs[1],
                "prob_normal": probs[2],
                "uncertainty(1-maxprob)": 1.0 - confidence,
                "confidence_maxprob": confidence
            })

        # Xuất CSV
        df = pd.DataFrame(rows)
        st.subheader("Kết quả tổng hợp")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("Tải CSV kết quả", csv, file_name="predictions.csv", mime="text/csv")
