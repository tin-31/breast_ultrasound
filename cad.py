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
    image = np.array(image)
    image = cv2.resize(image, segmentInputShape)
    #Normalize 
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
app_mode = st.sidebar.selectbox('Chọn trang',['Thông tin chung','Ứng dụng chẩn đoán']) #two pages
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
    st.markdown('<p class="name"> Lê Vũ Anh Tin - 10TH </p>', unsafe_allow_html=True)
    tin_ava = Image.open('member/Tin.jpg')
    st.image(tin_ava)
    st.markdown('<p class="big-font"> Trường học tham gia cuộc thi KHKT-Khởi nghiệp </p>', unsafe_allow_html=True)
    st.markdown('<p class="name"> Trường THPT chuyên Nguyễn Du </p>', unsafe_allow_html=True)
    school_ava = Image.open('member/school.jpg')
    st.image(school_ava)
    
elif app_mode=='Ứng dụng chẩn đoán':
    classifier, segmentor = load_model()
    st.title('Ứng dụng chẩn đoán bệnh ung thư vú dựa trên ảnh siêu âm vú')

    file = st.file_uploader("Bạn vui lòng nhập ảnh siêu âm vú để phân loại ở đây", type=["jpg", "png"])
# 

    if file is None:
        st.text('Đang chờ tải lên....')

    else:
        slot = st.empty()
        slot.text('Hệ thống đang thực thi chẩn đoán....')
        
        classify_output, segment_output = preprocessing_uploader(file, classifier, segmentor)
        test_image = Image.open(file)
        st.image(test_image, caption="Ảnh đầu vào", width = 400)
        class_names = ['benign', 'malignant','normal']
        result_name = class_names[np.argmax(classify_output)]
        st.image(segment_output, caption="Ảnh khối u", width = 400)
        if str(result_name) == 'benign':
            statement = str('Chẩn đoán của mô hình học máy: **Bệnh nhân có khối u lành tính.**')
            st.error(statement)
        elif str(result_name) == 'malignant':
            statement = str('Chẩn đoán của mô hình học máy: **Bệnh nhân mắc ung thư vú.**')
            st.warning(statement)
        elif str(result_name) == 'normal':
            statement = str('Chẩn đoán của mô hình học máy: **Không có dấu hiệu khối u ở vú.**')
        slot.success('Hoàn tất!')

#         st.success(output)
     
        #Plot bar chart
        bar_frame = pd.DataFrame({'Xác suất dự đoán': [classify_output[0,0] *100, classify_output[0,1]*100, classify_output[0,2]*100], 
                                   'Loại chẩn đoán': ["Lành tính", "Ác tính", "Bình thường"]
                                 })
        bar_chart = alt.Chart(bar_frame).mark_bar().encode(y = 'Xác suất dự đoán', x = 'Loại chẩn đoán' )
        st.altair_chart(bar_chart, use_container_width = True)
        #Note
        st.write('- **Xác suất bệnh nhân có khối u lành tính là**: *{}%*'.format(round(classify_output[0,0] *100,2)))
        st.write('- **Xác suất bệnh nhân mắc ung thư vú là**: *{}%*'.format(round(classify_output[0,1] *100,2)))
        st.write('- **Xác suất bệnh nhân khỏe mạnh là**: *{}%*'.format(round(classify_output[0,2] *100,2)))
