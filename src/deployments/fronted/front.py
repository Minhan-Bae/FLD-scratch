
import io
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_cropper import st_cropper

from utils.face_detector import return_box
from landmark_detector import run_detect

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="[Demo] 코메디클럽 안면 랜드마크 탐지", page_icon="😀", layout="centered")

st.header("😀 [Demo] 코메디클럽 안면 랜드마크 탐지")
uploaded_file = st.sidebar.file_uploader(label='파일 업로드', type=['png', 'jpg'])

if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))

    cropped_img = st_cropper(
        image,
        realtime_update=False,
        aspect_ratio=None,
        box_algorithm=return_box,
        return_type="image",
    )
        
    _ = cropped_img.thumbnail((512, 512))
    np_image = np.asarray(cropped_img).astype('uint8')
    height, width, _ = np_image.shape

    cropped_img_byte = io.BytesIO()
    cropped_img.save(cropped_img_byte, format='PNG')
    cropped_img_byte = cropped_img_byte.getvalue()
    files = ['files', (uploaded_file.name, cropped_img_byte, uploaded_file.type)]

    st.write("Face Detecting..")
    pil_image, landmarks = run_detect(cropped_img_byte)
    
    landmark_result = []
    for x,y in landmarks:
        landmark_result.append([x*width, y*height])

    # Draw landmark into crop image
    draw = ImageDraw.Draw(pil_image)
    for x,y in landmark_result:
        draw.ellipse((x, y, x+5, y+5), fill = 'red', outline ='red')
    
    st.image(pil_image)