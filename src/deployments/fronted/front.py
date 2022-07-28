import os
import io
import time
from datetime import datetime

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

from streamlit_cropper import st_cropper
from utils.face_detector import return_box
from landmark_detector import run_detect

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="[Demo] 코메디클럽 안면 랜드마크 탐지", page_icon="😀", layout="centered")

st.header("😀 [Demo] 코메디클럽 안면 랜드마크 탐지")

uploaded_file = st.file_uploader(label='파일 업로드', type=['png', 'jpg'])
st.write("얼굴이 잘 인식되었다면 박스 내부를 더블클릭 해주세요.")
if uploaded_file:
    image_name = uploaded_file.name

    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    
    cropped_img = st_cropper(
        image,
        realtime_update=False,
        aspect_ratio=None,
        box_algorithm=return_box,
        return_type="image",
    )
        
    _ = cropped_img.thumbnail((256, 256))
    np_image = np.asarray(cropped_img).astype('uint8')
    height, width, _ = np_image.shape

    cropped_img_byte = io.BytesIO()
    cropped_img.save(cropped_img_byte, format='PNG')
    cropped_img_byte = cropped_img_byte.getvalue()
    files = ['files', (uploaded_file.name, cropped_img_byte, uploaded_file.type)]

    start_time = time.time()
    pil_image, landmarks = run_detect(cropped_img_byte,
                                      pretrained="/data/komedi/komedi/logs/2022-07-28/xception_14_55/14_55_best.pt")
    
    landmark_result = []
    for x,y in landmarks:
        landmark_result.append([x*width, y*height])

    fig1 = plt.figure(figsize=(11,22))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.imshow(image)

    plt.subplot(1,2,2)
    plt.axis("off")
    for idx in range(len(landmarks)):
        plt.scatter(landmarks[idx][0]*width,landmarks[idx][1]*height,s=20,c='dodgerblue',marker='X')
        
        landmarks[idx][0] *= width
        landmarks[idx][1] *= height
        plt.annotate(idx, (landmarks[idx][0],landmarks[idx][1]))
        plt.imshow(pil_image,alpha=0.5)
    
    st.pyplot(fig1)
    st.success("얼굴 탐지 완료")
    st.write(f"경과시간: {time.time()-start_time:2.2f} 초")
    
    save = datetime.now()
    time = f"{str(save.day).zfill(2)}_{str(save.hour).zfill(2)}:{str(save.minute).zfill(2)}:{str(save.second).zfill(2)}"
    
    save_dir = f"/data/komedi/streamlit_logs"
    save_raw_image_dir = os.path.join(save_dir, "raw_image")
    save_crop_image_dir = os.path.join(save_dir, "crop_image")
    save_result_image_dir = os.path.join(save_dir, "result")
    if st.button("결과를 확인하시고, 버튼을 눌러주세요."):
        con = st.container()
        os.makedirs(save_raw_image_dir, exist_ok=True)
        os.makedirs(save_crop_image_dir, exist_ok=True)
        os.makedirs(save_result_image_dir, exist_ok=True)
        
        plt.savefig(f"{save_result_image_dir}/{Path(image_name).stem}.jpg",bbox_inches='tight', pad_inches=0)

        image = image.convert("RGB")
        image.save(f"{save_raw_image_dir}/{Path(image_name).stem}.jpg")
        cropped_img.save(f"{save_crop_image_dir}/{Path(image_name).stem}.jpg")
        
        csv_lists = pd.read_csv("/data/komedi/streamlit_logs/result.csv",header=None).values.tolist()
        
        csv_list = []
        csv_list.append(time)
        csv_list.append(f"{Path(image_name).stem}.jpg")
        csv_list.append(f"{save_raw_image_dir}/{time}.jpg")
        csv_list.append(f"{save_crop_image_dir}/{time}.jpg")
        
        for landmark in landmark_result:
            x,y = landmark
            csv_list.append((x,y))
        csv_lists.append(csv_list)
        df = pd.DataFrame(csv_lists)
        
        df.to_csv(f"{save_dir}/result.csv", index=None, header=None)