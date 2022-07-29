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
from utils.index import *
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="[Demo] 코메디클럽 안면 랜드마크 탐지", page_icon="😀", layout="wide")

st.header("😀 [Demo] 코메디클럽 안면 랜드마크 탐지")

pretrained_model = "/data/komedi/komedi/logs/2022-07-29/xception_13_16/13_16_best.pt"

uploaded_file = st.sidebar.file_uploader(label='파일 업로드', type=['png', 'jpg'])

rotation_check = st.sidebar.checkbox(label="자동 안면 회전 기능(30도 이상)", value=True)
show_in_input = st.sidebar.checkbox(label="원본 이미지에서 보기", value=False)
show_annotate = st.sidebar.checkbox(label="인덱스 보기", value=True)
box_color = st.sidebar.color_picker(label="박스 색상", value='#0000FF')
point_color = st.sidebar.color_picker(label="포인트 색상", value='#F70101')
point_scale = st.sidebar.slider(label="포인트크기", min_value=1, max_value=30, value=10)

if uploaded_file:
    image_name = uploaded_file.name

    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image.resize([2*image.width,2*image.height])
    st.write(f"__입력 이미지 사이즈__ : {image.width} x {image.height}")
    
    tab1, tab2 = st.tabs(["입력 이미지 및 안면 탐지", "안면 랜드마크 탐지 결과"])
    
    with tab1:
        col1, col2 = st.columns([4,1])
        with col1:
            rect = st_cropper(
                image,
                realtime_update=True,
                box_color=box_color,
                aspect_ratio=None,
                box_algorithm=return_box,
                return_type="box"
            )
            
            l, t, w, h = tuple(map(int, rect.values()))
            
            cropped_img = image.crop(
                (l,t,l+w,t+h)
            )

            np_image = np.asarray(cropped_img).astype('uint8')

            cropped_img_byte = io.BytesIO()
            cropped_img.save(cropped_img_byte, format='PNG')
            cropped_img_byte = cropped_img_byte.getvalue()
            
        with col2:
            start_time = time.time()
            image, pil_image, landmarks = run_detect(image,
                                            cropped_img_byte,
                                            pretrained=pretrained_model,
                                            rotation=rotation_check
                                            )
            st.image(pil_image)

        landmark_result = []
        for idx in range(len(landmarks)):
            x,y = landmarks[idx]
            landmark_result.append([landmark_index[idx], x*w, y*h])
            
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        st.success("안면 랜드마크 탐지 완료! :sunglasses:")
        
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = plt.figure(figsize=(10,10))
            plt.axis("off")
            for idx in range(len(landmarks)):
                x,y = landmarks[idx][0]*w, landmarks[idx][1]*h

                if show_in_input:
                    plt.scatter(x+l,y+t,s=point_scale,c=point_color,marker='X')
                    if show_annotate:
                        plt.annotate(idx, (x+l,y+t))
                    plt.imshow(image)
                else:
                    plt.scatter(x,y,s=point_scale,c=point_color,marker='X')
                    if show_annotate:
                        plt.annotate(idx, (x,y))
                    plt.imshow(pil_image)

            st.pyplot(fig = fig1)
            st.write(f"경과시간: {time.time()-start_time:2.2f} 초")
            
        with col2:
            df = pd.DataFrame(landmark_result, columns=["name","x","y"])
            st.write(df)

    save = datetime.now()
    

   
    

    if st.sidebar.button("결과를 확인하시고, 버튼을 눌러주세요."):
        con = st.container()
        
        save_dir = f"/data/komedi/streamlit_logs"
        save_raw_image_dir = os.path.join(save_dir, "raw_image")
        save_crop_image_dir = os.path.join(save_dir, "crop_image")
        save_result_image_dir = os.path.join(save_dir, "result")
        
        os.makedirs(save_raw_image_dir, exist_ok=True)
        os.makedirs(save_crop_image_dir, exist_ok=True)
        os.makedirs(save_result_image_dir, exist_ok=True)
        
        fig1 = plt.figure(figsize=(11,33))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.imshow(image)

        plt.subplot(1,2,2)
        plt.axis("off")
        for idx in range(len(landmarks)):
            x,y = landmarks[idx][0]*w, landmarks[idx][1]*h
            if show_in_input:
                plt.scatter(x+l,y+t,s=point_scale,c=point_color,marker='X')
                if show_annotate:
                    plt.annotate(idx, (x+l,y+t))
                plt.imshow(image.rotate())
            else:
                plt.scatter(x,y,s=point_scale,c=point_color,marker='X')
                if show_annotate:
                    plt.annotate(idx, (x,y))
                plt.imshow(pil_image)

        plt.savefig(f"{save_result_image_dir}/{Path(image_name).stem}.jpg",bbox_inches='tight', pad_inches=0)

        image = image.convert("RGB")
        image.save(f"{save_raw_image_dir}/{Path(image_name).stem}.jpg")
        pil_image.save(f"{save_crop_image_dir}/{Path(image_name).stem}.jpg")
        
        times = f"{str(save.day).zfill(2)}_{str(save.hour).zfill(2)}:{str(save.minute).zfill(2)}:{str(save.second).zfill(2)}"
        
        csv_lists = pd.read_csv("/data/komedi/streamlit_logs/result.csv",header=None).values.tolist()
        
        csv_list = []
        csv_list.append(f"{Path(image_name).stem}.jpg")
        csv_list.append("streamlit")
        csv_list.append(f"{save_raw_image_dir}/{Path(image_name).stem}.jpg")
        csv_list.append("")
        csv_list.append("")
        csv_list.append("")
        csv_list.append("")                        
        csv_list.append(times)
        csv_list.append("")
        csv_list.append("")        
                
        for landmark in landmark_result:
            _, x,y = landmark
            csv_list.append((round(x+l,4), round(y+t,4)))
        csv_lists.append(csv_list)
        df = pd.DataFrame(csv_lists)
        
        df.to_csv(f"{save_dir}/result.csv", index=None, header=None)
        st.sidebar.write("감사합니다! :sunglasses:")