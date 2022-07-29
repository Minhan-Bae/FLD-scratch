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

pretrained_model = "/data/komedi/komedi/logs/2022-07-29/xception_10_22/model_logs/10_22_flmk_500.pt"

uploaded_file = st.sidebar.file_uploader(label='파일 업로드', type=['png', 'jpg'])

rotation_check = st.sidebar.checkbox(label="자동 안면 회전", value=True)
show_in_input = st.sidebar.checkbox(label="원본 이미지에서 보기", value=False)
show_annotate = st.sidebar.checkbox(label="인덱스 보기", value=False)
box_color = st.sidebar.color_picker(label="박스 색상", value='#0000FF')
point_color = st.sidebar.color_picker(label="포인트 색상", value='#F70101')
point_scale = st.sidebar.slider(label="포인트크기", min_value=1, max_value=30, value=10)
# st.sidebar.write(f"학습 모델 버전: {pretrained_model.split('/')[-1]}")

if uploaded_file:
    image_name = uploaded_file.name

    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    st.write(f"__입력 이미지 사이즈__ : {image.width} x {image.height}")

    col1, col2 = st.columns(2)
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
        # st.image(cropped_img)
        # _ = cropped_img.thumbnail((256, 256))
        np_image = np.asarray(cropped_img).astype('uint8')

        cropped_img_byte = io.BytesIO()
        cropped_img.save(cropped_img_byte, format='PNG')
        cropped_img_byte = cropped_img_byte.getvalue()
        # files = ['files', (uploaded_file.name, cropped_img_byte, uploaded_file.type)]

        start_time = time.time()
        pil_image, landmarks = run_detect(cropped_img_byte,
                                        pretrained=pretrained_model,
                                        rotation=rotation_check
                                        )
        
        landmark_result = []
        for x,y in landmarks:
            landmark_result.append([x*w, y*h])
    with col2:
        st.image(cropped_img)

    fig1 = plt.figure(figsize=(11,22))
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
            plt.imshow(image)
        else:
            plt.scatter(x,y,s=point_scale,c=point_color,marker='X')
            if show_annotate:
                plt.annotate(idx, (x,y))
            plt.imshow(cropped_img)

    st.success("얼굴 탐지 완료")
    st.pyplot(fig = fig1,
                clear_figure=True)

    st.write(f"경과시간: {time.time()-start_time:2.2f} 초")

    save = datetime.now()
    time = f"{str(save.day).zfill(2)}_{str(save.hour).zfill(2)}:{str(save.minute).zfill(2)}:{str(save.second).zfill(2)}"

    save_dir = f"/data/komedi/streamlit_logs"
    save_raw_image_dir = os.path.join(save_dir, "raw_image")
    save_crop_image_dir = os.path.join(save_dir, "crop_image")
    save_result_image_dir = os.path.join(save_dir, "result")
    if st.sidebar.button("결과를 확인하시고, 버튼을 눌러주세요."):
        con = st.container()
        os.makedirs(save_raw_image_dir, exist_ok=True)
        os.makedirs(save_crop_image_dir, exist_ok=True)
        os.makedirs(save_result_image_dir, exist_ok=True)
        
        fig1 = plt.figure(figsize=(11,22))
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
                plt.imshow(image)
            else:
                plt.scatter(x,y,s=point_scale,c=point_color,marker='X')
                if show_annotate:
                    plt.annotate(idx, (x,y))
                plt.imshow(cropped_img)

        plt.savefig(f"{save_result_image_dir}/{Path(image_name).stem}.jpg",bbox_inches='tight', pad_inches=0)

        image = image.convert("RGB")
        image.save(f"{save_raw_image_dir}/{Path(image_name).stem}.jpg")
        cropped_img.save(f"{save_crop_image_dir}/{Path(image_name).stem}.jpg")
        
        csv_lists = pd.read_csv("/data/komedi/streamlit_logs/result.csv",header=None).values.tolist()
        
        csv_list = []
        csv_list.append(f"{Path(image_name).stem}.jpg")
        csv_list.append("streamlit")
        csv_list.append(f"{save_raw_image_dir}/{Path(image_name).stem}.jpg")
        # csv_list.append(f"{save_crop_image_dir}/{Path(image_name).stem}.jpg")
        csv_list.append("")
        csv_list.append("")
        csv_list.append("")
        csv_list.append("")                        
        csv_list.append(time)
        csv_list.append("")
        csv_list.append("")        
                
        for landmark in landmark_result:
            x,y = landmark
            csv_list.append((x+l,y+t))
        csv_lists.append(csv_list)
        df = pd.DataFrame(csv_lists)
        
        df.to_csv(f"{save_dir}/result.csv", index=None, header=None)
        st.sidebar.write("감사합니다! :sunglasses:")