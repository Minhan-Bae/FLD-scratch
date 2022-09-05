import streamlit as st
from streamlit_cropper import st_cropper

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import io
import time
import pandas as pd
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import requests

from utils.fig2img import fig2img
from utils.face_detector import return_box
from utils.visualization import LANDMARK_IDX

HOST_IP = "localhost"
BACKEND_PORT = 8000

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="[Demo] 코메디클럽 안면 랜드마크 탐지", page_icon="random", layout="centered")

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(index=None).encode('utf-8')

def main():
    # 세션 정보 초기화 정의
    def reset_session():
        for key in SESSION_LIST:
            st.session_state[key]=None
    
    # 세션 리스트 정의
    SESSION_LIST = ["input_image",
                    "crop_image",
                    "landmark_deploy"]

    for key in SESSION_LIST:
        if key not in st.session_state:
            st.session_state[key] = None
    
    st.title("코메디클럽 안면 랜드마크 탐지", anchor="title")
    st.header("안면 탐지", anchor="face_detection")

    df = None
    image = None
    
    with st.form("face_detection_form"):
        uploaded_file = st.file_uploader(
            label="이미지 업로드",
            type=["png", "jpg"],
            help = "이마가 보이도록 앞머리를 정돈한 이미지 사용을 권장합니다.")  
        try:
            if uploaded_file:
                image_bytes = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(image_bytes))
                image = image.convert("RGB")

            submitted_facecrop = None
            rect = None
            submitted_facecrop = st.form_submit_button(label="안면 탐지를 수행합니다",
                                                        on_click=reset_session)

        except UnboundLocalError:
            pass
        
        if submitted_facecrop:
            try:
                response = None
                st.caption("얼굴 영역이 정확하지 않을 경우, 직접 영역을 설정할 수 있습니다.")
                rect = st_cropper(
                    image,
                    realtime_update=True,
                    box_color='#0000FF',
                    aspect_ratio=None,
                    box_algorithm=return_box,
                    return_type="box"
                )
                l, t, w, h = tuple(map(int, rect.values()))
                cropped_img = image.crop(
                    (l,t,l+w,t+h)
                )

                cropped_img_byte = io.BytesIO()
                cropped_img.save(cropped_img_byte, format='PNG')

                files = {"image":cropped_img_byte.getvalue()}
                
                start_time = time.time()
                response = requests.post(f"http://{HOST_IP}:{BACKEND_PORT}/prediction", files=files)
                end_time = time.time()

                response_json = response.json()
                if response.status_code != 200:
                    raise Exception()
                        
                label = response_json["label"]

                letency = end_time-start_time
                st.write(f"inference time : {letency:.2f}second")

                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots()
                    plt.axis("off")
                    
                    ax.add_patch(
                        patches.Rectangle(
                            (l, t),
                            w,
                            h,
                            edgecolor = 'blue',
                            facecolor = 'red',
                            fill=False
                        ) )                
                    for idx in range(len(label)):
                        x,y = label[idx]['x'], label[idx]['y']
                        x = l+x
                        y = t+y
                        plt.scatter(x,y,c='red', s=5)
                        plt.annotate(idx, (x,y))
                    
                    plt.imshow(image)
                    st.pyplot(fig = fig)          

                with col2:
                    df = pd.DataFrame(label, columns=["name","x","y"])
                    st.write(df)
            except AttributeError:
                st.write("이미지를 업로드 해주세요.")


    if submitted_facecrop:
        try:
            # Download part
            csv = convert_df(df)

            image = fig2img(fig)      
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="결과 이미지 다운로드",
                data =byte_im,
                file_name=f"{uploaded_file.name}_result.png",
                mime="image/png"            
            )
            st.download_button(
                label="결과 랜드마크 다운로드",
                data =csv,
                file_name=f"{uploaded_file.name}_result.csv",
                mime="text/csv"            
            )            
        except AttributeError:
            pass
if __name__ == "__main__":
    main()