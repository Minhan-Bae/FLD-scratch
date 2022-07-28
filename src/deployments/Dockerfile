FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

COPY . /srv/deployments
WORKDIR /srv/deployments

RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list && apt-get clean
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx && apt-get -y install libglib2.0-0
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "./backend/main.py"]