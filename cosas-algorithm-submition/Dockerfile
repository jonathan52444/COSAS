FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PROJECTDIR=/opt/app
WORKDIR $PROJECTDIR
COPY inference_main.py checkpoint_199_512.pth requirements.txt $PROJECTDIR/
COPY PitSAM $PROJECTDIR/PitSAM

# RUN pip3 install --upgrade pip
# Install Python dependencies from the requirements file
# RUN pip3 install --no-cache-dir -r $PROJECTDIR/requirements.txt -v
RUN pip3 install numpy==1.21.6
RUN pip3 install einops==0.6.1
RUN pip3 install icecream==2.1.3
RUN pip3 install imageio==2.10.1
RUN pip3 install MedPy==0.4.0
RUN pip3 install nibabel==4.0.2
RUN pip3 install monai==1.1.0
RUN pip3 install opencv_python==4.5.4.58
RUN pip3 install pycocotools==2.0.6
RUN pip3 install safetensors==0.3.1
RUN pip3 install scipy==1.7.3
RUN pip3 install SimpleITK==2.2.1
RUN pip3 install tensorboardX==2.6
RUN pip3 install torch
RUN pip3 install torchvision
RUN pip3 install tqdm==4.62.3
RUN pip3 install ml-collections==0.1.1
RUN pip3 install onnx==1.13.1
RUN pip3 install onnxruntime==1.14.1
RUN pip3 install ftfy
RUN pip3 install regex
RUN pip3 install SimpleITK

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

LABEL maintainer="vibe <vibe.research@outlook.com>"
ENTRYPOINT ["python", "inference_main.py"]
