FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PROJECTDIR=/opt/app
WORKDIR $PROJECTDIR
COPY sam_vit_b_01ec64.pth requirements.txt $PROJECTDIR/
COPY inference_main.py sam_lora_image_encoder.py $PROJECTDIR/
COPY segment_anything $PROJECTDIR/segment_anything
COPY epoch_170.pth $PROJECTDIR/

# RUN pip3 install --upgrade pip
# Install Python dependencies from the requirements file
RUN pip3 install --no-cache-dir -r $PROJECTDIR/requirements.txt -v


RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

LABEL maintainer="vibe <vibe.research@outlook.com>"
ENTRYPOINT ["python", "inference_main.py"]
