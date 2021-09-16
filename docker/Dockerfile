FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

# install python3-pip
RUN apt update && apt install python3-pip -y

# install dependencies via pip
RUN pip3 install numpy scipy six wheel
RUN pip3 install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN apt-get install git -y
RUN pip3 install flatbuffers==1.12.0 gast==0.4.0 numpy==1.19.2 six==1.15.0 absl-py==0.10
RUN git clone https://github.com/belkakari/brax.git && cd brax && pip3 install -e .
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
