#!/bin/bash

echo "Creating conda environment: eims2vec"
conda create -n eims2vec python=3.11.5 -y

echo "Activating eims2vec environment"
conda activate eims2vec
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install pytorch::faiss-gpu
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "Environment setup complete!"