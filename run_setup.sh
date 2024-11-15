#!/bin/bash

echo "Creating conda environment: eims2vec"
conda create -n eims2vec python=3.11.5 -y

echo "Activating eims2vec environment"
source activate
conda activate eims2vec

echo "Installing dependencies"
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# conda install pytorch::faiss-gpu -y
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-raft=1.8.0 -y
pip install rdkit
pip install matchms
echo "Environment setup complete!"