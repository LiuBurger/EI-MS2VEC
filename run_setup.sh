#!/bin/bash

echo "Installing dependencies"
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# conda install pytorch::faiss-gpu -y
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-raft=1.8.0 -y
pip install rdkit
pip install matchms
echo "Environment setup complete!"