#!/bin/bash

echo "Creating conda environment: eims2vec"
conda create -n eims2vec python=3.11.5 -y

echo "Activating eims2vec environment"
source activate
conda activate eims2vec

echo "Installing dependencies"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install pytorch::faiss-gpu -y
pip install rdkit
pip install matchms
echo "Environment setup complete!"