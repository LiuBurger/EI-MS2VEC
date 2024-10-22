#!/bin/bash

echo "Creating conda environment: EIMS2VEC"
conda create -n EIMS2VEC -y

echo "Activating EIMS2VEC environment"
conda activate EIMS2VEC

if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt"
    pip install -r requirements.txt
else
    echo "requirements.txt not found! Please ensure the file is in the current directory."
    exit 1
fi

echo "Environment setup complete!"