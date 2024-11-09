# EI-MS2VEC

This is the implement of the paper **Deep Representation Learning for Electron Ionization Mass Spectra Retrieval**

Authors: Shibo Liu, Xuan Zhang, Anlei Jiao, Shiwei Sun, Longyang Dian*, Xuefeng Cui*

Contact: liuburger@qq.com

## Installation

The cuda version is 12.1

### Get clone
Clone this repository by:
    
    git clone https://github.com/xfcui/EI-MS2VEC.git

Create virtual environment and install packages:
    
    cd EI-MS2VEC
    chmod +x run_setup.sh
    bash ./run_setup.sh

## Quick start


### Get dataset
Get expanded in-silico library from:
https://zenodo.org/records/13968329

And then put the lib under 'EI-MS2VEC/data/mine/'

You need to get the NIST17 mainlib and replib by yourself, and follow the explore_nist17.ipynb to process the raw data.

Test hit rate by:

    conda activate eims2vec
    nohup python -Bu test.py >test.out
