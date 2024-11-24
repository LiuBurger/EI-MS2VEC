# EI-MS2VEC

This is the implement of the paper **Deep Representation Learning for Electron Ionization Mass Spectra Retrieval**

Authors: Shibo Liu, Xuan Zhang, Anlei Jiao, Shiwei Sun, Longyang Dian*, Xuefeng Cui*

Contact: liuburger@qq.com

## Installation

The suggested python version is 3.11.5 and the cuda version is 12.1

### Get clone
#### 1.Clone this repository by:
    
    git clone https://github.com/xfcui/EI-MS2VEC.git

#### 2.Create virtual environment: 

    conda create -n eims2vec python=3.11.5 -y
    conda activate eims2vec

#### 3.Install packages:
    
    cd EI-MS2VEC
    chmod +x run_setup.sh
    bash ./run_setup.sh

## Quick start


### Get dataset
Get expanded in-silico library from: https://zenodo.org/records/14202417, and then put the lib under 'EI-MS2VEC/data/mine/'

You need to get the NIST17 (https://chemdata.nist.gov/dokuwiki/doku.php?id=chemdata:nist17) mainlib and replib by yourself, and follow the explore_nist17.ipynb to process the raw data to get test set (put it under 'EI-MS2VEC/data/nist17/'). 

The metadata of test set (https://github.com/Qiong-Yang/FastEI/blob/main/data/NEIMS_test_11499molecules.csv) is supplied by FastEI and the extra test set (https://github.com/Qiong-Yang/FastEI/tree/main/data/extra_test_set) is supplied too.

### Test hit rate by:

    conda activate eims2vec
    nohup python -Bu test.py >test.out
