#!/bin/bash
# Simple setup.sh for configuring Ubuntu 20.04 LTS EC2 instance

## Once connected to the EC2 instance:
# wget http://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda*.sh
# nano ~/.profile
# export PATH="$HOME/anaconda3_linux/bin:$PATH"
# exit / reconnect

## Once python is setup and set to PATH:
# wget https://github.com/mendelevium/Final-Project/setup.sh
# bash setup.sh

# Install python packages
conda install flask
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
conda install -c anaconda nltk
conda install -c conda-forge newspaper3k
conda install -c conda-forge scikit-learn
conda install pandas

## Finalise the app setup:
#git clone https://github.com/mendelevium/Final-Project
#cd Final-Project
#export FLASK_APP=nwsfx_flask_app.py
#flask run --host=0.0.0.0