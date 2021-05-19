# NwsFx (NewsFix) 
Final project - Concordia Data Science Diploma (cb-ds-1)

The project is a complete implementation of an API that return a summary, bias metrics (left, right, opinion) and entities from any article urls. It is intended to be used as an input to a newsfeed.

Dataset: Quantifying News Media Bias through Crowdsourcing and Machine Learning Dataset (20k articles) [Source](https://deepblue.lib.umich.edu/data/concern/data_sets/8w32r569d?locale=en)

The dataset doesn't contain the text (only the url) so the articles must be scrape from 19 different sources.

## Prerequisites
```python
conda install flask
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
conda install -c anaconda nltk
#in python run: nltk.download('vader_lexicon')
conda install -c conda-forge newspaper3k
conda install -c conda-forge scikit-learn
conda install pandas
```

## How to run
- **Locally**
    - Install prerequisites
    - Run `python nwsfx_flask_app.py` in the terminal
    - Check the Demo notebook

- **AWS**
    - Create EC2 instance following the wizard
    - Create and Save SSH Key
    - EC2 > Instance > xxxx > Connect > SSH Client
    - in the SSH Client : follow instructions from AWS
    - or use Ec2 Instance Connect
    - wget http://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    - bash Miniconda*.sh
    - nano ~/.profile
    - export PATH="$HOME/anaconda3_linux/bin:$PATH"
    - exit / reconnect
    - conda install ...
    - git clone https://github.com/mendelevium/Final-Project
    - cd Final-Project
    - export FLASK_APP=nwsfx_flask_app.py
    - flask run --host=0.0.0.0

More info deployement in production on [codementor](https://www.codementor.io/@jqn/deploy-a-flask-app-on-aws-ec2-13hp1ilqy2)