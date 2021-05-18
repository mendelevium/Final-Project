from flask import Flask, request
import json

APP_NAME = 'nwsfx_api'
app = Flask(APP_NAME)
HTTP_ERROR_CLIENT = 400
HTTP_ERROR_SERVER = 500
EXPECTED_KEYS = ['url']

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from nltk.stem import WordNetLemmatizer

class Lemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y):
        return self

    def transform(self, X):
        return  X.apply(lambda text: " ".join([self.lemmatizer.lemmatize(word) for word in text.split()]))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub('\\s', ' ', text)
    text = re.sub("[^a-zA-Z' ]", "", text)
    text = re.sub(' +', ' ', text)
    #text = text.split(' ')
    return text

tfidf = TfidfVectorizer(
    stop_words="english",
    preprocessor=clean_text,
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=2,
    max_features=2000
)

import nwsfx 

def validate_json(j):
    """
    Validate that the input has the expected format
    """
    try:
        # Make input into a python dict
        if not isinstance(j, dict):
            d = json.loads(j)
        else:
            d = j
    except Exception as e:
        raise ValueError(e)
    for ek in EXPECTED_KEYS:
        if ek not in d:
            raise ValueError(
                f"Expected key {ek} not in JSON\n{j}"
            )
    return d

@app.route('/', methods=['GET'])
def model_info():
    """
    Returns expected input format
    """
    return str(
        """
        Available functions:
            - summary: date, author, title, summary, image
            - entities: top 10 entities and their sentiments
            - metrics: return opinion (0 or 1), left and rigth bias

        Expected JSON input:
        {
            "url" : "https://www.example.com"
        }
        """
    )

@app.route('/summary', methods=['POST'])
def summary_main():

    try:
        # This gets the data field in the post request
        j = validate_json(request.data)
        #j = json.loads(request.data)
        # Return a JSON back out
        return json.dumps(nwsfx.get_summary(j['url']))
    except ValueError as ex:  # failed schema/values validation
        return json.dumps({ "Incorrect JSON format:\n": str(ex)}), HTTP_ERROR_CLIENT
    except Exception as ex:
        return json.dumps({ "Server Error:\n": str(ex)}), HTTP_ERROR_SERVER


@app.route('/metrics', methods=['POST'])
def metrics_main():

    try:
        j = validate_json(request.data)
        #j = json.loads(request.data)
        return json.dumps(nwsfx.get_metrics(j['url']))
    except ValueError as ex:  # failed schema/values validation
        return json.dumps({ "Incorrect JSON format:\n": str(ex)}), HTTP_ERROR_CLIENT
    except Exception as ex:
        return json.dumps({ "Server Error:\n": str(ex)}), HTTP_ERROR_SERVER


@app.route('/entities', methods=['POST'])
def entities_main():

    try:
        j = validate_json(request.data)
        #j = json.loads(request.data)
        return json.dumps(nwsfx.get_entities(j['url']))
    except ValueError as ex:  # failed schema/values validation
        return json.dumps({ "Incorrect JSON format:\n": str(ex)}), HTTP_ERROR_CLIENT
    except Exception as ex:
        return json.dumps({ "Server Error:\n": str(ex)}), HTTP_ERROR_SERVER


if __name__ == '__main__':
    # debug should be False in production
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)