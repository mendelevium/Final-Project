# general
import pandas as pd
import numpy as np
import re
from collections import Counter
import statistics

# preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin, BaseEstimator

# nlp
import newspaper
#import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
#import spacy
#from spacy import displacy
import en_core_web_sm

# models
import joblib
#from tensorflow import keras


# load pretrained models
#python -m spacy download en_core_web_sm
nlp = en_core_web_sm.load()
#nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()


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


# load in house models
#tfidf_fit = joblib.load('models/tfidf_fit.pkl')
#right_bias = keras.models.load_model('models/right_bias_model-FFNN_2000_1000_500_100-epochs_30_acc_91.h5')
#left_bias = keras.models.load_model('models/left_bias_model-FFNN_2000_1000_500_100-epochs_30_acc_85.h5')
right_bias = joblib.load('models/left_tfidf_svc_2000_acc_86.pkl')
left_bias = joblib.load('models/right_tfidf_svc_2000_acc_92.pkl')
opinion_tfidf_svc = joblib.load('models/opinion_tfidf_svc_2000_acc_78.pkl')


def get_text_from_url(url, preclean=True):

    article = newspaper.Article(url)
    try: 
        article.download()
        article.parse()
        #article.nlp()
        if preclean:
            text = article.text
            text = re.sub(r'<.*?>', '', text)
            text = re.sub('\\s', ' ', text)
            text = re.sub(' +', ' ', text)
    except:
        pass

    df = pd.DataFrame(columns=['text'])
    df.loc[0] = [text]
    return df


def get_summary(url):
    s = []
    article = newspaper.Article(url)
    try: 
        article.download()
        article.parse()
        article.nlp()
    except:
        pass

    s.append(
        {
            'url': url,
            'date': str(article.publish_date),
            'author': article.authors,
            'title': article.title,
            'summary': article.summary,
            'image': article.top_image
        })
    return s[0]


def get_sentences(text, words):
    # take text and test every sentences for a list of words
    # if any word from the list in a sentence, join thoses sentenses into a list
    if type(words) is list:
        sentenses = [sentence + '.' for sentence in text.split('. ') if any(w in sentence for w in words)]
    else:
        sentenses = [sentence + '.' for sentence in text.split('. ') if words in sentence]
    #print(sentenses)
    return sentenses


def get_sentiment(text, keyword=''):
    # get the sentiment of text or from sentences containning a keyword
    if keyword == '':
        sentiment = sid.polarity_scores(text)['compound']
    else:
        # get sentences where keyword is present 
        sentences = get_sentences(text, keyword)
        if len(sentences) != 0:
            # sentiment analysis
            sentiment = statistics.mean([sid.polarity_scores(s)['compound'] for s in sentences])
        else :
            sentiment = 0
    return round(sentiment,4)


def get_entities(url, n_labels=10):
    df = get_text_from_url(url)
    text = df['text'][0]
    # get list of entities
    labels = [x.text for x in nlp(text).ents]
    if len(Counter(labels)) < 10: 
        n_labels = len(Counter(labels))
    # count and sort
    entities = Counter(labels).most_common()[:n_labels]
    return { e : get_sentiment(text, keyword=e) for e, c in entities }


def get_metrics(url):
    m = []
    df = get_text_from_url(url)

    pred_opinion = opinion_tfidf_svc.predict(df['text'])
    
    #X = tfidf_fit.transform(df['text']).toarray()
    pred_right_bias = right_bias.predict(df['text'])
    pred_left_bias = left_bias.predict(df['text'])

    m.append(
        {
            #'url': url,
            'opinion': int(pred_opinion[0]),
            'left_bias': int(pred_left_bias[0]),
            'right_bias': int(pred_right_bias[0])
        })

    return m[0]

if __name__ == "__main__":
    print("person mod is run directly")
else:
    print("person mod is imported into another module")