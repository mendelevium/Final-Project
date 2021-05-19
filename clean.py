# Ugly hack to prevent 

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


if __name__ == "__main__":
    print("run")
else:
    print("import")