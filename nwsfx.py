# general
import pandas as pd
import re
from collections import Counter
import statistics

# nlp
import newspaper
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import en_core_web_sm

# models
import joblib


# load pretrained models
nlp = en_core_web_sm.load()
sid = SentimentIntensityAnalyzer()

# load in house models
right_bias = joblib.load('models/right_bias_tfidf_svc_2000_acc_93_prob.pkl')
left_bias = joblib.load('models/left_bias_tfidf_svc_2000_acc_86_prob.pkl')
opinion_tfidf_svc = joblib.load('models/opinion_tfidf_svc_2000_acc_78_prob.pkl')


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
    pred_right_bias = right_bias.predict_proba(df['text'])
    pred_left_bias = left_bias.predict_proba(df['text'])

    m.append(
        {
            #'url': url,
            'opinion': int(pred_opinion[0]),
            'left_bias': round(pred_left_bias[0][1],4),
            'right_bias': round(pred_right_bias[0][1],4)
        })

    return m[0]

if __name__ == "__main__":
    print("run")
else:
    print("import")