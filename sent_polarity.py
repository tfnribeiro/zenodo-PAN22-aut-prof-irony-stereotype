from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from utils import *

sentiment_dict = {
    'pos':0.0,
    'compound':0.0,
    'neu':0.0,
    'neg':0.0
}

def get_sent_labels(filter_keys=[]):
    return [k for k in sentiment_dict.keys() if k not in filter_keys]

def get_sent_polarity(user_tweet_list, filter_keys=[]):
    sid = SentimentIntensityAnalyzer()
    author_sent_dict = copy_dictionary(sentiment_dict)
    polarity_vec = np.zeros((len(user_tweet_list),4))
    for tweet_i, tweet in enumerate(user_tweet_list):
        tweet = tokenize_tweet(tweet)
        sent_dict = sid.polarity_scores(tweet)
        polarity_vec[tweet_i,:] = np.array(list(sent_dict.values()))
    for i, k in enumerate(author_sent_dict.keys()):
        author_sent_dict[k] = polarity_vec[i]

    author_sent_dict = filter_dictionary(author_sent_dict, filter_keys)
    features = np.array(list(author_sent_dict.values())).std(axis=0)
    return features

"""
Best Parameters:  (40, 0.01) Acc:  0.6666666666666666
Test Results:
Correct:86 Total:126 | Acc:  0.6825396825396826

Current best params:  (35, 0.001)
Best Parameters:  (35, 0.001) Acc:  0.6996336996336996
93 147 Acc:  0.6326530612244898

Current best params:  (42.5, 0.0121)
Best Parameters:  (42.5, 0.0121) Acc:  0.6703296703296703
101 147 Acc:  0.6870748299319728

"""