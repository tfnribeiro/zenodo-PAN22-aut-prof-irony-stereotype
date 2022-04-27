from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from utils import *

def get_sent_polarity(user_tweet_list):
    sid = SentimentIntensityAnalyzer()
    polarity_vec = np.zeros(4)
    for tweet in user_tweet_list:
        tweet = tokenize_tweet(tweet)
        sent_dict = sid.polarity_scores(tweet)
        polarity_vec += np.array(list(sent_dict.values()))
    return polarity_vec/len(user_tweet_list)

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