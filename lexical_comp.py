from utils import *
from nltk.tag import pos_tag
import numpy as np
import regex

"""
Avg. Tweet LiX complexity

https://en.wikipedia.org/wiki/Lix_(readability_test)

https://readable.com/readability/lix-rix-readability-formulas/

"""
def lix_score(author_tweet_list, long_words_t=7):
    total_LIX = 0
    for tweet in author_tweet_list:
        tokenized_tweet = tokenize_tweet(tweet, True)
        long_words = len([w for w in tokenized_tweet if len(w) >= long_words_t])
        # |[A-Z][a-z]+
        # RegEx to find the ends of sentences 
        reg_filter = regex.findall(r'[ ]+[.;?!]|[.;?!][ ]+|[.;?!]\n', " ".join(tokenized_tweet))
        periods = len(reg_filter)
        n_words = len(tokenized_tweet)
        if n_words == 0:
            n_words = 1
        if periods == 0:
            periods = 1
        total_LIX += (n_words/periods)+((long_words*100)/n_words)
    return np.array([total_LIX])/len(author_tweet_list)