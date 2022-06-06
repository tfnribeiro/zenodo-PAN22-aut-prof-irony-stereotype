import regex
import numpy as np
from utils import *
import gensim
import gensim.downloader as api
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string

np.set_printoptions(suppress=True)

EMB_MODEL = api.load('glove-twitter-50')
# RandomForest with just embeddings on 20% test: 0.7976190476190477
# 50 seems to be the best performer.

def tweet_word_embs(author_tweet_list, filter_keys=[]):
    author_tweet_embeds = []
    for i, tweet in enumerate(author_tweet_list):
        tweet_tokens = tokenize_tweet(tweet.lower())
        tweet_tokens = remove_stopwords(tweet_tokens)
        tweet_tokens = tweet_tokens.split(" ")
        tweet_vectors = np.array([EMB_MODEL[w] for w in tweet_tokens if w in EMB_MODEL])
        if len(tweet_vectors) == 0:
            print("WARNING: Empty vector.")
            continue
        author_tweet_embeds.append(tweet_vectors.mean(axis=0))
    return np.array(author_tweet_embeds).mean(axis=0)
