import emoji
import regex
import numpy as np
from nltk import FreqDist
from nltk import word_tokenize
import string

from sqlalchemy import true
from utils import *
from nltk.stem.porter import *
from tfidf import *


np.set_printoptions(suppress=True)
"""
Calculates a number of different counts per tweet and generates the following features:

[auth_vocabsize_avg, type_token_rt, author_word_length_avg,
  avg_tweet_length, author_hashtag_count,
  author_usertag_count, author_urltag_count, 
  author_avg_emoji, avg_capital_lower_ratio]
  
"""
author_style_feature_dict = {
    "avg_auth_vocabsize":0,
    "type_token_rt":0,
    "author_word_length_avg":0,
    "avg_tweet_length":0,
    "avg_author_hashtag_count":0,
    "avg_author_usertag_count":0, 
    "avg_author_urltag_count":0, 
    "avg_author_emoji_count":0, 
    "avg_capital_lower_ratio":0
}



def get_author_style_labels(filter_keys=[]):
    return [k for k in author_style_feature_dict.keys() if k not in filter_keys]

def author_style_counts(author_tweet_list, filter_keys=[]):
    author_vocab = set()
    author_tweet_length = 0
    avg_author_word_length = 0
    author_total_word_count = 0
    author_hashtag_count = 0
    author_usertag_count = 0
    author_urltag_count = 0
    author_total_emoji = 0
    author_total_capital_lower_ratio = 0
    for tweet in author_tweet_list:
        reg_filter = regex.findall(r'#USER#', tweet)
        author_usertag_count += len(reg_filter)
        reg_filter = regex.findall(r'#HASHTAG#', tweet)
        author_hashtag_count += len(reg_filter)
        reg_filter = regex.findall(r'#URL#', tweet)
        author_urltag_count += len(reg_filter)
        tweet = tokenize_tweet(tweet, as_list=False)
        emoji_counter = len([c for c in tweet if c in emoji.UNICODE_EMOJI['en']])
        author_total_emoji += emoji_counter
        reg_filter = regex.findall(r'[A-Z]', tweet)
        tweet_capital_letters = len(reg_filter)
        reg_filter = regex.findall(r'[a-z]', tweet)
        tweet_lowercase_letters = len(reg_filter)
        author_tweet_length += len(tweet)
        word_length = [len(word) for word in tweet.split(" ")]
        vocab = {v for v in tweet.split(" ")}
        avg_author_word_length += sum(word_length)
        author_total_word_count += len(word_length)
        author_vocab = author_vocab | vocab
        if tweet_lowercase_letters == 0:
            author_total_capital_lower_ratio += 0
        else:
            author_total_capital_lower_ratio += tweet_capital_letters/tweet_lowercase_letters

    author_features = copy_dictionary(author_style_feature_dict)
    author_features["avg_auth_vocabsize"] = len(author_vocab)/len(author_tweet_list)
    author_features["type_token_rt"] = len(author_vocab)/author_total_word_count
    author_features["author_word_length_avg"] = avg_author_word_length/author_total_word_count
    author_features["avg_tweet_length"] = author_tweet_length/len(author_tweet_list)
    author_features["avg_author_hashtag_count"] = author_hashtag_count/len(author_tweet_list)
    author_features["avg_author_usertag_count"] = author_usertag_count/len(author_tweet_list)
    author_features["avg_author_urltag_count"] = author_urltag_count/len(author_tweet_list)
    author_features["avg_author_emoji_count"] = author_total_emoji/len(author_tweet_list)
    author_features["avg_capital_lower_ratio"] = author_total_capital_lower_ratio/len(author_tweet_list)

    author_features = filter_dictionary(author_features, filter_keys)

    return np.array(list(author_features.values()))

def emoji_embeds(author_tweet_list):
    emoji_counts = {k:0 for k in emoji.UNICODE_EMOJI['en'].keys()}
    for tweet in author_tweet_list:
        tweet = tokenize_tweet(tweet)
        for c in tweet:
            if c in emoji_counts:
                emoji_counts[c] += 1
    total_emojis = sum(emoji_counts.values())
    if total_emojis == 0:
        return np.array(list(emoji_counts.values()))/1
    return np.array(list(emoji_counts.values()))/total_emojis

def fit_emoji_embeds_tfidf(train_data, author_documents_flag=False):
    emoji_tfidf = tfidf(train_data, terms_filter=set(emoji.UNICODE_EMOJI['en'].keys()), authors_document=author_documents_flag)
    return emoji_tfidf

def fit_word_embeds_tfidf(train_data, lower_case_flag=True, authors_document=False):
    word_tfidf = tfidf(train_data, lowercase=lower_case_flag, authors_document=authors_document)
    return word_tfidf

def fit_profanity_embeds_tfidf(train_data, lower_case_flag=True, authors_document=False):
    prof_list = [word.rstrip() for word in open('profanity_list.txt', 'r', encoding= 'utf-8').readlines()]
    prof_tfidf = tfidf(train_data, terms_filter=set(prof_list), lowercase=lower_case_flag, authors_document=authors_document)
    return prof_tfidf

def profanity_embeds(author_tweet_list):
    author_tweet_list = np.char.lower(author_tweet_list) #lowercase
    author_tweet_list = np.char.strip(author_tweet_list, string.punctuation) #stripping

    prof_list = [word.rstrip() for word in open('profanity_list.txt', 'r', encoding= 'utf-8').readlines()]
    rel_count_array = np.full((len(prof_list)), 0)

    freq = FreqDist()
    stemmer = PorterStemmer()
    for tweet in author_tweet_list:
        for word in word_tokenize(tweet):
            word = stemmer.stem(word)
            freq[word] += 1
    total_words = sum(freq.values())
    profanity = set(prof_list).intersection(set(freq.keys()))
    for word in profanity:
        if total_words == 0:
            rel_count_array[prof_list.index(word)] = freq[word]/1
        else:
            rel_count_array[prof_list.index(word)] = freq[word]/total_words
    return rel_count_array




