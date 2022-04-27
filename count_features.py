import emoji
import regex
import numpy as np
from utils import *

np.set_printoptions(suppress=True)
"""
Calculates a number of different counts per tweet and generates the following features:

[auth_vocabsize, type_token_rt, author_word_length_avg,
  avg_tweet_length, author_hashtag_count,
  author_usertag_count, author_urltag_count, 
  author_avg_emoji, avg_capital_lower_ratio]
  
"""
def author_style_counts(author_tweet_list):
    author_vocab = set()
    author_tweet_length = 0
    author_word_length_avg = 0
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
        emoji_counter = len([c for c in tweet if c in emoji.UNICODE_EMOJI['en']])
        author_total_emoji += emoji_counter
        tweet = tokenize_tweet(tweet, as_list=False)
        reg_filter = regex.findall(r'[A-Z]', tweet)
        tweet_capital_letters = len(reg_filter)
        reg_filter = regex.findall(r'[a-z]', tweet)
        tweet_lowercase_letters = len(reg_filter)
        author_tweet_length += len(tweet)
        word_length = [len(word) for word in tweet.split(" ")]
        vocab = {v for v in tweet.split(" ")}
        author_word_length_avg += sum(word_length)
        author_total_word_count += len(word_length)
        author_vocab = author_vocab | vocab
        if tweet_lowercase_letters == 0:
            author_total_capital_lower_ratio += 0
        else:
            author_total_capital_lower_ratio += tweet_capital_letters/tweet_lowercase_letters

    avg_tweet_length = author_tweet_length/len(author_tweet_list)
    author_avg_emoji = author_total_emoji/len(author_tweet_list)
    author_word_length_avg = author_word_length_avg/author_total_word_count
    type_token_rt = len(author_vocab)/author_total_word_count
    auth_vocabsize = len(author_vocab)
    avg_capital_lower_ratio = author_total_capital_lower_ratio/len(author_tweet_list)
    
    return np.array([auth_vocabsize, type_token_rt, author_word_length_avg, avg_tweet_length, author_hashtag_count, author_usertag_count, author_urltag_count, author_avg_emoji, avg_capital_lower_ratio])

def emoji_embeds(author_tweet_list):
    emoji_counts = {k:0 for k in emoji.UNICODE_EMOJI['en'].keys()}
    for tweet in author_tweet_list:
        tweet = tokenize_tweet(tweet)
        for c in tweet:
            if c in emoji_counts:
                emoji_counts[c] += 1
    return np.array(list(emoji_counts.values()))
