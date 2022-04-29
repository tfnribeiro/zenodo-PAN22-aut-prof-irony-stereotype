from nltk.tokenize import word_tokenize
import re

def tokenize_tweet(tweet_to_tokenize, as_list=False):
    tweet_to_tokenize = tweet_to_tokenize.replace("#USER#","")
    tweet_to_tokenize = tweet_to_tokenize.replace("#HASHTAG#","")
    tweet_to_tokenize = tweet_to_tokenize.replace("#URL#","")
    if as_list:
        return word_tokenize(tweet_to_tokenize)
    return " ".join(word_tokenize(tweet_to_tokenize))

def tweet_to_wordlist(tweet):
    tweet = tweet.encode("ascii", "ignore").decode()
    tweet = tweet.replace("#USER#","")
    tweet = tweet.replace("#HASHTAG#","")
    tweet = tweet.replace("#URL#","")
    tweet = re.sub('(“|”)', '"', tweet)
    tweet = re.sub("(’|‘)", "'", tweet)
    tweet = re.sub("( '|' |…|&amp;)", "", tweet)
    tweet = re.sub('(\.|\?|\!|"|,|\(|\)|:)', "", tweet)
    

    return tweet.split()