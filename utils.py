from nltk.tokenize import TweetTokenizer 
import re

def tokenize_tweet(tweet_to_tokenize, as_list=False):
    tknzr = TweetTokenizer()
    tweet_to_tokenize = tweet_to_tokenize.replace("#USER#","")
    tweet_to_tokenize = tweet_to_tokenize.replace("#HASHTAG#","")
    tweet_to_tokenize = tweet_to_tokenize.replace("#URL#","")
    if as_list:
        return tknzr.tokenize(tweet_to_tokenize)
    return " ".join(tknzr.tokenize(tweet_to_tokenize))

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

def copy_dictionary(dic_to_copy):
    return {k:v for k, v in dic_to_copy.itmes()}

def filter_dictionary(dictionary, filter_list=[]):
    for keys_del in filter_list:
        del dictionary[keys_del]
    return dictionary