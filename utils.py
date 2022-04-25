from nltk.tokenize import word_tokenize

def tokenize_tweet(tweet_to_tokenize, as_list=False):
    tweet_to_tokenize = tweet_to_tokenize.replace("#USER#","")
    tweet_to_tokenize = tweet_to_tokenize.replace("#HASHTAG#","")
    tweet_to_tokenize = tweet_to_tokenize.replace("#URL#","")
    if as_list:
        return word_tokenize(tweet_to_tokenize)
    return " ".join(word_tokenize(tweet_to_tokenize))