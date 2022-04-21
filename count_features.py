from read_files import *
import emoji
import regex
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

np.set_printoptions(suppress=True)
# [VocabSize, Type/TokenRatio, Avg.WordLengh, TweetLength, EmojiCount, HashTagCount, UsetagCount]
def author_style_counts(author_tweet_list):
    author_vocab = set()
    author_tweet_length = 0
    author_word_length_avg = 0
    author_total_word_count = 0
    author_hashtag_count = 0
    author_usertag_count = 0
    author_total_emoji = 0
    for tweet in author_tweet_list:
        reg_filter = regex.findall(r'#USER#', tweet)
        author_usertag_count += len(reg_filter)
        reg_filter = regex.findall(r'#HASHTAG#', tweet)
        author_hashtag_count += len(reg_filter)
        emoji_counter = len([c for c in tweet if c in emoji.UNICODE_EMOJI['en']])
        author_total_emoji += emoji_counter
        tweet = tweet.replace("#USER#","")
        tweet = tweet.replace("#HASHTAG#","")
        author_tweet_length += len(tweet)
        word_length = [len(word) for word in tweet.split(" ")]
        vocab = {v for v in tweet.split(" ")}
        author_word_length_avg += sum(word_length)
        author_total_word_count += len(word_length)
        author_vocab = author_vocab | vocab

    avg_tweet_length = author_tweet_length/len(author_tweet_list)
    author_avg_emoji = author_total_emoji/len(author_tweet_list)
    author_word_length_avg = author_word_length_avg/author_total_word_count
    type_token_rt = len(author_vocab)/author_total_word_count
    auth_vocabsize = len(author_vocab)
    
    return np.array([auth_vocabsize, type_token_rt, author_word_length_avg, avg_tweet_length, author_hashtag_count, author_usertag_count, author_total_emoji, author_avg_emoji])

list_features = []
for i in range(len(X)):
    tweet_list = X[i]
    get_features = author_style_counts(tweet_list)
    print(y[i], get_features)
    list_features.append(get_features)

count_featues = np.array(list_features)