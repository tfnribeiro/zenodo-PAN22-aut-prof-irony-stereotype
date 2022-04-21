from read_files import *
import emoji
import regex
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.65)

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

list_features = np.array(list_features)
kf = KFold(n_splits=4)

random_forest_list = []
one_nn_list = []
three_nn_list = []

for train_index, test_index in kf.split(X):
    X_train, X_test = list_features[train_index], list_features[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #X_train, X_test, y_train, y_test = train_test_split(list_features, y, test_size=0.3)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    acc_train = np.sum(y_train_pred == y_train) / len(y_train)
    acc_test = np.sum(y_test_pred == y_test) / len(y_test)
    random_forest_list.append((acc_train,acc_test))
    print(f"Random Forest Classifier acc (Train): {acc_train:.4f}")
    print(f"Random Forest Classifier acc (Test): {acc_test:.4f}")

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    acc_train = np.sum(y_train_pred == y_train) / len(y_train)
    acc_test = np.sum(y_test_pred == y_test) / len(y_test)
    three_nn_list.append((acc_train,acc_test))
    print(f"3-NN acc (Train): {acc_train:.4f}")
    print(f"3-NN (Test): {acc_test:.4f}")

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    acc_train = np.sum(y_train_pred == y_train) / len(y_train)
    acc_test = np.sum(y_test_pred == y_test) / len(y_test)
    one_nn_list.append((acc_train,acc_test))
    print(f"1-NN acc (Train): {acc_train:.4f}")
    print(f"1-NN (Test): {acc_test:.4f}")

random_forest_list = np.array(random_forest_list)
one_nn_list = np.array(one_nn_list)
three_nn_list = np.array(three_nn_list)


